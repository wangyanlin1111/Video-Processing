from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import sys
import multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("qwen_tts").setLevel(logging.ERROR)
logging.getLogger("qwen_tts.core").setLevel(logging.ERROR)
logging.getLogger("qwen_tts.core.models.configuration_qwen3_tts").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
for _logger_name in ("vllm", "vllm_omni"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

import math
import time
import gc
import torchaudio
import queue
import threading

import numpy as np
import torch as th
import soundfile as sf
import torchaudio.transforms as T

from typing import Any
from vllm_omni import Omni
from vllm_omni.utils.tracking_parser import TrackingArgumentParser
from commonfunc import converttime, get_error_detail

RED = "\033[91m"    
GREEN = "\033[92m"  
YELLOW = "\033[93m"  
BLUE = "\033[94m"  
PUPPLE = "\033[95m"  
RESET = "\033[0m"

Qwen3TTSModel_SAMPLE_RATE = 24000   #24KHz
Advance_Time = 100 / 1000           #100ms
Advance_SAMPLE = int(Advance_Time * Qwen3TTSModel_SAMPLE_RATE)


def _estimate_prompt_len(
    additional_information: dict[str, Any],
    model_name: str,
    _cache: dict[str, Any] = {},
) -> int:
    """Estimate prompt_token_ids placeholder length for the Talker stage.

    The AR Talker replaces all input embeddings via ``preprocess``, so the
    placeholder values are irrelevant but the **length** must match the
    embeddings that ``preprocess`` will produce.
    """
    try:
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
        from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
            Qwen3TTSPromptEmbedsBuilder,
        )

        if model_name not in _cache:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
            cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)

            # Load speech tokenizer (codec encoder) for exact ref_code_len.
            speech_tok = None
            try:
                import os

                from transformers.utils import cached_file

                from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSTokenizer

                st_cfg_path = cached_file(model_name, "speech_tokenizer/config.json")
                if st_cfg_path:
                    speech_tok = Qwen3TTSTokenizer.from_pretrained(
                        os.path.dirname(st_cfg_path), torch_dtype=th.bfloat16
                    )
            except Exception as e:
                logger.debug("Could not load speech tokenizer: %s", e)

            _cache[model_name] = (tok, getattr(cfg, "talker_config", None), speech_tok)

        tok, tcfg, speech_tok = _cache[model_name]
        task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]

        def _estimate_ref_code_len(ref_audio: object) -> int | None:
            """Encode ref_audio with the actual codec to get exact frame count."""
            if not isinstance(ref_audio, (str, list)):
                return None
            audio_path = ref_audio[0] if isinstance(ref_audio, list) else ref_audio
            if not isinstance(audio_path, str) or not audio_path.strip():
                return None
            try:
                from urllib.parse import urlparse

                import numpy as np

                def _is_url(path: str) -> bool:
                    try:
                        parsed = urlparse(path)
                        if parsed.scheme in ("http", "https"):
                            return bool(parsed.netloc)
                        return parsed.scheme in ("file", "data")
                    except Exception:
                        return False

                if _is_url(audio_path):
                    from vllm.multimodal.media import MediaConnector

                    connector = MediaConnector(allowed_local_media_path="/")
                    audio, sr = connector.fetch_audio(audio_path)
                else:
                    from vllm.multimodal.media.audio import load_audio

                    audio, sr = load_audio(audio_path, sr=None, mono=True)

                wav_np = np.asarray(audio, dtype=np.float32)

                if speech_tok is not None:
                    enc = speech_tok.encode(wav_np, sr=int(sr), return_dict=True)
                    ref_code = getattr(enc, "audio_codes", None)
                    if isinstance(ref_code, list):
                        ref_code = ref_code[0] if ref_code else None
                    if ref_code is not None and hasattr(ref_code, "shape"):
                        shape = ref_code.shape
                        return int(shape[0]) if len(shape) == 2 else int(shape[1]) if len(shape) == 3 else None

                # Fallback: estimate from duration
                codec_hz = getattr(tcfg, "codec_frame_rate", None) or 12
                return int(len(audio) / sr * codec_hz)
            except Exception:
                return None

        return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
            additional_information=additional_information,
            task_type=task_type,
            tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
            codec_language_id=getattr(tcfg, "codec_language_id", None),
            spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
            estimate_ref_code_len=_estimate_ref_code_len,
        )
    except Exception as exc:
        logger.warning("Failed to estimate prompt length, using fallback 2048: %s", exc)
        return 2048

def parse_args(args=None):
    parser = TrackingArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="Base",
        help="Query type.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable writing detailed statistics (default: disabled)",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing a single stage in seconds (default: 300)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=5,
        help="Timeout for batching in seconds (default: 5)",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing stages in seconds (default: 300)",
    )
    parser.add_argument(
        "--shm-threshold-bytes",
        type=int,
        default=65536,
        help="Threshold for using shared memory in bytes (default: 65536)",
    )
    parser.add_argument(
        "--output-dir",
        default="output_audio",
        help="Output directory for generated wav files (default: output_audio).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line (preferred).",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to a stage configs file.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file. If not provided, uses default audio asset.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate for audio loading (default: 16000).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory (default: logs).",
    )
    parser.add_argument(
        "--py-generator",
        action="store_true",
        default=False,
        help="Use py_generator mode. The returned type of Omni.generate() is a Python Generator object.",
    )
    parser.add_argument(
        "--use-batch-sample",
        action="store_true",
        default=False,
        help="Use batch input sample for CustomVoice/VoiceDesign/Base query.",
    )
    parser.add_argument(
        "--mode-tag",
        type=str,
        default="xvec_only",
        choices=["icl", "xvec_only"],
        help="Mode tag for Base query x_vector_only_mode (default: icl).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Stream audio chunks as they arrive via AsyncOmni (async_chunk mode only).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts per batch (default: 1, sequential).",
    )

    return parser.parse_args(args=args)

class VoiceSynthesisVLLMQwen3TTS:
    def __init__(self, batchsize: int = None):
        self.VOCALPATH = "vocal_clone.mp3"
        self.BATCHSIZE = batchsize if batchsize is not None else 16
        self.DEVICE = "cuda" if th.cuda.is_available() else "cpu"
        self.DTYPE = th.bfloat16 if self.DEVICE == "cuda" else th.float32
        self.MODELNAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base" if self.DEVICE == "cuda" else "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        self.SAMPLERATE = Qwen3TTSModel_SAMPLE_RATE
        self.calibrationsampler = T.Resample(orig_freq=self.SAMPLERATE, new_freq=self.SAMPLERATE).to(self.DEVICE)
        self.SYNTHESISMODEL = None
        self.VOICECLONEPROMPT = None
        self._resampler_cache = {}
        self._silence_cache = {}
        self._args = parse_args(args=[])
        self._stdout_handler_added = False
        if self.DEVICE == "cuda":
            free, total = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}Voice Synthesis Allocated: All: {total / 1024 ** 3:.2f} GB, Free: {free / 1024 ** 3:.2f} GB{RESET}")

    def get_resampler(self, target_freq: int) -> T.Resample:
        """Acquire Resampler on self.DEVICE"""
        key = (target_freq, self.SAMPLERATE)
        if key not in self._resampler_cache:
            resampler = T.Resample(
                orig_freq=self.SAMPLERATE,
                new_freq=target_freq,
                resampling_method='sinc_interp_kaiser',
                lowpass_filter_width=64,
                rolloff=0.95,
                beta=14.0
            ).to(self.DEVICE)
            self._resampler_cache[key] = resampler
        return self._resampler_cache[key]

    def generate_silence(self, duration_insamples):
        """Generate silence from duration(duration in seconds)"""
        key = int(duration_insamples)
        if key not in self._silence_cache:
            self._silence_cache[key] = th.zeros(key, dtype=self.DTYPE, device=self.DEVICE)
        return self._silence_cache[key].clone()

    def get_base_query(
            self,
            mode_tag: str = "xvec_only", 
            ref_audio: str = None, 
            ref_text: str = None,
            syn_text_batch: list = None
        ):
        """Build Base (voice clone) sample inputs.
        Args:
            use_batch_sample: When True, return a batch of prompts (Case 2).
            mode_tag: "icl" or "xvec_only" to control x_vector_only_mode behavior.
        """

        inputs = []
        for idx, text in enumerate(syn_text_batch):
            additional_information = {
                "task_type": ["Base"],
                "ref_audio": [ref_audio],
                "ref_text": [ref_text],
                "text": [text],
                "language": ["Chinese"],
                "x_vector_only_mode": [mode_tag == "xvec_only"],
                "max_new_tokens": [2048]
            }
            inputs.append(
                {                    
                    "request_id": f"idx_{idx}",                    
                    "prompt_token_ids": [0] * _estimate_prompt_len(additional_information, self.MODELNAME),
                    "additional_information": additional_information,
                }
            )
        return inputs

    def build_inputs(self, ref_audio_path, ref_text, syn_text_batch):
        """Resolve model name and inputs list from CLI args."""
        if self._args.batch_size < 1 or (self._args.batch_size & (self._args.batch_size - 1)) != 0:
            raise ValueError(
                f"--batch-size must be a power of two (got {self._args.batch_size}); "
                "non-power-of-two values do not align with CUDA graph capture sizes "
                "of Code2Wav."
            )
        inputs = self.get_base_query(mode_tag=self._args.mode_tag, ref_audio=ref_audio_path, ref_text=ref_text, syn_text_batch=syn_text_batch)
        inputs = inputs if isinstance(inputs, list) else [inputs]
        return inputs

    def synthesize_batch(self, omni, ref_audio_path, ref_text, syn_text_batch):
        try:
            inputs = self.build_inputs(ref_audio_path, ref_text, syn_text_batch)
            indexed_audios = []  # (idx, audio_tensor)
            """Pass ALL inputs at once so vLLM can batch them internally"""
            for stage_outputs in omni.generate(inputs):
                output = stage_outputs.request_output
                mm = output.outputs[0].multimodal_output
                request_id = output.request_id
                idx = int(request_id.split("_")[0])
                audio_data = mm["audio"]
                audio_tensor = th.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
                audio_tensor = audio_tensor.to(self.DEVICE, self.DTYPE)
                """Extract index from request metadata to restore original order"""
                indexed_audios.append((idx, audio_tensor))
            """Sort by original index and return just the tensors"""
            indexed_audios.sort(key=lambda x: x[0])
            audios = [audio for _, audio in indexed_audios]
            return audios
        except Exception as e:
            error_detail = get_error_detail(e)
            """ Write to both stderr (fd 2 may be redirected) and via logger """
            print(f"{RED}{error_detail}{RESET}", file=sys.stderr, flush=True)
            logger.error(error_detail)
            raise RuntimeError(f"Single generation failed") from e
        
    def apply_speed_adjustment(self, audio_tensor: np.ndarray, speed: float):
        """Apply speed adjustment to the audio tensor while preserving pitch.

        Uses torchaudio's phase vocoder (Spectrogram → TimeStretch →
        InverseSpectrogram) to stretch/compress audio in time without
        changing pitch.
        """
        if speed == 1.0:
            return audio_tensor

        try:
            if not np.issubdtype(audio_tensor.dtype, np.floating):
                audio_tensor = audio_tensor.astype(np.float32)

            # Stereo numpy arrays use channels-last (T, C);
            # torch expects channels-first (C, T).
            channels_last = audio_tensor.ndim == 2
            if channels_last:
                waveform = th.from_numpy(audio_tensor.T)
            else:
                waveform = th.from_numpy(audio_tensor).unsqueeze(0)

            # Match librosa.stft defaults: n_fft=2048, hop_length=n_fft//4
            n_fft = 512
            hop_length = n_fft // 4
            to_spec = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                power=None,
            )
            stretch = torchaudio.transforms.TimeStretch(
                n_freq=n_fft // 2 + 1,
                hop_length=hop_length,
            )
            to_wave = torchaudio.transforms.InverseSpectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
            )

            spec = to_spec(waveform)
            stretched = stretch(spec, speed)
            expected_length = int(audio_tensor.shape[0] / speed)
            result = to_wave(stretched, length=expected_length)

            result = result.squeeze(0).numpy()
            if channels_last:
                result = result.T
            return result
        except Exception as e:
            logger.error(f"An error occurred during speed adjustment: {e}")
            raise ValueError("Failed to apply speed adjustment.") from e

    def synthesize_parallel(
            self, 
            total_sentences, 
            sampling_rate, 
            ori_len, 
            ori_sr, 
            debug_en : bool=False, 
            vocal_path : str=None,
            ref_audio : str=None,
            ref_text:str=None,
        ):
        """Route logger to stdout before stderr is redirected to /dev/null"""
        if not self._stdout_handler_added:
            _stdout_handler = logging.StreamHandler(sys.stdout)
            _stdout_handler.setLevel(logging.INFO)
            _stdout_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(_stdout_handler)
            self._stdout_handler_added = True

        """ -- Redirect stderr at the OS file-descriptor level -------------------"""
        _stderr_fd = sys.stderr.fileno()
        _saved_fd = os.dup(_stderr_fd)
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, _stderr_fd)
        try:
            total_start_time = time.time()
            prev_end_sample = 0
            N = len(total_sentences)
            iteration_num = math.ceil(N / self.BATCHSIZE)
            omni_kwargs = vars(self._args).copy()
            omni_kwargs["model"] = self.MODELNAME
            omni_kwargs["batch-size"] = self.BATCHSIZE
            omni_kwargs["use-batch-sample"] = True
            omni_kwargs["enable_prompt_embed_cache"] = True
            omni_kwargs["prompt_embed_cache_size"] = 256
            omni_kwargs["quantization"] = "fp8"
            omni_kwargs["tts_batch_max_items"] = self.BATCHSIZE
            omni = Omni(**omni_kwargs)
            init_end_time = time.time()
            init_hours,init_minutes,init_seconds,init_miliseconds = converttime(init_end_time - total_start_time)
            logger.info(f"{GREEN}Initialization Time is {init_hours} hours, {init_minutes} minutes, {init_seconds} seconds and {init_miliseconds} ms{RESET}")
            if debug_en == True:
                file = open('regeneration_log.txt', 'w')
            num_advance = 0
            advance_flag = 0

            _vocal_path = vocal_path if vocal_path is not None else self.VOCALPATH
            if os.path.isfile(_vocal_path):
                os.remove(_vocal_path)
            total_len = 0

            write_queue = queue.Queue()
            def writer(sf_file):
                while True:
                    item = write_queue.get()
                    if item is None:
                        break
                    sf_file.write(item)

            with sf.SoundFile(_vocal_path, 'w', samplerate=sampling_rate, channels=2) as f:    
                writer_thread = threading.Thread(target=writer, args=(f,), daemon=True)
                writer_thread.start()
                for i in range(iteration_num):
                    audio_segment = th.tensor([], dtype = self.DTYPE, device = self.DEVICE)
                    start_idx = i * self.BATCHSIZE
                    end_idx = min((i + 1)  * self.BATCHSIZE, N)
                    proc_num = end_idx - start_idx
                    index_mask = np.arange(start_idx, end_idx)
                    texts = [total_sentences[j]["chinese"].strip() for j in index_mask]
                    """Synthesize audios in batch and calibrate the audio length according to the time difference"""
                    audios = self.synthesize_batch(omni, ref_audio, ref_text, texts)
                    current_length = np.array([len(audio) for audio in audios])
                    target_length = []
                    for j in index_mask:
                        real_duration = total_sentences[j]["end"] - total_sentences[j]["start"]
                        correction_term = (total_sentences[j+1]["start"] - total_sentences[j]["end"]) * 0.95 if j != (N - 1) else 0
                        target_length.append(round((real_duration + correction_term) * self.SAMPLERATE))
                    target_length = np.array(target_length)
                    regenerate_flag = current_length > target_length
                    calibration_mask = index_mask[regenerate_flag] - start_idx
                    advance_samples = [round((total_sentences[flag]["start"] - total_sentences[flag - 1]["end"]) * self.SAMPLERATE) if flag > 0 else 0 for flag in index_mask[regenerate_flag]]
                    advance_flag = np.zeros(proc_num, dtype=int)
                    """If the generated audio is longer than the target length, then calibrate the audio by time-stretching and speed up the audio according to the time difference"""
                    if len(calibration_mask) > 0:
                        for idx, k in enumerate(calibration_mask):
                            real_calibration_factor = math.ceil(current_length[k] / target_length[k] * 100) / 100
                            if (advance_samples[idx] > 0):
                                advance_flag[idx] = 1
                                calibration_factor = math.ceil(current_length[k] / (target_length[k] + advance_samples[idx]) * 100) / 100
                            else:
                                calibration_factor = real_calibration_factor
                            audio_tensor = audios[k].squeeze().cpu().to(th.float32).numpy()
                            stretched_audio_np = self.apply_speed_adjustment(audio_tensor, speed=calibration_factor)
                            audios[k] = th.from_numpy(stretched_audio_np).to(self.DEVICE, self.DTYPE)
                            
                            if debug_en == True:
                                content = f"The {k+start_idx}th data, Calibration Factor = {real_calibration_factor}, Target = {target_length[k]}, Original = {current_length[k]}, Updated = {len(audios[k])}"+"\n"
                                file.write(content)
                    """Concatenate the audio segment with silence according to the time difference and write the audio segment into local file in each iteration to avoid OOM, which is caused by concatenating all audio segments in memory and writing once in the end"""
                    for j in range(proc_num):
                        start_sample = round(total_sentences[index_mask[j]]["start"] * self.SAMPLERATE)
                        silence_samples = 0 if advance_flag[j] == 1 else max(start_sample - prev_end_sample, 0)
                        if silence_samples > 0:
                            temp_silence = self.generate_silence(silence_samples)
                            audio_segment = th.concatenate([audio_segment, temp_silence])
                            del temp_silence
                        audio_segment = th.concatenate([audio_segment, audios[j]])
                        prev_end_sample += (silence_samples + len(audios[j]))
                    """Resample the audio segment to the target sampling rate if needed and write the audio segment into local file"""
                    if sampling_rate != self.SAMPLERATE:
                        resampler = self.get_resampler(target_freq = sampling_rate)
                        audio = resampler(audio_segment.float())
                    else:
                        audio = audio_segment
                    total_len += len(audio)

                    if audio.dim() == 1:
                        audio = audio.repeat(2, 1)
                    write_queue.put(audio.cpu().numpy().T)
                    del audios, audio_segment, audio
                write_queue.put(None)
                """"After writing all audio segments, if the total length of the audio is shorter than the target length, then pad zeros at the end of the audio"""
                target_samples = int(round(ori_len * sampling_rate))
                if int(total_len * ori_sr) < target_samples:
                    numzeros = int(round(target_samples / ori_sr)) - int(total_len)
                    paddimgzeros = np.zeros((numzeros,2), dtype=np.float32)
                    write_queue.put(paddimgzeros)
                writer_thread.join()

                if debug_en == True:
                    file.close()

            total_end_time = time.time()
            hours,minutes,seconds,miliseconds = converttime(total_end_time - total_start_time)
            logger.info(f"{GREEN}Synthesis Time is {hours} hours, {minutes} minutes, {seconds} seconds and {miliseconds} ms{RESET}")
            omni.close()
            gc.collect()
            if self.DEVICE == "cuda":
                th.cuda.synchronize()
                th.cuda.empty_cache()
                free, _ = th.cuda.mem_get_info()
                logger.info(f"{YELLOW}After Synthesis Model Sleep Free: {free / 1024 ** 3:.2f} GB{RESET}")
        finally:
            os.dup2(_saved_fd, _stderr_fd)
            os.close(_saved_fd)
            os.close(_devnull_fd)
            