from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("qwen_tts").setLevel(logging.ERROR)
logging.getLogger("qwen_tts.core").setLevel(logging.ERROR)
logging.getLogger("qwen_tts.core.models.configuration_qwen3_tts").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
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
import pyrubberband as pyrb

from qwen_tts import Qwen3TTSModel
from commonfunc import converttime, get_error_detail

RED = "\033[91m"    
GREEN = "\033[92m"  
YELLOW = "\033[93m"  
BLUE = "\033[94m"  
PUPPLE = "\033[95m"  
RESET = "\033[0m"

Qwen3TTSModel_SAMPLE_RATE = 24000   #24KHz
Advance_Time = 50 / 1000           #50ms
Advance_SAMPLE = int(Advance_Time * Qwen3TTSModel_SAMPLE_RATE)
default_text = "我们的星球已经经历了数十亿年的各种灾难性事件，而这些事件才使得地球变成了我们现在所生活的这个样子。"
default_language = "Chinese"

class VoiceSynthesis:
    def __init__(self, batchsize: int = None, gen_model_path: str = None, syn_model_path: str = None, instruction: str = None):
        default_instruction = instruction if instruction is not None else "45岁男性, 低沉而有磁性, 颗粒感强, 语速非常快，句子间间隔短"
        init_start_time = time.time()
        self.VOCALPATH = "vocal_clone.mp3"
        self.BATCHSIZE = batchsize if batchsize is not None else 10
        self.REFINSTRUCTION = default_instruction
        self.DEVICE = "cuda" if th.cuda.is_available() else "cpu"
        self.DTYPE = th.bfloat16 if self.DEVICE == "cuda" else th.float32
        if syn_model_path is not None:
            self.MODELNAME = syn_model_path 
        else:
            self.MODELNAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base" if self.DEVICE == "cuda" else "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        _gen_model_name_path = gen_model_path if gen_model_path is not None else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        self.ATTENTION = "flash_attention_2" if self.DEVICE == "cuda" else "eager"
        self.SAMPLERATE = Qwen3TTSModel_SAMPLE_RATE
        self.calibrationsampler = T.Resample(orig_freq=self.SAMPLERATE, new_freq=self.SAMPLERATE).to(self.DEVICE)
        """First Design the reference audio for voice clone"""
        design_model = Qwen3TTSModel.from_pretrained(
            _gen_model_name_path,
            device_map = self.DEVICE,
            dtype = self.DTYPE,
            attn_implementation = self.ATTENTION,
        )
        ref_wavs, sr = design_model.generate_voice_design(
            text = default_text,
            language = default_language,
            instruct = default_instruction
        )
        """Empty Cache"""
        del design_model
        gc.collect()
        th.cuda.empty_cache()
        """Then generate the synthesis model"""
        self.SYNTHESISMODEL = Qwen3TTSModel.from_pretrained(
            self.MODELNAME,
            device_map = self.DEVICE,
            dtype = self.DTYPE,
            attn_implementation= self.ATTENTION
        )
        self.VOICECLONEPROMPT = self.SYNTHESISMODEL.create_voice_clone_prompt(
            ref_audio=(ref_wavs[0], sr),   
            ref_text=default_text,
        )
        init_end_time = time.time()
        logger.info(f"{GREEN}Voice Synthesis Init Time: {init_end_time - init_start_time:.4f}s{RESET}")
        self._resampler_cache = {}
        if self.DEVICE == "cuda":
            free, total = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}Voice Synthesis Allocated: All: {total / 1024 ** 3:.2f} GB, Free: {free / 1024 ** 3:.2f} GB{RESET}")
        self.occupied_mem =  (total - free) / 1024 ** 3 

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
        return th.zeros(int(duration_insamples), dtype = self.DTYPE, device=self.DEVICE)

    @th.no_grad()
    def synthesize_single(self, text, speed=1.0):
        try:
            with th.no_grad():
                wav, _ = self.SYNTHESISMODEL.generate_voice_clone(
                    text=text, 
                    language="Chinese",
                    voice_clone_prompt=self.VOICECLONEPROMPT, 
                    speed=speed
                )
                if th.cuda.is_available():
                    th.cuda.empty_cache()
            return th.from_numpy(wav[0]).squeeze().to(self.DEVICE, self.DTYPE)
        except Exception as e:
            error_detail = get_error_detail(e)
            logger.error(f"{RED}{error_detail}{RESET}")
            raise RuntimeError(f"Single generation failed")
    
    @th.no_grad()
    def synthesize_batch(self, texts, speed=1.0):
        audios = []
        try:
            with th.no_grad():
                wavs, _ = self.SYNTHESISMODEL.generate_voice_clone(
                    text=texts,
                    language="Chinese",
                    voice_clone_prompt = self.VOICECLONEPROMPT,
                    speed=speed
                )
                for w in wavs:
                    audio = th.from_numpy(w).to(self.DEVICE, self.DTYPE)
                    audios.append(audio)
                del wavs
                if th.cuda.is_available():
                    th.cuda.empty_cache()
                return audios
        except Exception as e:
            error_detail = get_error_detail(e)
            logger.error(f"{RED}{error_detail}{RESET}")
            raise RuntimeError(f"Batch generation failed")
            
    def synthesize(self, total_sentences, sampling_rate, ori_len, ori_sr, debug_en:bool=False, vocal_path:str=None):

        total_start_time = time.time()
        full_audio = th.tensor([], dtype = self.DTYPE, device = self.DEVICE)
        prev_end_sample = 0
        N = len(total_sentences)
        iteration_num = math.ceil(N / self.BATCHSIZE)
        if debug_en == True:
            file = open('regeneration_log.txt', 'w')
        num_advance = 0
        advance_flag = 0
        for i in range(iteration_num):
            start_idx = i * self.BATCHSIZE
            end_idx = min((i + 1)  * self.BATCHSIZE, N)
            proc_num = end_idx - start_idx
            index_mask = np.arange(start_idx, end_idx)
            texts = [total_sentences[j]["chinese"].strip() for j in index_mask]
            audios = self.synthesize_batch(texts, speed=1.25)
            current_length = np.array([len(audio) for audio in audios])
            target_length = []
            for j in index_mask:
                real_duration = total_sentences[j]["end"] - total_sentences[j]["start"]
                correction_term = (total_sentences[j+1]["start"] - total_sentences[j]["end"]) * 0.85 if j != (N - 1) else 0
                target_length.append(round((real_duration + correction_term) * self.SAMPLERATE))
            target_length = np.array(target_length)
            regenerate_flag = current_length > target_length
            calibration_mask = index_mask[regenerate_flag] - start_idx
            if len(calibration_mask) > 0:
                for k in calibration_mask:
                    real_calibration_factor = math.ceil(current_length[k] / target_length[k] * 100) / 100
                    """If calibration_factor is less than 1.15, then speed up the original synthesised vocal"""
                    """Else speed up 1.25x and advance the next few sentences 100ms each according to the time difference"""
                    calibration_factor = min(real_calibration_factor, 1.15)
                    audio_np = audios[k].squeeze().cpu().to(th.float32).numpy()
                    stretched_audio_np = pyrb.time_stretch(audio_np, self.SAMPLERATE, rate = calibration_factor)
                    audios[k] = th.from_numpy(stretched_audio_np).to(self.DEVICE, self.DTYPE)
                    if (calibration_factor < real_calibration_factor):
                        difference_samples = len(audios[k]) - target_length[k]
                        num_advance += int(difference_samples // Advance_SAMPLE)
                        advance_flag = 1
                    
                    if debug_en == True:
                        content = f"The {k+start_idx}th data, Calibration Factor = {calibration_factor}, Target = {target_length[k]}, Original = {current_length[k]}, Updated = {len(audios[k])}"+"\n"
                        file.write(content)
            ###################################################################################################
            for j in range(proc_num):
                start_sample = round(total_sentences[index_mask[j]]["start"] * self.SAMPLERATE)
                silence_samples = max(start_sample - prev_end_sample, 0)
                if advance_flag == 1 & silence_samples > Advance_SAMPLE:
                    silence_samples -= Advance_SAMPLE
                    num_advance -= 1
                    advance_flag = 1 if num_advance > 0 else 0
                temp_silence = self.generate_silence(silence_samples)
                full_audio = th.concatenate([full_audio, temp_silence])
                del temp_silence  
                full_audio = th.concatenate([full_audio, audios[j]])
                prev_end_sample += (silence_samples + len(audios[j]))

            """Empty Cache"""
            del audios
            gc.collect()
            if self.DEVICE == "cuda":
                th.cuda.empty_cache()
                # th.cuda.synchronize()
        if debug_en == True:
            file.close()
        if sampling_rate != self.SAMPLERATE:
            resampler = self.get_resampler(target_freq = sampling_rate)
            audio = resampler(full_audio.float())
        else:
            audio = full_audio

        if int(round(len(audio) * ori_sr)) < int(round((sampling_rate * ori_len))):
            missingzeros = int(round(sampling_rate * ori_len / ori_sr)) - len(audio)
            audio = th.nn.functional.pad(audio, (0, missingzeros))

        if audio.dim() == 1:
            audio = audio.repeat(2, 1)

        _vocal_path = vocal_path if vocal_path is not None else self.VOCALPATH
        if os.path.isfile(_vocal_path):
            os.remove(_vocal_path)
        torchaudio.save(_vocal_path, audio.cpu(), sampling_rate)

        """Empty Cache"""
        del full_audio, audio
        gc.collect()
        if self.DEVICE == "cuda":
            th.cuda.synchronize()
            th.cuda.empty_cache()
            free, _ = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}After Vocal Synthesis Free: {free / 1024 ** 3:.2f} GB{RESET}")
        total_end_time = time.time()
        hours,minutes,seconds,miliseconds = converttime(total_end_time - total_start_time)
        logger.info(f"{GREEN}Synthesis Time is {hours} hours, {minutes} minutes, {seconds} seconds and {miliseconds} ms{RESET}")


    def synthesize_parallel(self, total_sentences, sampling_rate, ori_len, ori_sr, debug_en:bool=False, vocal_path:str=None):

        total_start_time = time.time()
        prev_end_sample = 0
        N = len(total_sentences)
        iteration_num = math.ceil(N / self.BATCHSIZE)
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
                audios = self.synthesize_batch(texts, speed=1.2)
                current_length = np.array([len(audio) for audio in audios])
                target_length = []
                for j in index_mask:
                    real_duration = total_sentences[j]["end"] - total_sentences[j]["start"]
                    correction_term = (total_sentences[j+1]["start"] - total_sentences[j]["end"]) * 0.85 if j != (N - 1) else 0
                    target_length.append(round((real_duration + correction_term) * self.SAMPLERATE))
                target_length = np.array(target_length)
                regenerate_flag = current_length > target_length
                calibration_mask = index_mask[regenerate_flag] - start_idx

                """If the generated audio is longer than the target length, then calibrate the audio by time-stretching and speed up the audio according to the time difference"""
                if len(calibration_mask) > 0:
                    for k in calibration_mask:
                        real_calibration_factor = math.ceil(current_length[k] / target_length[k] * 100) / 100
                        """If calibration_factor is less than 1.15, then speed up the original synthesised vocal"""
                        """Else speed up 1.25x and advance the next few sentences 100ms each according to the time difference"""
                        calibration_factor = min(real_calibration_factor, 1.15)
                        audio_np = audios[k].squeeze().cpu().to(th.float32).numpy()
                        stretched_audio_np = pyrb.time_stretch(audio_np, self.SAMPLERATE, rate = calibration_factor)
                        audios[k] = th.from_numpy(stretched_audio_np).to(self.DEVICE, self.DTYPE)
                        if (calibration_factor < real_calibration_factor):
                            difference_samples = len(audios[k]) - target_length[k]
                            num_advance += int(difference_samples // Advance_SAMPLE)
                            advance_flag = 1
                        
                        if debug_en == True:
                            content = f"The {k+start_idx}th data, Calibration Factor = {calibration_factor}, Target = {target_length[k]}, Original = {current_length[k]}, Updated = {len(audios[k])}"+"\n"
                            file.write(content)
                """Concatenate the audio segment with silence according to the time difference and write the audio segment into local file in each iteration to avoid OOM, which is caused by concatenating all audio segments in memory and writing once in the end"""
                for j in range(proc_num):
                    start_sample = round(total_sentences[index_mask[j]]["start"] * self.SAMPLERATE)
                    silence_samples = max(start_sample - prev_end_sample, 0)
                    if advance_flag == 1 and silence_samples > Advance_SAMPLE:
                        silence_samples -= Advance_SAMPLE
                        num_advance -= 1
                        advance_flag = 1 if num_advance > 0 else 0
                    temp_silence = self.generate_silence(silence_samples)
                    audio_segment = th.concatenate([audio_segment, temp_silence])
                    del temp_silence  
                    gc.collect()
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
                """Empty Cache"""
                del audios, audio_segment, audio
                gc.collect()
                if self.DEVICE == "cuda":
                    th.cuda.empty_cache()
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

        gc.collect()
        if self.DEVICE == "cuda":
            th.cuda.synchronize()
            th.cuda.empty_cache()
            free, _ = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}After Vocal Synthesis Free: {free / 1024 ** 3:.2f} GB{RESET}")
        total_end_time = time.time()
        hours,minutes,seconds,miliseconds = converttime(total_end_time - total_start_time)
        logger.info(f"{GREEN}Synthesis Time is {hours} hours, {minutes} minutes, {seconds} seconds and {miliseconds} ms{RESET}")