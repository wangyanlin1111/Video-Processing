from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import os
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

import numpy as np
import torch as th
import torchaudio.transforms as T
import pyrubberband as pyrb

from qwen_tts import Qwen3TTSModel
from commonfunc import converttime

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
    def __init__(self, batchsize: int = None, instruction: str = None):
        default_instruction = instruction if instruction is not None else "35岁男性, 播音腔, 颗粒感强, 语速很快，句子间间隔短"
        init_start_time = time.time()
        self.VOCALPATH = "vocal_clone.mp3"
        self.BATCHSIZE = batchsize if batchsize is not None else 10
        self.REFINSTRUCTION = default_instruction
        self.DEVICE = "cuda" if th.cuda.is_available() else "cpu"
        self.DTYPE = th.bfloat16 if self.DEVICE == "cuda" else th.float32
        self.MODELNAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base" if self.DEVICE == "cuda" else "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        self.ATTENTION = "flash_attention_2" if self.DEVICE == "cuda" else "eager"
        self.SAMPLERATE = Qwen3TTSModel_SAMPLE_RATE
        self.calibrationsampler = T.Resample(orig_freq=self.SAMPLERATE, new_freq=self.SAMPLERATE).to(self.DEVICE)
        """First Design the reference audio for voice clone"""
        design_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
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
        with th.no_grad():
            wav, _ = self.SYNTHESISMODEL.generate_voice_clone(
                text=text, 
                language="Chinese",
                voice_clone_prompt=self.VOICECLONEPROMPT, 
                speed=speed
            )
        return th.from_numpy(wav[0]).squeeze().to(self.DEVICE, self.DTYPE)
    
    @th.no_grad()
    def synthesize_batch(self, texts, speed=1.0):
        audios = []
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
        return audios
    
    def synthesize(self, total_sentences, sampling_rate, ori_len, ori_sr, debug = 0, vocal_path:str=None):

        total_start_time = time.time()

        full_audio = th.tensor([], dtype = self.DTYPE, device = self.DEVICE)
        prev_end_sample = 0
        N = len(total_sentences)
        iteration_num = math.ceil(N / self.BATCHSIZE)
        if debug == 1:
            file = open('regeneration_log.txt', 'w')
        num_advance = 0
        advance_flag = 0
        for i in range(iteration_num):
            start_idx = i * self.BATCHSIZE
            end_idx = min((i + 1)  * self.BATCHSIZE, N)
            proc_num = end_idx - start_idx
            index_mask = np.arange(start_idx, end_idx)
            texts = [total_sentences[j]["chinese"].strip() for j in index_mask]
            audios = self.synthesize_batch(texts)
            current_length = np.array([len(audio) for audio in audios])
            target_length = []
            for j in index_mask:
                real_duration = total_sentences[j]["end"] - total_sentences[j]["start"]
                correction_term = (total_sentences[j+1]["start"] - total_sentences[j]["end"]) * 0.7 if j != (N - 1) else 0
                target_length.append(round((real_duration + correction_term) * self.SAMPLERATE))
            target_length = np.array(target_length)
            regenerate_flag = current_length > target_length
            calibration_mask = index_mask[regenerate_flag] - start_idx
            if len(calibration_mask) > 0:
                for k in calibration_mask:
                    real_calibration_factor = math.ceil(current_length[k] / target_length[k] * 100) / 100
                    # """If calibration_factor is less than 1.25, then speed up the original synthesised vocal"""
                    # """Else keep the original synthesised vocal and advance the next few sentences 100ms each according to the time difference"""
                    # if calibration_factor < 1.25:
                    #     speed = torchaudio.transforms.Speed(orig_freq=self.SAMPLERATE, factor=calibration_factor).to(self.DEVICE)
                    #     stretched_audio, _ = speed(audios[k].float())
                    #     stretched_audio = self.calibrationsampler(stretched_audio.float())
                    #     audios[k] = stretched_audio
                    # else:
                    #     difference_samples = current_length[k] - target_length[k]
                    #     num_advance += int(difference_samples // Advance_SAMPLE)
                    #     advance_flag = 1
                    """If calibration_factor is less than 1.25, then speed up the original synthesised vocal"""
                    """Else speed up 1.25x and advance the next few sentences 100ms each according to the time difference"""
                    calibration_factor = min(real_calibration_factor, 1.25)
                    #####################################################################################################
                    # audio_np = audios[k].squeeze().cpu().to(th.float32).numpy()
                    # stretched_audio_np = pyrb.time_stretch(audio_np, self.SAMPLERATE, rate = calibration_factor)
                    # audios[k] = th.from_numpy(stretched_audio_np).to(self.DEVICE, self.DTYPE)
                    #####################################################################################################
                    speed = T.Speed(orig_freq=self.SAMPLERATE, factor=calibration_factor).to(self.DEVICE)
                    stretched_audio, _ = speed(audios[k].float())
                    audios[k] = stretched_audio
                    #####################################################################################################
                    # n_fft = 2048
                    # win_length = 1536   
                    # hop_length = 384       
                    # spectrogram = T.Spectrogram(
                    #     n_fft=n_fft,
                    #     win_length=win_length,
                    #     hop_length=hop_length,
                    #     window_fn=th.hann_window
                    # ).to(self.DEVICE)
                    # transform = T.InverseSpectrogram(
                    #     n_fft=n_fft,
                    #     win_length=win_length,
                    #     hop_length=hop_length,
                    #     window_fn=th.hann_window
                    # ).to(self.DEVICE)
                    # stretch = T.TimeStretch(
                    #     hop_length=hop_length,  
                    #     n_freq=None
                    # ).to(self.DEVICE)
                    # original = spectrogram(audios[k])
                    # stretched_spectrogram = stretch(original, calibration_factor)
                    # stretched_audio = transform(stretched_spectrogram)
                    if (calibration_factor < real_calibration_factor):
                        difference_samples = len(audios[k]) - target_length[k]
                        num_advance += int(difference_samples // Advance_SAMPLE)
                        advance_flag = 1
                    
                    if debug == 1:
                        content = f"The {k+start_idx}th data, Calibration Factor = {calibration_factor}, Target = {target_length[k]}, Original = {current_length[k]}, Updated = {len(audios[k])}"+"\n"
                        file.write(content)
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
            if self.DEVICE == "cuda":
                gc.collect()
                th.cuda.empty_cache()
                th.cuda.synchronize()
        
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