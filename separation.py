import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import time
import gc
import torch as th
import torch.nn.functional as F
import torchaudio

from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs.pretrained import get_model
from commonfunc import converttime

RED = "\033[91m"    
GREEN = "\033[92m"  
YELLOW = "\033[93m"  
BLUE = "\033[94m"  
PUPPLE = "\033[95m"  
RESET = "\033[0m"

def _normalize_audio(tensor):
    max_val = th.max(th.abs(tensor))
    if max_val > 1.0:
        tensor = tensor / max_val
    return tensor

class AudioSeparation:

    def __init__(self, device: str = None, model_name: str = None, save_flag: bool = False):
        """
        :param device: Default value is None
        :param model_name: Default value is mdx_extra
        :param audio_path: Default value is None
        :param print_flag: Default value is False
        """
        init_start_time = time.time()
        if device is not None:
            self.DEVICE = device
        else:
            self.DEVICE = "cuda" if th.cuda.is_available() else "cpu"
        self.MODEL_NAME = model_name if model_name is not None else "mdx_extra"
        self.SAVE_FLAG = save_flag if save_flag is not None else False
        self.MODEL = get_model(self.MODEL_NAME)
        self.OUT_VOCALS = "vocal_original.mp3"
        self.OUT_BG = "background.mp3"
        init_end_time = time.time()
        logger.info(f"{GREEN}Audio Separation Init Time: {init_end_time - init_start_time:.4f}s{RESET}")
        if self.DEVICE == "cuda":
            free, total = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}Audio Separation: All: {total / 1024 ** 3:.2f} GB, Free: {free / 1024 ** 3:.2f} GB{RESET}")
        self.occupied_mem =  (total - free) / 1024 ** 3  


    def audio_separate(self, audio_path:str = None, bgm_path:str = None, vocal_path:str = None):
        total_start_time = time.time()
        _AUDIO_PATH = audio_path if audio_path is not None else ""
        """Converting audio"""
        if not os.path.exists(_AUDIO_PATH):
            raise FileNotFoundError(f"{RED}Audio file not found: {_AUDIO_PATH}{RESET}")
        try:
            wav, sr = torchaudio.load(_AUDIO_PATH)
            original_len = wav.shape[1]
            origianl_sr = sr
        except Exception as e:
            raise RuntimeError(f"{RED}Failed to load audio: {str(e)}{RESET}")

        """Standardization"""
        wav = convert_audio(wav.to(self.DEVICE), sr, self.MODEL.samplerate, self.MODEL.audio_channels)
        wav = wav.unsqueeze(0)
        demcus_sr = self.MODEL.samplerate

        """Separate Vocal and Background"""
        with th.no_grad():
            sources = apply_model(
                self.MODEL,
                wav,
                shifts = 0,
                split = True,
                overlap = 0.25,
                progress = False
            )[0]
        drums, bass, other, vocal = sources
        background = drums + bass + other
        
        """Normalization to 1 and truncate it to original length"""
        vocal = _normalize_audio(vocal)
        background = _normalize_audio(background)
        if ((vocal.shape[1] * sr) < original_len * demcus_sr):
            missingzeros = int(round(original_len * demcus_sr / sr)) - vocal.shape[1]
            vocal = F.pad(vocal, (0, missingzeros))
            background = F.pad(background, (0, missingzeros))

        """Make sure the vocal is Single Channel"""
        vocal_return = vocal.clone().to(self.DEVICE)

        """Saving to files"""
        _bgm_path = bgm_path if bgm_path is not None else self.OUT_BG
        if os.path.isfile(_bgm_path):
            os.remove(_bgm_path)
        torchaudio.save(_bgm_path, background.detach().cpu(), self.MODEL.samplerate)
        if self.SAVE_FLAG:
            _vocal_path = vocal_path if vocal_path is not None else self.OUT_VOCALS
            if os.path.isfile(_vocal_path):
                os.remove(_vocal_path)
            torchaudio.save(_vocal_path, vocal.detach().cpu(), self.MODEL.samplerate)

        """Clear Memory"""
        del wav, sources, drums, bass, other, vocal, background
        gc.collect()
        if self.DEVICE == "cuda":
            th.cuda.empty_cache()
            th.cuda.synchronize()
            free, _ = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}After Audio Separation Free: {free / 1024 ** 3:.2f} GB{RESET}")
        total_end_time = time.time()
        hours,minutes,seconds,miliseconds = converttime(total_end_time - total_start_time)
        logger.info(f"{GREEN}Audio Separation Time is {hours} hours, {minutes} minutes, {seconds} seconds and {miliseconds} ms{RESET}")
        return vocal_return, demcus_sr, original_len, origianl_sr