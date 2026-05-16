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
import threading
import queue
import torchaudio

import numpy as np
import torch as th
import torchaudio.transforms as T

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

    def __init__(self, segment_duration: int = None):
        """
        :param segment_duration: segment length in minutes
        """
        init_start_time = time.time()
        self.DEVICE = "cuda" if th.cuda.is_available() else "cpu"
        self.MODEL_NAME = "mdx_extra"
        self.OUT_VOCALS = "vocal_original.mp3"
        self.OUT_BG = "background.mp3"
        self.segment_duration = segment_duration * 60 if segment_duration is not None else 30 * 60  #Default length is 30 minutes
        self.overlapsecond = 1
        init_end_time = time.time()
        logger.info(f"{GREEN}Audio Separation Init Time: {init_end_time - init_start_time:.4f}s{RESET}")
        if self.DEVICE == "cuda":
            free, total = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}Audio Separation: All: {total / 1024 ** 3:.2f} GB, Free: {free / 1024 ** 3:.2f} GB{RESET}")

    def separate_long_audio(self, audio_path: str, output_bg: str, output_vocal: str = None):

        total_start_time = time.time()

        self.MODEL = get_model(self.MODEL_NAME).to(self.DEVICE)
        """Acquire audio information"""
        info = torchaudio.info(audio_path)
        total_frames = info.num_frames
        orig_sr = info.sample_rate
        self.total_duration = total_frames / orig_sr
        num_segments = int(np.ceil(self.total_duration / self.segment_duration))

        """Create queues: loader→processor→saver pipeline"""
        load_queue = queue.Queue(maxsize=2)      # Loader → Processor
        process_queue = queue.Queue(maxsize=2)   # Processor → Saver
        
        """Start 3 worker threads"""
        loader = threading.Thread(
            target=self._loader_worker,
            args=(audio_path, orig_sr, num_segments, load_queue)
        )
        processor = threading.Thread(
            target=self._processor_worker,
            args=(load_queue, orig_sr, process_queue)
        )
        saver = threading.Thread(
            target=self._saver_worker,
            args=(process_queue, output_bg, output_vocal)
        )
        
        loader.start()
        processor.start()
        saver.start()
        
        # Waiting for the jobs
        saver.join()
        processor.join()
        loader.join()

        total_end_time = time.time()
        hours,minutes,seconds,miliseconds = converttime(total_end_time - total_start_time)
        logger.info(f"{GREEN}Audio Separation Time is {hours} hours, {minutes} minutes, {seconds} seconds and {miliseconds} ms{RESET}")
        return total_frames, orig_sr

    def _loader_worker(self, audio_path, original_sr, num_segments, load_queue):
    
        for seg_idx in range(num_segments):
            start_sec = max(0, seg_idx * self.segment_duration - self.overlapsecond)
            end_sec = min((seg_idx + 1) * self.segment_duration + self.overlapsecond, self.total_duration)
            
            # Loading audio segments
            wav, _ = torchaudio.load(
                audio_path, 
                frame_offset=int(start_sec * original_sr),
                num_frames=int((end_sec - start_sec) * original_sr)
            )        
            load_queue.put((seg_idx, wav, int(start_sec * original_sr), int(end_sec * original_sr)))

        load_queue.put(None)
    
    def _processor_worker(self, load_queue, original_sr, process_queue):
        # Runs on GPU - processes audio segments
        # Only one _processor_worker exists, so GPU is naturally protected
        # from running multiple files' processing simultaneously
        while True:
            item = load_queue.get()
            if item is None:
                process_queue.put(None)
                break
                
            seg_idx, wav, valid_start, valid_end = item
            # Convert audio on CPU first, then move to GPU (safer)
            wav = convert_audio(wav, original_sr, self.MODEL.samplerate, self.MODEL.audio_channels)
            wav = wav.unsqueeze(0).to(self.DEVICE)
            
            with th.no_grad():
                sources = apply_model(
                    self.MODEL, 
                    wav,
                    shifts=0, 
                    split=True, 
                    overlap=0.25,
                    progress=False
                )[0]
                        
            drums, bass, other, vocal = sources
            bg = drums + bass + other
            vocal = _normalize_audio(vocal)
            bg = _normalize_audio(bg)
            
            vocal_cpu = vocal.detach().cpu()
            bg_cpu = bg.detach().cpu()
            process_queue.put((seg_idx, vocal_cpu, bg_cpu, valid_start, valid_end))
            
            del wav, sources, drums, bass, other, vocal, bg
            if self.DEVICE == "cuda":
                th.cuda.empty_cache()

    def _saver_worker(self, process_queue, output_bg, output_vocal):
        # Runs on CPU - saves and crossfades segments
        accumulated_vocal = None
        accumulated_bg = None
        first_seg = True

        while True:
            item = process_queue.get()
            if item is None:
                break

            _, vocal, bg, valid_start, valid_end = item

            if first_seg:
                # 第一段直接保存（包含尾部重叠部分，用于后续交叉淡变）
                accumulated_vocal = vocal
                accumulated_bg = bg
                first_seg = False
            else:
                # 与前一段的重叠部分做交叉淡变（渐进式合并）
                # 相邻两段实际重叠为 2 * overlapsecond 秒
                samplerate = self.MODEL.samplerate
                overlap_samples = int(self.overlapsecond * samplerate)
                crossfade_len = 2 * overlap_samples  # 实际重叠长度

                # 确保有足够的样本做淡变
                actual_len = min(crossfade_len, accumulated_vocal.shape[-1], vocal.shape[-1])

                if actual_len > 0:
                    # 对累积音频尾部做淡出
                    fade_out = T.Fade(fade_out_len=actual_len, fade_shape='half_sine')
                    faded_tail_vocal = fade_out(accumulated_vocal[:, -actual_len:])
                    faded_tail_bg = fade_out(accumulated_bg[:, -actual_len:])

                    # 对当前段头部做淡入
                    fade_in = T.Fade(fade_in_len=actual_len, fade_shape='half_sine')
                    faded_head_vocal = fade_in(vocal[:, :actual_len])
                    faded_head_bg = fade_in(bg[:, :actual_len])

                    # 交叉淡变后的重叠区
                    crossfaded_vocal = faded_tail_vocal + faded_head_vocal
                    crossfaded_bg = faded_tail_bg + faded_head_bg

                    # 拼接：非重叠尾部 + 交叉淡变区 + 非重叠头部
                    accumulated_vocal = th.cat([
                        accumulated_vocal[:, :-actual_len],
                        crossfaded_vocal,
                        vocal[:, actual_len:]
                    ], dim=-1)
                    accumulated_bg = th.cat([
                        accumulated_bg[:, :-actual_len],
                        crossfaded_bg,
                        bg[:, actual_len:]
                    ], dim=-1)
                else:
                    # 无重叠，直接拼接
                    accumulated_vocal = th.cat([accumulated_vocal, vocal], dim=-1)
                    accumulated_bg = th.cat([accumulated_bg, bg], dim=-1)

                # 释放当前段张量
                del vocal, bg
                gc.collect()

        if accumulated_vocal is None or accumulated_bg is None:
            logger.warning("No audio data to save!")
            return

        # 裁剪末尾多出的 overlap 部分（最后一段的尾部重叠）
        trim_samples = int(self.overlapsecond * self.MODEL.samplerate)
        if accumulated_vocal.shape[-1] > trim_samples:
            accumulated_vocal = accumulated_vocal[:, :-trim_samples]
            accumulated_bg = accumulated_bg[:, :-trim_samples]

        # 保存
        if output_bg is not None:
            if os.path.isfile(output_bg):
                os.remove(output_bg)
            torchaudio.save(output_bg, accumulated_bg, samplerate)
        if output_vocal is not None:
            if os.path.isfile(output_vocal):
                os.remove(output_vocal)
            torchaudio.save(output_vocal, accumulated_vocal, samplerate)

        del accumulated_vocal, accumulated_bg
        gc.collect()