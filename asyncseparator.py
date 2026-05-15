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
import torch.nn.functional as F
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

def crossfade_2d(signal_a, signal_b, fade_len_samples, fade_shape:str = None):
    """
    对两个信号（形状相同，可能是 [C, T]）的重叠部分进行交叉淡变
    - signal_a: 前一段，取其最后 fade_len_samples 个样本
    - signal_b: 后一段，取其前 fade_len_samples 个样本
    返回混合后的重叠段
    """
    _fade_shape = fade_shape if fade_shape is not None else 'half_sine'
    if fade_len_samples <= 0:
        return signal_b[:, :fade_len_samples]
    a_len = signal_a.shape[-1]
    b_len = signal_b.shape[-1]
    if a_len < fade_len_samples or b_len < fade_len_samples:
        return signal_b[:, :min(fade_len_samples, b_len)]
    head_transform = T.Fade(fade_in_len=fade_len_samples, fade_shape=_fade_shape)
    tail_transform = T.Fade(fade_out_len=fade_len_samples, fade_shape=_fade_shape)
    faded_a = tail_transform(signal_a[:, -fade_len_samples:])[:, -fade_len_samples:]
    faded_b = head_transform(signal_b[:, :fade_len_samples])[:, :fade_len_samples]
    return faded_a + faded_b

class AudioSeparation:

    def __init__(self, segment_duration: int = None):
        """
        :param segment_duration: segment length in minutes
        """
        init_start_time = time.time()
        self.DEVICE = "cuda" if th.cuda.is_available() else "cpu"
        self.MODEL_NAME = "mdx_extra"
        self.MODEL = get_model(self.MODEL_NAME)
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
        """Acquire audio information"""
        info = torchaudio.info(audio_path)
        total_frames = info.num_frames
        orig_sr = info.sample_rate
        self.total_duration = total_frames / orig_sr
        num_segments = int(np.ceil(self.total_duration / self.segment_duration))

        """Create 3 queues"""
        load_queue = queue.Queue(maxsize=2)      
        process_queue = queue.Queue(maxsize=2)   
        save_queue = queue.Queue(maxsize=2)      
        
        """Mark event"""
        finish_event = threading.Event()
        
        """Start 3 queues"""
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
        
        while True:
            item = load_queue.get()
            if item is None:
                process_queue.put(None)
                break
                
            seg_idx, wav, valid_start, valid_end = item
            wav = convert_audio(wav.to(self.DEVICE), original_sr, self.MODEL.samplerate, self.MODEL.audio_channels)
            wav = wav.unsqueeze(0)
            
            # 执行分离
            with th.no_grad():
                sources = apply_model(
                    self.MODEL, 
                    wav,
                    shifts=0, 
                    split=True, 
                    overlap=0.25,
                    progress = False
                )[0]
                        
            # 立即释放GPU内存
            drums, bass, other, vocal = sources
            bg = drums + bass + other
            vocal = _normalize_audio(vocal)
            bg = _normalize_audio(bg)
            
            # 保存结果（移回CPU）
            vocal_cpu = vocal.detach().cpu()
            bg_cpu = bg.detach().cpu()
            process_queue.put((seg_idx, vocal_cpu, bg_cpu, valid_start, valid_end))
            
            # 清理
            del wav, sources, drums, bass, other, vocal, bg
            if self.DEVICE == "cuda":
                th.cuda.empty_cache()

    def _saver_worker(
            self, 
            process_queue, 
            output_bg, 
            output_vocal
        ): 
        vocal_parts = []
        bg_parts = []
        first_seg = True
        vocal, bg = None, None

        while True:
            item = process_queue.get()
            if item is None:
                break
            _, vocal, bg, valid_start, valid_end = item  

            load_start = valid_start - self.overlapsecond

            if first_seg:
                # 第一段直接保存
                vocal_parts.append(vocal)
                bg_parts.append(bg)
                first_seg = False
            else:
                # 与前一段重叠部分做交叉淡变
                # 假设 vocal 和 prev_vocal 的形状一致，前一段已经保存在 vocal_parts[-1] 中
                prev_vocal = vocal_parts[-1]
                prev_bg = bg_parts[-1]
                overlap_samples = self.MODEL.samplerate * self.overlapsecond
                overlap_len = min(overlap_samples, prev_vocal.shape[1], vocal.shape[1])
                if overlap_len <= 0:
                    # 无重叠，直接拼接
                    new_vocal = th.cat([prev_vocal, vocal], dim=-1)
                    new_bg = th.cat([prev_bg, bg], dim=-1)
                else:
                    # 交叉淡变重叠区
                    vocal_overlap = crossfade_2d(prev_vocal, vocal, overlap_len, fade_shape='half_sine')
                    bg_overlap = crossfade_2d(prev_bg, bg, overlap_len, fade_shape='half_sine')
                    # 更新前一段：去掉原来的重叠尾部，加上新的重叠区
                    new_vocal = th.cat([
                        prev_vocal[:, :-2*overlap_len],
                        vocal_overlap,
                        vocal[:, 2*overlap_len:]
                    ], dim=-1)
                    new_bg = th.cat([
                        prev_bg[:, :-2*overlap_len],
                        bg_overlap,
                        bg[:, 2*overlap_len:]
                    ], dim=-1)
                    # 替换最后一段为新的完整段
                    vocal_parts[-1] = new_vocal
                    bg_parts[-1] = new_bg
                del prev_vocal, prev_bg, vocal, bg  # 释放前序张量
                gc.collect()

        # 最终所有段已经拼接在 vocal_parts[0] 中（因为我们是渐进式更新）
        final_vocal = vocal_parts[0] if vocal_parts else None
        final_bg = bg_parts[0] if bg_parts else None
        # 保存
        if output_bg is not None:
            if os.path.isfile(output_bg):
                os.remove(output_bg)
            torchaudio.save(output_bg, final_bg.cpu(), self.MODEL.samplerate)
        if output_vocal is not None:    
            if os.path.isfile(output_vocal):
                os.remove(output_vocal)    
            torchaudio.save(output_vocal, final_vocal.cpu(), self.MODEL.samplerate)
        del final_vocal, final_bg
        gc.collect()