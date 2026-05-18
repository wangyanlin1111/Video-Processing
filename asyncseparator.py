import concurrent
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
import soundfile as sf

from concurrent import futures
from pydub.utils import mediainfo
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

DEMUCS_SAMPLE_RATE = 44100

def _normalize_audio(tensor):
    max_val = th.max(th.abs(tensor))
    if max_val > 1.0:
        tensor = tensor / max_val
    return tensor

class AsyncAudioSeparation:

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

    def audio_separate(self, audio_path:str = None, bgm_path:str = None, on_final_save_callback = None):
        total_start_time = time.time()

        """Acquire audio information"""
        info = mediainfo(audio_path)
        duration = float(info["duration"])
        orig_sr = int(info["sample_rate"])
        total_frames = int(duration * float(orig_sr))
        self.total_duration = duration
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
            args=(process_queue, bgm_path, on_final_save_callback, total_frames, orig_sr)
        )
        
        loader.start()
        processor.start()
        saver.start()
        
        saver.join()
        processor.join()
        loader.join()

        total_end_time = time.time()
        hours,minutes,seconds,miliseconds = converttime(total_end_time - total_start_time)
        logger.info(f"{GREEN}Audio Separation Time is {hours} hours, {minutes} minutes, {seconds} seconds and {miliseconds} ms{RESET}")
        free, _ = th.cuda.mem_get_info()
        logger.info(f"{YELLOW}After Audio Separation Free: {free / 1024 ** 3:.2f} GB{RESET}")
        return getattr(self, 'final_vocal', None), DEMUCS_SAMPLE_RATE, total_frames, orig_sr, self._recognition_future

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
            load_queue.put((seg_idx, wav))

        load_queue.put(None)
    
    def _processor_worker(self, load_queue, original_sr, process_queue):

        """Runs on GPU - processes audio segments"""
        """Only one _processor_worker exists, so GPU is naturally protected from running multiple files' processing simultaneously"""
        while True:
            item = load_queue.get()
            if item is None:
                process_queue.put(None)
                break
                
            seg_idx, wav = item
            separation_model = get_model(self.MODEL_NAME).to(self.DEVICE)
            wav = convert_audio(wav, original_sr, DEMUCS_SAMPLE_RATE, separation_model.audio_channels)
            wav = wav.unsqueeze(0).to(self.DEVICE)
            
            with th.no_grad():
                sources = apply_model(
                    separation_model, 
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
            process_queue.put((seg_idx, vocal_cpu, bg_cpu))
            
            del wav, sources, drums, bass, other, vocal, bg, separation_model
            gc.collect()
            if self.DEVICE == "cuda":
                th.cuda.empty_cache()

    def _saver_worker(self, process_queue, output_bg, on_final_save_callback, total_frames, orig_sr):

        accumulated_vocal = None
        accumulated_bg = None
        first_seg = True

        while True:
            item = process_queue.get()
            if item is None:
                break

            _, vocal, bg = item

            if first_seg:
                accumulated_vocal = vocal
                accumulated_bg = bg
                first_seg = False
            else:

                """Overlap-Add with Crossfade"""
                overlap_samples = int(self.overlapsecond * DEMUCS_SAMPLE_RATE)
                crossfade_len = 2 * overlap_samples  
                actual_len = min(crossfade_len, accumulated_vocal.shape[-1], vocal.shape[-1])

                if actual_len > 0:

                    """Fade tail of accumulated audio and fade head of current audio"""
                    fade_out = T.Fade(fade_out_len=actual_len, fade_shape='half_sine')
                    faded_tail_vocal = fade_out(accumulated_vocal[:, -actual_len:])
                    faded_tail_bg = fade_out(accumulated_bg[:, -actual_len:])
                    fade_in = T.Fade(fade_in_len=actual_len, fade_shape='half_sine')
                    faded_head_vocal = fade_in(vocal[:, :actual_len])
                    faded_head_bg = fade_in(bg[:, :actual_len])

                    """Create crossfaded part by adding faded tail of accumulated audio and faded head of current audio"""
                    crossfaded_vocal = faded_tail_vocal + faded_head_vocal
                    crossfaded_bg = faded_tail_bg + faded_head_bg

                    """Concatenate non-overlapping parts with crossfaded part"""
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

                    """If no overlap possible, just concatenate (happens when segments are too short)"""
                    accumulated_vocal = th.cat([accumulated_vocal, vocal], dim=-1)
                    accumulated_bg = th.cat([accumulated_bg, bg], dim=-1)

                del vocal, bg
                gc.collect()

        if accumulated_vocal is None or accumulated_bg is None:
            logger.warning("No audio data to save!")
            return

        """Cut redundant overlap at the end"""
        trim_samples = int(self.overlapsecond * DEMUCS_SAMPLE_RATE)
        if accumulated_vocal.shape[-1] > trim_samples:
            accumulated_vocal = accumulated_vocal[:, :-trim_samples]
            accumulated_bg = accumulated_bg[:, :-trim_samples]

        """Output vocal for down-streaming recognition callback"""
        logger.info(f"{PUPPLE}Vocal separated successfully.{RESET}")
        self.final_vocal = accumulated_vocal.cpu()
        if on_final_save_callback is not None and accumulated_vocal is not None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._recognition_future = executor.submit(
                on_final_save_callback, accumulated_vocal.cpu(), 
                DEMUCS_SAMPLE_RATE,
                total_frames, 
                orig_sr
            )
            executor.shutdown(wait=False)
        else:
            self._recognition_future = None

        """Save background music"""
        audio_bg_np = accumulated_bg.cpu().numpy().T
        del accumulated_bg
        chunk_samples = self.segment_duration * DEMUCS_SAMPLE_RATE
        total_samples = audio_bg_np.shape[0]
        if output_bg is not None:
            if os.path.isfile(output_bg):
                os.remove(output_bg)
        with sf.SoundFile(output_bg, 'w', samplerate=DEMUCS_SAMPLE_RATE, channels=2) as f:
            for start_idx in range(0, total_samples, chunk_samples):
                end_idx = min(start_idx + chunk_samples, total_samples)
                f.write(audio_bg_np[start_idx:end_idx])
        del audio_bg_np
        logger.info(f"{PUPPLE}Background music saved successfully.{RESET}")
        gc.collect()
        if self.DEVICE == "cuda":
            th.cuda.empty_cache()
        