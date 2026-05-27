import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import logging
import time
import re
import gc
import soundfile as sf
import numpy as np
import torch as th
import torchaudio.transforms as T
from faster_whisper import WhisperModel
from commonfunc import converttime, get_error_detail

WHISPER_SAMPLE_RATE = 16000
QWENTSS_SAMPLE_RATE = 24000
MAX_DURATION = 10
REF_DURATION = 10

SENTENCE_ENDINGS = re.compile(r'[.!?。！？]+$')
SPLIT_PATTERN = re.compile(r'''
    (?<=[。！？；.!?;])\s+ |  
    (?<=[^0-9])(,)\s         
''', re.VERBOSE)

RED = "\033[91m"    
GREEN = "\033[92m"  
YELLOW = "\033[93m"  
BLUE = "\033[94m"  
PUPPLE = "\033[95m"  
RESET = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def format_time(seconds: float) -> str:
    """
    Convert second into time format for SRT(HH:MM:SS,mmm)
    """
    """Handle exception"""
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "00:00:00,000"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = round((seconds % 1) * 1000)  
    millis = millis if millis < 1000 else 999  
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

class Recognition:

    def __init__(self, model_size: str = None, compute_type: str = None, max_duration: float = None):
        self.MAXDURATION = max_duration if max_duration is not None else MAX_DURATION
        self.DEVICE = "cuda" if th.cuda.is_available() else "cpu"

        """Loading Recognition Model"""
        try:
            init_start_time = time.time()
            """Small and int8 works just fine, don't need to load larger model"""
            _mode_size = model_size if model_size is not None else "small"
            # _compute_type = compute_type if compute_type is not None else "int8"
            if compute_type is not None:
                _compute_type = compute_type
            else:
                _compute_type = "int8_float16" if self.DEVICE == "cuda" else "int8"
            """Set cpu_threads to half of the number logical cores if it is loaded on cpu otherwise set to the number logical cores"""
            """Set num_workers to the number of gpu or 1 if it is loaded on cpu"""
            _num_thread = os.cpu_count() if th.cuda.is_available() else max(1, os.cpu_count() // 2)
            self.RecognitionMODEL = WhisperModel(
                device = self.DEVICE,
                model_size_or_path = _mode_size,
                compute_type = _compute_type,
                cpu_threads = _num_thread,
                num_workers = th.cuda.device_count() if self.DEVICE == "cuda" else 1,
            )
            init_end_time = time.time()
            logger.info(f"{GREEN}Voice Recognition Init Time: {init_end_time - init_start_time:.4f}s{RESET}")
            if self.DEVICE == "cuda":
                free, total = th.cuda.mem_get_info()
                logger.info(f"{YELLOW}Voice Recognition: All: {total / 1024 ** 3:.2f} GB, Free: {free / 1024 ** 3:.2f} GB{RESET}")
            self.occupied_mem =  (total - free) / 1024 ** 3 
        except Exception as e:
            """Throw Exception"""
            logger.error(f"{RED}Failed to load Whisper model: {e}{RESET}")
            raise
        self._resampler_cache = {}

    def preprocess_audio(self, audio: th.Tensor, orig_sr: int) -> th.Tensor:
        """
        Resample it to 16KHZ
        """
        """Make sure the audio is on specific device"""
        audio_device = audio.to(self.DEVICE)

        """Resample it to 16KHz and delete original audio"""
        resampler = self.get_resampler(orig_sr)
        audio_16k = resampler(audio_device)
        """Make sure it's single channel"""""
        if audio_16k.dim() > 1:
            audio_16k = audio_16k.mean(dim=0, keepdim=True)
        return audio_16k.squeeze(0)

    def get_resampler(self, orig_freq: int) -> T.Resample:
        """Acquire Resampler on self.DEVICE"""
        if orig_freq not in self._resampler_cache:
            self._resampler_cache[orig_freq] = T.Resample(
                orig_freq=orig_freq,
                new_freq=WHISPER_SAMPLE_RATE,
                resampling_method='sinc_interp_kaiser',
                lowpass_filter_width=64,
                rolloff=0.95,
                beta=14.0
            ).to(self.DEVICE)
        return self._resampler_cache[orig_freq]
    
    def process_segments_to_final_subtitles(self, segments_list):
    
        sentences = []
        current_sentence = []
        sentence_start_time = None

        for i, segment in enumerate(segments_list):
            text = segment.text.strip()
            if not text:
                continue

            current_sentence.append(text)
            if SENTENCE_ENDINGS.search(text):
                sentence_text = " ".join(current_sentence).strip()
                sentence_start_time = sentence_start_time if sentence_start_time is not None else segment.start
                sentence_end_time = segment.end

                if sentence_text:
                    sentences.append({
                        "text": sentence_text,
                        "start": sentence_start_time,
                        "end": sentence_end_time
                    })

                current_sentence = []
                sentence_start_time = None
            else:
                if sentence_start_time is None:
                    sentence_start_time = segment.start

        if current_sentence:
            sentence_text = " ".join(current_sentence).strip()
            if sentence_text and sentence_start_time is not None:
                sentences.append({
                    "text": sentence_text,
                    "start": sentence_start_time,
                    "end": segments_list[-1].end if i < len(segments_list) - 1 else segment.words[-1].end  
                })
                
        final_subtitles = []
        durations = []
        prevgaps = []
        aftergaps = []
        for i, sentence in enumerate(sentences):
            start = sentence["start"]
            end = sentence["end"]
            text = sentence["text"]
            duration = end - start

            if duration <= self.MAXDURATION:
                final_subtitles.append({
                    "text": text,
                    "start": start,
                    "end": end
                })
                prev_gap = 1.0 if i == 0 else (sentences[i]["start"] - sentences[i-1]["end"])
                after_gap = 1.0 if i == len(sentences) - 1 else (sentences[i+1]["start"] - sentences[i]["end"])
                prevgaps.append(prev_gap)
                aftergaps.append(after_gap)
                durations.append(duration)
                continue

            parts = re.split(SPLIT_PATTERN, text)
            parts = [p.strip() for p in parts if p is not None and p.strip()]

            chunks = []
            temp = ""
            for part in parts:
                if part in [',', ';', '。', '！', '？', '.', '!', '?']:
                    if temp:
                        temp += part
                        chunks.append(temp)
                        temp = ""
                    else:
                        chunks.append(part)
                else:
                    if temp:
                        chunks.append(temp)
                        temp = part
                    else:
                        temp = part
            if temp:
                chunks.append(temp)

            if not chunks:
                chunks = [text]

            chunk_count = len(chunks)
            time_per_chunk = duration / chunk_count

            current_start = start
            for chunk_idx, chunk in enumerate(chunks):
                current_end = current_start + time_per_chunk
                if chunk.strip():
                    final_subtitles.append({
                        "text": chunk,
                        "start": current_start,
                        "end": current_end
                    })
                    # Gaps for split chunks: only sentence boundaries have real gaps
                    if chunk_idx == 0:
                        prev_gap = 1.0 if i == 0 else (sentences[i]["start"] - sentences[i-1]["end"])
                    else:
                        prev_gap = 0.0
                    if chunk_idx == chunk_count - 1:
                        after_gap = 1.0 if i == len(sentences) - 1 else (sentences[i+1]["start"] - sentences[i]["end"])
                    else:
                        after_gap = 0.0
                    prevgaps.append(prev_gap)
                    aftergaps.append(after_gap)
                    durations.append(current_end - current_start)
                current_start = current_end

        # Prefer segments with duration 5~10s and gaps > 0.5s on both sides
        idx = None
        candidate_idx = []
        for idx_candidate, (dur, prev_g, after_g) in enumerate(zip(durations, prevgaps, aftergaps)):
            if 5.0 <= dur <= 10.0 and prev_g >= 0.5 and after_g >= 0.5:
                candidate_idx.append((idx_candidate, dur))
        if candidate_idx is not None and len(candidate_idx) > 0:        
            candidate_idx.sort(key=lambda x: x[1], reverse=True)
            idx = candidate_idx[0][0]
        else:
            # Fallback to original logic
            if durations[0] > 3:
                idx = 0
            else:
                if durations:
                    durations_tensor = th.tensor(durations)
                    idx = th.argmin(th.abs(durations_tensor - REF_DURATION)).item()
                else:
                    idx = 0

        return final_subtitles, idx

    def recognition(self, audio, sr, debug_en, ref_audio_path = None):
        total_start_time = time.time()
        try:
            """Resample to 16KHz and for faster_whisper the input must be numpy on cpu"""
            audio_recognization = self.preprocess_audio(audio, sr).cpu().numpy().astype(np.float32)
                
            """Speech Recognition and convert to list"""
            segments_generator, _ = self.RecognitionMODEL.transcribe(
                    audio=audio_recognization,
                    task="transcribe",
                    language="en",
                    word_timestamps=True
            )
            segments_list = list(segments_generator)
             
            """Generate sentences from split segments"""
            english_sentences, idx = self.process_segments_to_final_subtitles(segments_list)

            """Save reference audio if ref_audio_path is provided"""
            if ref_audio_path is not None:
                if os.path.isfile(ref_audio_path):
                    os.remove(ref_audio_path)
                ref_audio = audio[:, int((english_sentences[idx]["start"] - 0.25) * sr): int((english_sentences[idx]["end"] + 0.25) * sr)]
                if ref_audio.ndim > 1 and ref_audio.shape[0] > 1:
                    mono_audio = th.mean(ref_audio, dim=0)  
                else:
                    mono_audio = ref_audio.squeeze(0)
                sf.write(ref_audio_path, mono_audio.cpu().float().numpy(), sr)
            ref_text = english_sentences[idx]["text"]
            """Save debug info"""   
            # if debug_en == True:
            #     with open(r"EnglishContent_origin.txt", "w", encoding="utf-8") as f0:
            #         for i, segment in enumerate(segments_list):
            #             f0.write(f"{i}\n")
            #             f0.write(format_time(segment.start) + "\t"+ "->" + format_time(segment.end) + "\n")
            #             f0.write(segment.text + "\n\n") 
            #     with open(r"EnglishContent_merged.txt", "w", encoding="utf-8") as f1:
            #         for i, segment in enumerate(english_sentences):
            #             f1.write(f"{i}\n")
            #             f1.write(format_time(segment["start"]) + "\t"+ "->" + format_time(segment["end"]) + "\n")
            #             f1.write(segment["text"] + "\n\n")    

            """Clear Memory"""
            del audio_recognization, segments_generator, segments_list
            gc.collect()
            if self.DEVICE == "cuda":
                th.cuda.empty_cache()
                th.cuda.synchronize()
                free, _ = th.cuda.mem_get_info()
                logger.info(f"{YELLOW}After Speech Recognition Free: {free / 1024 ** 3:.2f} GB{RESET}")
            total_end_time = time.time()
            hours,minutes,seconds,miliseconds = converttime(total_end_time - total_start_time)
            logger.info(f"{GREEN}Transcript Time is {hours} hours, {minutes} minutes, {seconds} seconds and {miliseconds} ms{RESET}")
            return english_sentences, ref_text
        except Exception as e:
            """Throw Exception"""
            error_detail = get_error_detail(e)
            logger.error(f"{RED}{error_detail}{RESET}")
            raise

