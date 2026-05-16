import time
import gc
import threading
import re

import multiprocessing
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

import torch as th

from pathlib import Path
from pymediainfo import MediaInfo
from pydub.utils import mediainfo
from concurrent.futures import ThreadPoolExecutor, Future

from separation import AudioSeparation
from recognition import Recognition
from translation import SubscriptTranslation
from synthesis import VoiceSynthesis
from merge import compose_video
from commonfunc import converttime

RED = "\033[91m"    
GREEN = "\033[92m"  
YELLOW = "\033[93m"  
BLUE = "\033[94m"  
PUPPLE = "\033[95m"  
RESET = "\033[0m"

def clean_filename(filename):
    """Clean all extra marks"""
    name_part, ext_part = os.path.splitext(filename)
    
    """Only retain alphabets, numbers and spaces"""
    cleaned_name = re.sub(r'[^\w\s]', '', name_part)
    new_filename = cleaned_name + ext_part
    return new_filename


def classify_video_audio_and_matched_pairs(folder_path):
    video_files = []  
    audio_files = []
    video_info = {}
    audio_duration = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".srt"):
            srt_path = os.path.join(folder_path, filename)
            try:
                os.remove(srt_path)
            except:
                pass

        file_path = os.path.join(folder_path, filename)

        """Skip folder""" 
        if not os.path.isfile(file_path):
            continue

        """Look for usual format"""
        if not filename.lower().endswith(('.mp4', '.webm', '.m4a')):
            continue

        """Automatically Rename"""
        new_filename = clean_filename(filename)
        if new_filename != filename:
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            filename = new_filename  
            file_path = os.path.join(folder_path, filename)

        media = MediaInfo.parse(file_path)
        is_video = False
        is_audio = False
        width = height = None
        for track in media.tracks:
            if track.track_type == "Video":
                is_video = True
                width = track.width
                height = track.height
            if track.track_type == "Audio":
                is_audio = True
        if is_video and not is_audio:
            video_files.append(filename)
            video_info[filename] = (width, height)
        elif is_audio and not is_video:
            audio_files.append(filename)
            try:
                info = mediainfo(file_path)
                dur = float(info["duration"])
            except:
                dur = 0.0
            audio_duration[filename] = dur
            
    
    audio_dict = {os.path.splitext(a)[0]: a for a in audio_files}
    matched = []
    for video in video_files:
        base = os.path.splitext(video)[0]
        if base in audio_dict:
            audio_file = audio_dict[base]
            dur = audio_duration.get(audio_file, 0.0)
            w, h = video_info.get(video, (None, None))
            matched.append((video, audio_dict[base], w, h, dur))
    matched.sort(key=lambda x: x[4], reverse=True)
    final_matched = [(item[0], item[1], item[2], item[3]) for item in matched]
    return final_matched

class VideoProc:

    def __init__(
            self, 
            debug_en = False,               # Debug Mode
            sentence_len_in_second = 7.0,   # Threshold of sentence segmentation
            translation_mode = 0,           # 0:Transformer; 1:ollama(not execuated); 2:vllm(code completed, no header included)
            synthesis_batch_size = 10,      # Batch size of vocal synthesis
            operation_path = './',          # Path of video and audio
            h256flag = True,               # H.265 coding protocl
            max_cpu_queue_size = 5          # Maximum number of cpu work to be processed
        ):

        self.debug_en = debug_en
        self.operation_path = operation_path
        self.h256flag = h256flag
        self.max_cpu_queue_size = max_cpu_queue_size

        self.separator = AudioSeparation(save_flag = debug_en)
        self.recognition = Recognition(max_duration = sentence_len_in_second)
        self.translation = SubscriptTranslation(option = translation_mode)
        self.synthesis = VoiceSynthesis(batchsize = synthesis_batch_size)
        self.matched = classify_video_audio_and_matched_pairs(operation_path)

    def gpu_workload0(self, idx, video, audio, width, height):
        try:
            job_start = time.time()
            temp_file_name = self.operation_path + "/temp"
            """Extract Video and Audio Name"""
            audio_name = os.path.join(self.operation_path, audio)
            video_name = os.path.join(self.operation_path, video)
            filename = os.path.splitext(audio)[0]
            logger.info(f"{PUPPLE}Start Separation, Transcript, Translation and Synthesis for the {idx} th File:{filename} {RESET}")

            """Generate Merged Video Name"""
            name_stem, _ = os.path.splitext(audio)
            clean_stem = name_stem.rstrip()

            """Tempfile filename"""
            bgm_name = os.path.join(temp_file_name, f"{clean_stem}_background.mp3")
            vocal_ori_name = os.path.join(temp_file_name, f"{clean_stem}_vocal_original.mp3")
            vocal_clone_name = os.path.join(temp_file_name, f"{clean_stem}_vocal_clone.mp3")

            """Subtitle filename"""
            sub_file_name = self.operation_path + "/sub"
            subtitle_name = os.path.join(sub_file_name, f"{clean_stem}_subscript.srt")

            """Start Processing"""
            vocals, sr, ori_len, ori_sr = self.separator.audio_separate(audio_path = audio_name, bgm_path = bgm_name, vocal_path = vocal_ori_name)
            english_sentences = self.recognition.recognition(vocals, sr, self.debug_en)
            total_sentences = self.translation.Subscript_Translation_Srt_Generation(english_sentences, subtitle_name)
            self.synthesis.synthesize(total_sentences, sr, ori_len, ori_sr, debug_en = self.debug_en, vocal_path = vocal_clone_name)
            job_end = time.time()
            elapse_time = job_end - job_start

            """Clear Memory"""
            gc.collect()
            if th.cuda.is_available():
                th.cuda.synchronize()
                th.cuda.empty_cache()
            return {
                "idx": idx,
                "video_name": video_name,
                "audio_name": audio_name,
                "merge_name": os.path.join(self.operation_path, f"{clean_stem}_chn.mp4"),
                "bgm_name": bgm_name,
                "vocal_original_name":vocal_ori_name,
                "subtitle_name": subtitle_name,
                "vocal_clone_name": vocal_clone_name,
                "success": True,
                "elapse_time": elapse_time, 
                "target_width": width,
                "target_height": height
            }
        except Exception as e:
            logger.error(f"{RED}Processing {idx}th File Failed: {e}{RESET}")
            return {"idx": idx, "success": False, "error": str(e)}
        
    def gpu_workload1(self, preprocess_result):

        if not preprocess_result["success"]:
            logger.error(f"{RED}Skip {preprocess_result.get("idx")}th File{RESET}")
            return
        
        try:
            logger.info(f"{PUPPLE}Start Video Merging for the {preprocess_result.get("idx")} th File{RESET}")
            flag = compose_video(        
                video_path=preprocess_result["video_name"],
                subtitle_path=preprocess_result["subtitle_name"],
                bgm_path=preprocess_result["bgm_name"],
                voice_path=preprocess_result["vocal_clone_name"],
                output_path=preprocess_result["merge_name"],
                use_h265=self.h256flag
            )

            """If not in debug mode after sucessful merging video files, temp files should be removed"""
            if flag == 1 and self.debug_en == False:
                video_name = preprocess_result.get("video_name")
                audio_name = preprocess_result.get("audio_name")
                bgm_name = preprocess_result.get("bgm_name")
                vocal_original_name = preprocess_result.get("vocal_original_name")
                vocal_clone_name = preprocess_result.get("vocal_clone_name")
                """If Succeed, delete Redundant Files"""
                if os.path.isfile(video_name):
                    os.remove(video_name)
                if os.path.isfile(audio_name):
                    os.remove(audio_name)
                if os.path.isfile(bgm_name):
                    os.remove(bgm_name)
                if os.path.isfile(vocal_original_name):
                    os.remove(vocal_original_name)
                if os.path.isfile(vocal_clone_name):
                    os.remove(vocal_clone_name)
        except Exception as e:
            logger.error(f"{RED}Merge {preprocess_result['idx']}th File Failed: {e}{RESET}")
        finally:
            free, total = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}After Video Processing All: {total / 1024 ** 3:.2f} GB, Free: {free / 1024 ** 3:.2f} GB{RESET}")

    def _gpu_task_wrapper(self, result, gpu1_sem):
        """Wrap Task, to release signal"""
        idx = result.get("idx")
        gpu1_sem.acquire()
        job_start = time.time()
        try:
            self.gpu_workload1(result)
        except Exception as e:
            logger.error(f"{RED}Error in gpu_workload1 for {idx}: {e}{RESET}")
        else:
            job_end = time.time()
            total_elapsed = job_end - job_start + result.get("elapse_time", 0)
            hours,minutes,seconds,miliseconds = converttime(total_elapsed)
            logger.info(f"{GREEN}Total Processing time for {idx}th file is {hours}hour, {minutes}minutes, {seconds}seconds and {miliseconds}ms{RESET}")
        finally:
            gpu1_sem.release()

    def process(self):
        start_time = time.time()
        if not self.matched:
            logger.info(f"{PUPPLE}Files not found, Exit{RESET}")
            return
        """Make dir"""
        temp_file_name = os.path.join(self.operation_path, "temp")
        tempfolder = Path(temp_file_name) 
        tempfolder.mkdir(parents=True, exist_ok=True)

        subtitle_name = os.path.join(self.operation_path, "sub")
        subfolder = Path(subtitle_name)  
        subfolder.mkdir(parents=True, exist_ok=True)
   
        logger.info(f"{PUPPLE}Detect {len(self.matched)} files{RESET}")

        gpu0_sem = threading.Semaphore(1)   
        gpu1_sem = threading.Semaphore(self.max_cpu_queue_size)

        def task_pipeline(idx, video, audio, width, height):
            with gpu0_sem:
                result = self.gpu_workload0(idx, video, audio, width, height)
            if not result["success"]:
                logger.error(f"Skip {idx} due to workload0 failure")
                return
            self._gpu_task_wrapper(result, gpu1_sem)
        with ThreadPoolExecutor(max_workers=3) as executor:
            for idx, (video, audio, w, h) in enumerate(self.matched, 1):
                executor.submit(task_pipeline, idx, video, audio, w, h)

        end_time = time.time()
        hours,minutes,seconds,miliseconds = converttime(end_time - start_time)
        logger.info(f"{GREEN}Total Processing time is {hours}hour, {minutes}minutes, {seconds}seconds and {miliseconds}ms{RESET}")