import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

import os
import time
import re
import multiprocessing
import subprocess
from pathlib import Path
from shlex import quote
from commonfunc import converttime

RED = "\033[91m"    
GREEN = "\033[92m"  
YELLOW = "\033[93m"  
BLUE = "\033[94m"  
PUPPLE = "\033[95m"  
RESET = "\033[0m"
subtitle_font_config=":fontsdir=/usr/share/fonts/opentype/noto:force_style='FontName=Noto Sans CJK SC,FontSize=18'"

def get_crop_filter(video_path):
    try:
        cmd = [
            'ffmpeg', '-hide_banner', '-i', video_path,
            '-t', '60',
            '-vf', 'cropdetect', '-f', 'null', '-'
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        match = re.search(r'crop=(\d+:\d+:\d+:\d+)', res.stderr)
        if match:
            return f"crop={match.group(1)}"
    except:
        pass
    return ""

def get_video_width_after_crop(video_path, crop_str):
    """Acquire video width after crop"""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
           '-show_entries', 'stream=width', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    original_width = int(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
    
    if crop_str:
        match = re.search(r'crop=(\d+):', crop_str)
        if match:
            return int(match.group(1))
    return original_width

def compose_video(
    video_path, 
    subtitle_path, 
    bgm_path, 
    voice_path, 
    output_path,
    use_h265=False, 
):
    """Check whether input files exist"""
    for p in [video_path, bgm_path, voice_path, subtitle_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{RED}File doesn't exist: {p}{RESET}")
    
    """Check whether output file exists, if exist then delete"""    
    if os.path.isfile(output_path):
        os.remove(output_path)

    vcodec = "hevc_nvenc" if use_h265 == True else "h264_nvenc"
    codingmethod = "H.265" if use_h265 == True else "H.264"
    sub_safe = quote(subtitle_path)

    """Check whether there are meaningless borders, if they exist, crop them"""
    crop_str = get_crop_filter(video_path)
    if crop_str:
        width = get_video_width_after_crop(video_path, crop_str)
        max_width = int(width * 0.8)  
        subtitle_font_config = f":fontsdir=/usr/share/fonts/opentype/noto:force_style='FontName=Noto Sans CJK SC,FontSize=16,MarginL=20,MarginR=20,Alignment=2,MaxWrapWidth={max_width}'"
        video_chain = f"[0:v]{crop_str},subtitles={sub_safe}{subtitle_font_config}[vout]"
    else:
        subtitle_font_config = ":fontsdir=/usr/share/fonts/opentype/noto:force_style='FontName=Noto Sans CJK SC,FontSize=16,MarginL=20,MarginR=20,Alignment=2'"
        video_chain = f"[0:v]subtitles={sub_safe}{subtitle_font_config}[vout]"
        
    filter_complex = (
        f"[1:a]volume=0.4[bgm];"
        f"[2:a]volume=1.0[voice];"
        f"{video_chain};"
        f"[bgm][voice]amix=inputs=2:duration=first[aout]"
    )
    if use_h265:
        crf = 32  
        preset = "slow"
        extra_video_params = [
            "-rc", "vbr_hq",        # High-quality VBR mode (better than default vbr)
            "-cq", str(crf),        # NVENC CQ corresponds to CRF (constant quality)
            "-profile:v", "main10", # 10-bit encoding (more accurate colors, higher compression ratio)
            "-spatial_aq", "1",     # Enable spatial AQ
            "-temporal_aq", "1",    # Enable temporal AQ
            "-bf", "4",             # Enable B-frames (improves compression ratio if hardware supports)
        ]
    else:
        crf = 28
        preset = "medium"
        extra_video_params = [
            "-cq", str(crf),
            "-profile:v", "main",
        ]

    cmd = [
        'ffmpeg', '-y',
        '-hide_banner',      
        '-loglevel', 'quiet',
        '-stats',
        '-hwaccel', 'cuda',        
        '-i', video_path,
        '-i', bgm_path,
        '-i', voice_path,
        '-filter_complex', filter_complex,
        '-map', '[vout]',
        '-map', '[aout]',
        '-c:v', vcodec,
        '-preset', preset,
        *extra_video_params,
        '-c:a', 'aac',
        '-b:a', '128k',
        '-ac', '2',
        '-movflags', '+faststart', 
        '-vsync', 'cfr',  # Force constant frame rate to avoid duplicate frames
        '-flags', '+cgop',  # Optimize keyframe interval to improve compression ratio
        output_path
    ]
    total_start_time = time.time()

    original_ld_preload = os.environ.get("LD_PRELOAD", "")
    try:
        os.environ["LD_PRELOAD"] = "/opt/libnvenc_fix.so"
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stderr:
            speed_values = []
            for line in result.stderr.splitlines():
                if 'speed=' in line:
                    match = re.search(r'speed=([\d.]+)x', line)
                    if match:
                        speed_num = float(match.group(1))
                        speed_values.append(speed_num)
            if speed_values:
                avg_speed = sum(speed_values) / len(speed_values)
                logger.info(f"{GREEN}Video Saving Ratio = {avg_speed:.2f} using {codingmethod} {RESET}")            
    finally:
        if original_ld_preload:
            os.environ["LD_PRELOAD"] = original_ld_preload
        else:
            os.environ.pop("LD_PRELOAD", None)
    # result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"{RED}Merge Fail:\n{result.stderr}{RESET}")
        allcmd = ' '.join(cmd)
        print(allcmd)
        flag = 0
    else:
        total_end_time = time.time()
        hours,minutes,seconds,miliseconds = converttime(total_end_time - total_start_time)
        logger.info(f"{GREEN}Video Saving Time is {hours} hours, {minutes} minutes, {seconds} seconds and {miliseconds} ms{RESET}")
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024*1024)  # MB
            logger.info(f"{GREEN}Output File Size: {output_size:.2f} MB ({codingmethod}, CRF={crf}){RESET}")
        flag = 1
    return flag