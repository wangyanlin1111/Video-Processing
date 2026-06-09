import traceback
import multiprocessing
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

import sys

from video_proc import VideoProc

if __name__ == '__main__':
    if len(sys.argv) > 1:
        operation_path = sys.argv[1]
    else:
        operation_path = "../videos"
    if len(sys.argv) > 2:
        debug_en = sys.argv[2].lower() in ('true', '1', 'yes')
    else:
        debug_en = False 
    try:
        video_proc = VideoProc(
            debug_en = debug_en, 
            sentence_len_in_second = 7, 
            translation_mode = 2, 
            synthesis_batch_size = 128, 
            operation_path = operation_path, 
            h256flag = True, 
            max_cpu_queue_size = 5,
            monitor_interval = 2.0
        )
        video_proc.process()
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
    if debug_en == False:
        os.system("/usr/bin/shutdown")







