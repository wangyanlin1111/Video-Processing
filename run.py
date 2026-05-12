import traceback
import multiprocessing
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

import sys

from video_proc import VideoProc

if __name__ == '__main__':
    if len(sys.argv) > 1:
        operation_path = sys.argv[1]
    else:
        operation_path = "../videos"
    try:
        video_proc = VideoProc(
            debug_en = 0, 
            vocal_save_flag = True, 
            sentence_len_in_second = 7, 
            translation_mode = 0, 
            synthesis_batch_size = 15, 
            operation_path = operation_path, 
            h256flag = True, 
            max_cpu_queue_size = 5
        )
        video_proc.process()
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
    os.system("/usr/bin/shutdown")







