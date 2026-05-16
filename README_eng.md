# Video Auto-Dubbing System

<div align="center">
<a href="README.md" style="font-size: 24px">简体中文</a> | 
<a href="README_eng.md" style="font-size: 24px">English</a>
</div>

## Project Overview

This project is a fully automated video processing pipeline focused on dubbing English videos into Chinese versions. The system integrates five core modules: audio track separation, speech recognition, machine translation, AI speech synthesis, and video merging, enabling end-to-end automated processing from raw video to bilingual dubbed video

## Module Details

### 1. Vocal Separation Module
This module is based on Meta's open-source Demucs (Music Source Separation) model with the mdx_extra configuration, separating the input audio track into two independent tracks: Vocal and Background. The vocal track is sent to the subsequent speech recognition module, while the background music and ambient vocals are mixed with the newly synthesized Chinese vocal track during the final video merging stage to ensure auditory integrity of the video. Users can choose whether to save the separated original vocal files for debugging and backup during initialization.

### 2. Speech Recognition Module
This module uses faster-whisper (a lightweight model based on OpenAI Whisper) to transcribe English vocal audio into text. The transcription results are saved in the form of word-level timestamps, and through internal post-processing logic, the token sequences are merged into complete sentences required for bilingual subtitles based on punctuation marks. Meanwhile, independent time segments are retained for the subsequent translation module to ensure synchronization of the final subtitles.

### 3. Translation Module
This module uses the Tencent-Hunyuan/HY-MT1.5-1.8B model to batch-translate English sentences into Chinese. To adapt to different deployment environments and computing budgets, the module has two built-in inference backends：

**Transformer mode (option=0): Implements sequential inference for single sentences, with a small footprint, suitable for resource-constrained scenarios or scenarios requiring flexible debugging.**

**vLLM mode (option=2): Implements efficient batch concurrent inference with significantly improved throughput, especially suitable for long-duration and large-scale content production tasks.**

In addition, the module integrates automatic alignment of bilingual subtitles and SRT file generation functions. After translation, it can directly output subtitle files that can be directly used in the video merging step.

<span style="color:green">PS：Due to the limitations of the graphics card I can reach, it is impossible to integrate vLLM into the entire project for joint debugging (otherwise the video memory will definitely be exhausted). The functional correctness is only verified through unit tests, so only the Transformer mode is available in the actual project. Anyone who is interested in can modify the code for testing by themselves<span>

### 4. Speech Synthesis Module

This module fully integrates the latest capabilities of Qwen3-TTS and is the most technically in-depth part of the project. It follows a combined process of Voice Design + Voice Clone to achieve high-quality and anthropomorphic Chinese TTS:

### 5. Video Merging Module
It merges the background music, newly synthesized Chinese vocal track, and bilingual subtitle files back into the original video stream. To balance compatibility and video quality, the module provides dual encoding schemes of H.264 and H.265. Meanwhile, the module integrates the cropdetect intelligent cropping function for the first time, which can automatically detect and crop out black borders and invalid areas without actual images in the video, saving unnecessary volume for subsequent encoding.

## 🚀 Installation and Usage

### Environment Preparation
It is recommended to use Conda to create an independent runtime environment for Python 3.12 to avoid dependency conflicts.
`conda create -n video_proc python=3.12`
`conda activate video_proc`

### Install Dependencies
`pip install torch==2.10.0 torchaudio==2.10.0 torchcodec==0.10.0 torchvision==0.25.0 transformers`
`pip qwen_tts demucs faster_whisper pyrubberband huggingface_hub pymediainfo pydub`
`conda install -c conda-forge ffmpeg mkl=2021.4.0 libstdcxx-ng -y`
`conda install numpy tqdm -y`
Execuate the following two lines if you encounter issues like 'ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found':
`echo 'export LD_LIBRARY_PATH=/root/miniconda3/envs/video_proc/lib:$LD_LIBRARY_PATH' >> ~/.bashrc`
`source ~/.bashrc`

**Since the model's attention in the code uses attn_implementation = flash_attention_2, it is necessary to install flash-attn. This library is strongly related to the CUDA version number and PyTorch version, so you can refer to (https://github.com/Dao-AILab/flash-attention) to select the version suitable for you, which will not be stated here.**

## Quick Start
`git clone to your-repo-url`
`cd your-repo-url`

### Ensure `run.sh` has executable permissions
`chmod +x run.sh`

### Run the script with the path of the folder containing videos/audio to be processed as the parameter
`./run.sh your-video&audio-path`

## ⚙️ Parameter Details
VideoProc Core Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :---: | :--- |
| debug_en | int | 0 | Debug switch, outputs intermediate log files when set to 1 |
| vocal_save_flag | bool | False | Whether to save the separated original vocal files |
| sentence_len_in_second | float | 7.0 | Maximum duration of subtitle sentences (seconds) |
| translation_mode | int | 0 | Translation mode: 0=Transformer, 1=Ollama (unimplemented), 2=vLLM |
| synthesis_batch_size | int | 10 | Batch size for TTS synthesis |
| operation_path | str | N/A | Input folder path (storing .mp4 and same-named .m4a files) |
| h256flag | bool | False | Whether to use H.265 encoding (requires GPU supporting NVENC) |
| max_cpu_queue_size | int | 5 | Maximum parallelism of merging tasks |

## 💡 Core Dependencies
| Component | Description|
| :--- | :--- |
| Demucs| Audio track separation (Vocal vs Background)|
| faster-whisper| High-speed speech recognition |
| Tencent-Hunyuan/HY-MT1.5-1.8B | English to Chinese translation model |
| Qwen3-TTS | Chinese speech synthesis and cloning |
| PyTorch | Deep learning framework |
| FFmpeg | Audio and video encoding/decoding and merging |

## 🔬 Deep Integration with Qwen3-TTS
This project adopts a two-step strategy to fully utilize the cutting-edge capabilities of Qwen3-TTS and achieve cost-effective batch synthesis through engineering optimization:

**(1)Voice Design Stage: Load the Qwen3-TTS-12Hz-1.7B-VoiceDesign model and generate unique reference audio through the following natural language prompts (instruct):**

**(2)Voice Clone Stage: (This is a key cost reduction strategy) Using the reference audio generated in the previous stage, load the Qwen3-TTS-12Hz-1.7B-Base model, calculate the embedding vector of the voice in one go, generate a reusable clone prompt and cache it in memory. Subsequent batch inference for hundreds of Chinese text segments all reuse the same voice_clone_prompt, and all sentences can be synthesized in batches with a single inference, avoiding repeated transmission of reference audio and significantly reducing latency and inference overhead.**

*Design Intent: This "separation-loading" mode aims to minimize inference costs while ensuring naturalness. Because the Base model only needs to be initialized once and accepts short-duration, small-size reference audio, it can perform batch inference for all subsequent texts, and completely isolate the large and high-cost VoiceDesign model during operation.*

<span style="color:green">PS:My original intention was to use the original audio as the reference audio for voice cloning, but after deployment, I found that the video memory often overflows, causing the code to exit directly. Therefore, I adopted this design + cloning method as a second-best option. It is speculated that the program crash may be caused by the deviation between the reference audio extraction and the real timeline, leading to leakage of previous and subsequent sentences.</span>

## Pipeline Processing Mode
Tests have shown that using the hardware video encoder built into the NVIDIA graphics card to save and merge video files has very low video memory requirements, and the graphics card load is basically below 3%, with a maximum of no more than 5%. Therefore, this project adopts a pipeline mode: after the previous file completes audio track separation, speech recognition, machine translation, and AI speech synthesis, the next file immediately starts audio track separation, speech recognition, machine translation, and AI speech synthesis, while the previous file starts video merging to reduce the overall processing time. At the same time, due to the limited number of built-in hardware video encoders, the length of the file queue that can be saved simultaneously is limited to the number of built-in hardware video encoders. Once this number is exceeded, the work of audio track separation, speech recognition, machine translation, and AI speech synthesis will stop until it drops below the threshold.

## 📂Input and Output Specifications
### Input Files

The project requires that video files and same-named audio track files appear in pairs in the input folder. The system will automatically filter out special characters in the file names through the clean_filename function to complete preliminary cleaning.
If there are old `.srt` subtitle files in the folder, the system will automatically delete them.

### Output Files
After processing, the system generates the following important files:

Bilingual merged video: In the root directory, `<original_filename>_chn.mp4.`
Logs and subtitles: In the sub folder, contains the `.srt` subtitle file `<original_filename>_subscript.srt` generated for the bilingual subtitles.


## ❓ Notes and Troubleshooting

## Graphics Card Requirements
Due to the need to load multiple large models, model loading requires about 8GB of video memory, and tests have shown that video memory will surge instantaneously during speech synthesis operation. It is recommended to choose a model with video memory>=32GB. The RTX5090 I rented can work just fine.

### Issues in Docker Containers
Since I run the code on a rented server and the graphics card is started in a Docker container, I encountered the same issue as (https://github.com/NVIDIA/k8s-device-plugin/issues/1282) when calling the hardware video accelerator inside the graphics card, which is caused by the combination of design flaws of the NVIDIA NVENC hardware encoder and the volume-mounts strategy of the Kubernetes device plugin.

The NVIDIA NVENC encoder hardcodes a wrong assumption at the bottom layer: NVENC does not identify the GPU through the device's major/minor number, UUID, or other unique identifiers, but directly parses the numeric part in the /dev/nvidiaN path to find the corresponding GPU hardware.

When the NVIDIA device plugin uses the deviceListStrategy: volume-mounts mode:
*The device plugin directly mounts the GPU device files on the host into the container*
*But the mounted path may not be consistent with the actual index of the GPU*

For example: the GPU with index 0 on the host (/dev/nvidia0) may be mounted to the /dev/nvidia5 path in the container

At this time, NVENC sees /dev/nvidia5 in the container and tries to use the GPU with index 5, but only the GPU with index 0 is actually available in the container, leading to initialization failure.

The solution is in (https://github.com/flexgrip/nvidia-gpu-enumeration/), and the core code is as follows
`COPY nvenc_fix.c /opt/nvenc_fix.c`
`RUN gcc -shared -fPIC -O2 -o /opt/libnvenc_fix.so /opt/nvenc_fix.c -ldl`
`ENV LD_PRELOAD=/opt/libnvenc_fix.so`
which is, move, compile, and set the environment.
Lines 119~143 in `merge.py` implement the function of setting the environment.
If running this project on a local graphics card, you only need to execute `result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)`

### Error when running `run.sh`: `bad interpreter: No such file or directory`
Open the script with Vim, enter the directory where `run.sh` is located in the terminal, and execute: `vim run.sh`
Enter: `:set ff=unix`(modify to Linux format)
Continue to enter:`:wq`(save and exit Vim)

### Model loading method for HY-MT1.5-1.8B and Qwen3-TTS-12Hz-1.7B 
If you don't use local model, you can delete these thress variables `model_path、gen_model_path and syn_model_path` in `video_proc.py` or you can replace it with your own local path.