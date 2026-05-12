# 视频自动配音系统

<div align="center">
<a href="README.md" style="font-size: 24px">简体中文</a> | 
<a href="README_eng.md" style="font-size: 24px">English</a>
</div>

## 项目概述

本项目是一个全自动的视频处理流水线，专注于将英文视频配音为中文版本。系统集成了音轨分离、语音识别、机器翻译、AI语音合成和视频合并五大核心模块，实现了从原始视频到双语配音视频的端到端自动化处理。

## 模块详解

### 1. 人声分离模块
该模块基于 Meta 开源的 Demucs（Music Source Separation） 模型，采用 mdx_extra 的模型配置，将输入音轨分离为人声（Vocal）和背景音乐（Background）两个独立音轨。其中，人声轨将被送入后续的语音识别模块，而背景音乐与背景人声将在视频最终合并阶段与全新合成的中文人声轨进行混音，以保证视频的听觉完整性。用户在初始化时，可选择是否保存分离出的原始人声文件以供调试、备份之用。

### 2. 语音识别模块
该模块采用 faster-whisper（基于 OpenAI Whisper 的小型模型）将英文人声转录为文本。转录结果以带精准时间戳的词元（word-level timestamps）形式保存，并通过内部的后处理逻辑，依据标点符号将词元序列合并为双语字幕所需的完整句子，同时为后续翻译模块保留独立的时间分段，确保最终字幕的同步性。

### 3. 翻译模块
该模块利用 Tencent-Hunyuan/HY-MT1.5-1.8B 模型将英文句子批量翻译为中文。为适配不同的部署环境和计算预算，模块内置两种推理后端：

**Transformer 模式 (option=0)：实现单句顺序推理，体积小巧，适合资源受限或需要灵活调试的场景。**

**vLLM 模式 (option=2)：实现高效的批量并发推理，吞吐能力大幅提升，特别适合长时间、大型内容生产任务。**

此外，模块集成了双语字幕自动对齐与 SRT 文件生成功能，翻译完成后可直接输出可供视频合并步骤直接调用的字幕文件。

<span style="color:green">PS：由于本人使用的显卡限制，无法将vLLM嵌入在整个项目中联调否则必爆显存，仅通过单元测试验证功能正确性，因而实际工程中仅有Transformer模式，感兴趣的同学可自行修改代码进行测试。<span>

### 4. 语音合成模块

该模块完整集成了 Qwen3-TTS 的最新能力，是项目中最具技术深度的部分。它遵循 Voice Design（音色设计） + Voice Clone（音色克隆） 的组合流程，实现高质量、拟人化的中文 TTS：

### 5. 视频合并模块
将背景音乐、新合成的中文人声轨与双语字幕文件合并回原始视频流。为兼顾兼容性与画质，模块提供 H.264 与 H.265 双编码方案。同时，该模块首次集成了 cropdetect 智能裁剪功能，能自动检测并裁剪掉视频中无实际画面的黑边与无效区域，为后续编码节省不必要的体积。

## 🚀 安装与使用

### 环境准备
建议使用 Conda 创建 Python 3.12 的独立运行环境，以避免依赖冲突。
`conda create -n video_proc python=3.12`
`conda activate video_proc`

### 安装依赖库
`pip install torch==2.10.0 torchaudio==2.10.0 torchcodec==0.10.0 torchvision==0.25.0 transformers`
`pip qwen_tts demucs faster_whisper pyrubberband huggingface_hub pymediainfo pydub`
`conda install -c conda-forge ffmpeg mkl=2021.4.0 -y`
`conda install numpy tqdm -y`

**由于代码中模型的注意力使用了 `attn_implementation = flash_attention_2`，需要安装flash-attn，而这个库cuda版本号和pytorch版本等强相关，因此可参考(https://github.com/Dao-AILab/flash-attention)选择合适自己的版本，在此不再赘述。**

## 快速开始
`git clone to your-repo-url`
`cd your-repo-url`

### 确保 run.sh 具有执行权限
`chmod +x run.sh`

### 运行脚本，参数为待处理的视频/音频文件夹路径
`./run.sh your-video&audio-path`

## ⚙️ 参数详解
VideoProc 核心参数
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :---: | :--- |
| debug_en | int | 0 | 调试开关，为 1 时输出中间日志文件 |
| vocal_save_flag | bool | False | 是否保存分离出的原始人声文件 |
| sentence_len_in_second | float | 7.0 | 字幕句子的最长持续时间（秒） |
| translation_mode | int | 0 | 翻译模式：0=Transformer，1=Ollama(未实现)，2=vLLM |
| synthesis_batch_size | int | 10 | TTS 合成的批量大小 |
| operation_path | str | N/A | 输入文件夹路径（存放 .mp4 和同名 .m4a 文件） |
| h256flag | bool | False | 是否使用 H.265 编码（需要 GPU 支持 NVENC） |
| max_cpu_queue_size | int | 5 | 合并任务的最大并行数 |

## 💡 核心依赖
| 组件 | 说明|
| :--- | :--- |
| Demucs| 音轨分离（人声 vs 背景）|
| faster-whisper| 高速语音识别 |
| Tencent-Hunyuan/HY-MT1.5-1.8B | 英译中翻译模型 |
| Qwen3-TTS | 中文语音合成与克隆 |
| PyTorch | 深度学习框架 |
| FFmpeg | 音视频编解码与合并 |

## 🔬 与 Qwen3-TTS 的深度集成
本项目采用 两步走策略，充分利用了 Qwen3-TTS 最前沿的能力，并通过工程优化实现了高性价比的批量合成：

**(1)音色设计阶段（Voice Design）：加载 Qwen3-TTS-12Hz-1.7B-VoiceDesign 模型，通过以下自然语言提示（instruct）生成独特的参考音频：**

**(2)音色克隆阶段（Voice Clone）：（此为关键降本策略） 利用上一阶段生成的参考音频，加载 Qwen3-TTS-12Hz-1.7B-Base 模型，一次性计算出该音色的嵌入向量，并生成可重复使用的克隆提示（clone prompt）并缓存在内存中。后续对数百段中文文本的批量推理，均复用同一个 voice_clone_prompt，单次推理即可批量合成所有句子，避免了反复传递参考音频，显著降低了时延与推理开销。**

*设计意图：这种“分离—加载”的模式，是为了在保证自然度的前提下最大限度地降低推理成本。因为 Base 模型只需初始化一次并接受时长短、体积小的参考音频，便能进行后续所有文本的批量推理，在运行期间完全隔离了体积庞大且高成本的 VoiceDesign 模型。*

<span style="color:green">PS:我的本意是想用原始音频作为参考音频进行语音克隆的，但是部署之后发现经常出现爆显存的现象导致代码直接退出，因此退而求其次采用这种设计+克隆的方式。猜测可能是由于参考音频在提取时往往与真正的时间轴有偏差而导致前后句泄漏进来导致的程序崩坏。</span>

## 流水线处理模式
经过测试，使用NVIDIA显卡内置的硬件视频编码器保存对视频文件进行合并时对显存要求很低且显卡的负载基本在3%以下，最高不超过5%。因此本工程采用流水线模式，即上一个文件完成音轨分离、语音识别、机器翻译、AI语音合成后，下一个文件随即开始音轨分离、语音识别、机器翻译、AI语音合成，同时上一个文件开始视频合并，以减少整体处理时间。同时由于内置的硬件视频编码器数量有限，限制了同时能保存的文件队列长度=内置的硬件视频编码器数量，一旦超过这个数即停止音轨分离、语音识别、机器翻译、AI语音合成的工作，直到降至阈值以下。

## 📂输入输出规范
### 输入文件

项目要求输入的文件夹内，视频文件与同名的音轨文件需成对出现。系统会自动通过 clean_filename 函数过滤掉文件名中的特殊字符，完成初步清洗。
若该文件夹下存在旧的 `.srt` 字幕文件，系统会自动将其删除。

### 输出文件
处理完成后，系统会生成以下重要文件：

双语合并视频：根目录下，`原文件名_chn.mp4`。
日志与字幕：sub 子文件夹下，包含中文双语生成的 `.srt` 字幕文件 `原文件名_subscript.srt`。


## ❓ 注意点与故障排查

## 显卡要求
由于需要加载多个大模型，模型加载需要大约8GB的显存，且经过测试语音合成运行时会有显存瞬时暴涨的现象，建议选择显存>=32GB的型号，我租的RTX5090可以跑

### Docker容器下的问题
由于本人是在租用的服务器下运行代码，同时显卡是docker容器内启动的，因此在调用显卡内部的硬件视频加速器时遇上了(https://github.com/NVIDIA/k8s-device-plugin/issues/1282)的同款问题,由NVIDIA NVENC 硬件编码器的设计缺陷与Kubernetes 设备插件volume-mounts策略共同作用导致的。
NVIDIA NVENC 编码器在底层硬编码了一个错误的假设：NVENC 不是通过设备的主从号 (major/minor)、UUID或其他唯一标识来识别 GPU，而是直接解析/dev/nvidiaN路径中的数字部分来查找对应的 GPU 硬件。
当 NVIDIA 设备插件使用deviceListStrategy: volume-mounts模式时：
*设备插件会直接将主机上的 GPU 设备文件挂载到容器中*
*但挂载的路径不一定与 GPU 的实际索引一致*
例如：主机上索引为0的 GPU（/dev/nvidia0）可能被挂载到容器内的/dev/nvidia5路径
此时，NVENC 在容器内看到的是/dev/nvidia5，它会尝试去使用索引为5的 GPU，但容器内实际只有索引为0的 GPU 可用，导致初始化失败。

解决方法在（https://github.com/flexgrip/nvidia-gpu-enumeration/）中，核心代码如下
`COPY nvenc_fix.c /opt/nvenc_fix.c`
`RUN gcc -shared -fPIC -O2 -o /opt/libnvenc_fix.so /opt/nvenc_fix.c -ldl`
`ENV LD_PRELOAD=/opt/libnvenc_fix.so`
即移动、编译、设置环境。
`merge.py`中119~143行即执行了设置环境的功能。
如果是在本机显卡上运行此项目，只需要执行`result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)`即可。

### 运行 `run.sh` 报错 `bad interpreter: No such file or directory`
用 Vim 打开脚本，在终端进入 `run.sh` 所在目录，执行：`vim run.sh`
输入：`:set ff=unix`(修改为 Linux 格式)
继续输入：`:wq`(保存并退出 Vim)

