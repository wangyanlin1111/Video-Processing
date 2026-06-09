"""
Combined TTS API Server + Client
- VoiceSynthesisVllm 封装 IndexTTS2 模型加载、API 服务、请求发送
- 支持自适应语速调节 (speed 参数)
- 完成后自动关闭服务，释放 GPU 资源
"""

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import os
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["LOGURU_LEVEL"] = "WARNING"
os.environ["LOGURU_AUTOINIT"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["NINJA_STATUS"] = "0"
os.environ["GLOO_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["NCCL_DEBUG"] = "WARN"

import time
import io
import sys
import queue
import threading
import requests
import math
import gc
import traceback

import uvicorn
import soundfile as sf
import torch as th
import numpy as np
import pyrubberband as pyrb
import torchaudio.transforms as T

from dataclasses import asdict, dataclass
from typing import List, Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from indextts.infer_vllm_v2 import IndexTTS2
from commonfunc import converttime, get_error_detail

# ---------- 日志配置 ----------
from loguru import logger as _loguru_logger
_loguru_logger.remove()
_loguru_logger.add("logs/voice_synthesis.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.basicConfig(level=logging.WARNING)
for _name in ("vllm", "wetext", "wetext-zh_normalizer", "wetext-en_normalizer",
              "tn", "processor", "torch", "transformers"):
    logging.getLogger(_name).setLevel(logging.ERROR)
for _name in ("uvicorn", "uvicorn.access", "fastapi"):
    logging.getLogger(_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PUPPLE = "\033[95m"
RESET = "\033[0m"

# IndexTTS2 internally uses 22050 Hz
INDEXTTS_SAMPLE_RATE = 22050
Advance_Time = 50 / 1000  # 50ms
Advance_SAMPLE = int(Advance_Time * INDEXTTS_SAMPLE_RATE)

default_host = "0.0.0.0"
default_port = 6006

@dataclass
class TTSRequest:
    """语音合成请求参数"""
    text: str
    spk_audio_path: str
    emo_control_method: int = 0
    emo_ref_path: Optional[str] = None
    emo_weight: float = 1.0
    emo_vec: List[float] = None
    emo_text: Optional[str] = None
    emo_random: bool = False
    max_text_tokens_per_sentence: int = 120
    speed: float = 1.0
    max_mel_tokens: Optional[int] = None
    ref_reset: int = 1
    emo_reset: int = 1

    def __post_init__(self):
        if self.emo_vec is None:
            self.emo_vec = [0.0] * 8

    def to_dict(self) -> dict:
        return asdict(self)


class VoiceSynthesisIndexTTS2:
    """Voice synthesis using IndexTTS2 (vLLM backend) with external reference audio.

    Mirrors the structure of ``VoiceSynthesis`` (synthesis.py) but:
      - Uses IndexTTS2 instead of Qwen3TTSModel
      - Accepts externally provided ``ref_audio_path`` and ``ref_text``
    """

    def __init__(
        self,
        model_dir: str = "../local/index-tts-vllm/checkpoints/IndexTTS-2-vLLM",
        is_fp16: bool = True,
        gpu_memory_utilization: float = 0.25,
        qwenemo_gpu_memory_utilization: float = 0.10,
        load_qwen_emo=False,
        enable_sleep_mode=True
    ):
        """
        Args:
            model_dir: Path to the IndexTTS2 model directory.
            is_fp16: Whether to use FP16 inference.
            gpu_memory_utilization: GPU memory fraction for the main model.
            qwenemo_gpu_memory_utilization: GPU memory fraction for the emotion model.
            load_qwen_emo: Whether to load the QwenEmotion model (for text-based emotion control).
            enable_sleep_mode: Whether to enable sleep mode for memory management.
        """
        self.vocalpath = "vocal_clone_indextts.mp3"
        self.host = default_host
        self.port = default_port
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.dtype = th.bfloat16 if self.device == "cuda" else th.float32
        self.samplerate = INDEXTTS_SAMPLE_RATE  # IndexTTS2 outputs 22050 Hz

        self.model_dir = model_dir
        self.is_fp16 = is_fp16
        self.gpu_memory_utilization = gpu_memory_utilization
        self.qwenemo_gpu_memory_utilization = qwenemo_gpu_memory_utilization
        self.load_qwen_emo = load_qwen_emo
        self.enable_sleep_mode = enable_sleep_mode
        """Caches for resamplers and silence tensors to avoid redundant computation"""
        self._resampler_cache = {}
        self._silence_cache = {}
        """Service  management"""
        self._server_process = None
        self._url = f"http://127.0.0.1:{self.port}/tts_url"
        self._health_url = f"http://127.0.0.1:{self.port}/health"

    # ------------------------------------------------------------------
    # Server management (from api_server.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _create_app(model_dir, is_fp16, gpu_memory_utilization,
                    qwenemo_gpu_memory_utilization, load_qwen_emo, enable_sleep_mode):
        tts = None

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            nonlocal tts
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            stdout_fd, stderr_fd = sys.stdout.fileno(), sys.stderr.fileno()
            saved_out, saved_err = os.dup(stdout_fd), os.dup(stderr_fd)
            # 重定向到日志文件而非 /dev/null，以便排查 vLLM 启动错误
            log_fd = os.open("logs/vllm_startup.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.dup2(log_fd, stdout_fd)
            os.dup2(log_fd, stderr_fd)
            _loguru_logger.info("Loading IndexTTS2 model...")
            try:
                tts = IndexTTS2(
                    model_dir=model_dir, is_fp16=is_fp16,
                    gpu_memory_utilization=gpu_memory_utilization,
                    qwenemo_gpu_memory_utilization=qwenemo_gpu_memory_utilization,
                    load_qwen_emo=load_qwen_emo,
                    enable_sleep_mode=enable_sleep_mode,
                )
                _loguru_logger.info("Model loaded successfully.")
            finally:
                os.dup2(saved_out, stdout_fd)
                os.dup2(saved_err, stderr_fd)
                os.close(saved_out); os.close(saved_err); os.close(log_fd); os.close(devnull_fd)
            yield
            _loguru_logger.info("Shutting down model...")
            del tts
            import gc; import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _loguru_logger.info("GPU resources released.")

        app = FastAPI(lifespan=lifespan)
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                           allow_methods=["*"], allow_headers=["*"])

        @app.get("/health")
        async def health():
            if tts is None:
                return JSONResponse(status_code=503, content={"status": "unhealthy"})
            return JSONResponse(status_code=200, content={"status": "healthy", "timestamp": time.time()})

        @app.post("/sleep")
        async def sleep_endpoint(request: Request):
            data = await request.json()
            level = data.get("level", 1)
            await tts.sleep(level=level)
            return JSONResponse(status_code=200, content={"status": "sleeping", "level": level})

        @app.post("/wake_up")
        async def wake_up_endpoint():
            await tts.wake_up()
            return JSONResponse(status_code=200, content={"status": "awake"})

        @app.post("/reset_cache")
        async def reset_cache_endpoint():
            await tts.reset_engine_cache()
            return JSONResponse(status_code=200, content={"status": "cache_cleared"})

        @app.post("/tts_url")
        async def tts_endpoint(request: Request):
            try:
                data = await request.json()
                emo_control_method = data.get("emo_control_method", 0)
                if not isinstance(emo_control_method, int):
                    emo_control_method = emo_control_method.value

                emo_ref_path = data.get("emo_ref_path", None)
                emo_weight = data.get("emo_weight", 1.0)
                vec = None
                if emo_control_method == 0:
                    emo_ref_path, emo_weight = None, 1.0
                elif emo_control_method == 2:
                    vec = data.get("emo_vec", [0] * 8)
                    if sum(vec) > 1.5:
                        return JSONResponse(status_code=500,
                            content={"status": "error", "error": "情感向量之和不能超过1.5"})

                sr, wav = await tts.infer(
                    spk_audio_prompt = data["spk_audio_path"],
                    text = data["text"],
                    output_path = None,
                    emo_audio_prompt = emo_ref_path,
                    emo_alpha = emo_weight,
                    emo_vector = vec,
                    use_emo_text = (emo_control_method == 3),
                    emo_text = data.get("emo_text", None),
                    use_random = data.get("emo_random", False),
                    max_text_tokens_per_sentence = int(data.get("max_text_tokens_per_sentence", 120)),
                    speed = data.get("speed", 1.0),
                    max_mel_tokens = data.get("max_mel_tokens", None),
                    ref_reset = data.get("ref_reset", True),
                    emo_reset = data.get("emo_reset", True),
                )
                with io.BytesIO() as buf:
                    sf.write(buf, wav, sr, format='WAV')
                    return Response(content=buf.getvalue(), media_type="audio/wav")
            except Exception as ex:
                tb = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
                return JSONResponse(status_code=500, content={"status": "error", "error": str(tb)})

        return app

    @property
    def is_running(self) -> bool:
        return self._server_process is not None and self._server_process.is_alive()

    def start(self) -> None:
        if self._server_process is not None:
            return
        # Clear CUDA-tensor caches to avoid pickling errors when spawning subprocess
        self._resampler_cache.clear()
        self._silence_cache.clear()
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
        _loguru_logger.info("Starting TTS server subprocess...")
        t0 = time.time()
        self._server_process = multiprocessing.Process(
            target=VoiceSynthesisIndexTTS2._run_server,
            args=(self.model_dir, self.is_fp16, self.gpu_memory_utilization,
                  self.qwenemo_gpu_memory_utilization, self.load_qwen_emo,
                  self.enable_sleep_mode, self.host, self.port),
            name="tts_server",
        )
        self._server_process.start()
        logger.info(f"{GREEN}Vocal Synthesis via vLLM Init Time: {time.time() - t0:.4f}s{RESET}")
        if th.cuda.is_available():
            free, total = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}Vocal Synthesis via vLLM: All: {total / 1024 ** 3:.2f} GB, Free: {free / 1024 ** 3:.2f} GB{RESET}")
        self._wait_ready()

    @staticmethod
    def _run_server(model_dir, is_fp16, gpu_memory_utilization,
                    qwenemo_gpu_memory_utilization, load_qwen_emo,
                    enable_sleep_mode, host, port) -> None:
        # expandable_segments 与 vLLM 内存池不兼容，子进程中必须清除
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        app = VoiceSynthesisIndexTTS2._create_app(
            model_dir=model_dir, is_fp16=is_fp16,
            gpu_memory_utilization=gpu_memory_utilization,
            qwenemo_gpu_memory_utilization=qwenemo_gpu_memory_utilization,
            load_qwen_emo=load_qwen_emo,
            enable_sleep_mode=enable_sleep_mode,
        )
        uvicorn.run(app, host=host, port=port, log_level="warning")

    def _wait_ready(self, timeout: float = 300.0, interval: float = 2.0) -> None:
        start = time.time()
        while time.time() - start < timeout:
            # Early exit: subprocess crashed
            if self._server_process is not None and not self._server_process.is_alive():
                exitcode = self._server_process.exitcode
                raise RuntimeError(
                    f"TTS server process exited prematurely with code {exitcode}. "
                    f"Check logs/vllm_startup.log for details."
                )
            try:
                if requests.get(self._health_url, timeout=5).status_code == 200:
                    _loguru_logger.info("Server is healthy and ready.")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(interval)
        raise TimeoutError(f"Server did not become healthy within {timeout}s")

    def shutdown(self, force: bool = False) -> None:
        if self._server_process is None:
            return
        _loguru_logger.info("Shutting down server...")
        self._server_process.terminate()
        self._server_process.join(timeout=30)
        if self._server_process.is_alive():
            if force:
                _loguru_logger.warning("Force killing server...")
                self._server_process.kill()
                self._server_process.join()
        self._server_process = None
        gc.collect()
        if th.cuda.is_available():
            th.cuda.synchronize()
            th.cuda.empty_cache()
            free, total = th.cuda.mem_get_info()
            _loguru_logger.info(f"GPU: {total / 1024**3:.2f} GB total, {free / 1024**3:.2f} GB free")
        _loguru_logger.info("Server stopped, GPU resources released.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_resampler(self, target_freq: int) -> T.Resample:
        key = (target_freq, self.samplerate)
        if key not in self._resampler_cache:
            resampler = T.Resample(
                orig_freq=self.samplerate,
                new_freq=target_freq,
                resampling_method="sinc_interp_kaiser",
                lowpass_filter_width=64,
                rolloff=0.95,
                beta=14.0,
            ).to(self.device)
            self._resampler_cache[key] = resampler
        return self._resampler_cache[key]

    def generate_silence(self, duration_in_samples: int) -> th.Tensor:
        key = int(duration_in_samples)
        if key not in self._silence_cache:
            self._silence_cache[key] = th.zeros(
                key, dtype=self.dtype, device=self.device
            )
        return self._silence_cache[key].clone()
    
    def sleep(self, level: int = 1) -> None:
        """
        VLLM Memory Management:
        level=1: Release KV/MM cache (release ~30% GPU memory, wake up is fast)
        level=2: Unload model weights to CPU (release ~80% GPU memory, wake up requires re-loading)
        """
        _loguru_logger.info(f"Sending sleep command (level={level})...")
        resp = requests.post(f"http://127.0.0.1:{self.port}/sleep", json={"level": level})
        if resp.status_code != 200:
            raise RuntimeError(f"Sleep failed: {resp.text}")
        _loguru_logger.info("Engine is now sleeping.")

    def wake_up(self) -> None:
        """Wake up the engine from sleep mode. If level=2 sleep was used, this will require re-loading the model weights."""
        _loguru_logger.info("Waking up engine...")
        resp = requests.post(f"http://127.0.0.1:{self.port}/wake_up")
        if resp.status_code != 200:
            raise RuntimeError(f"Wake up failed: {resp.text}")
        _loguru_logger.info("Engine is awake.")

    def reset_cache(self) -> None:
        """Reset the KV/MM cache (does not enter sleep mode)"""
        resp = requests.post(f"http://127.0.0.1:{self.port}/reset_cache")
        if resp.status_code != 200:
            raise RuntimeError(f"Reset cache failed: {resp.text}")
        _loguru_logger.info("Cache cleared.")


    def synthesize_single(self, text: str, spk_audio_path: str, speed: float, max_mel_tokens: int, reset: int, **kwargs) -> th.Tensor:
        try:    
            req = TTSRequest(
                text=text, 
                spk_audio_path=spk_audio_path, 
                speed=speed, 
                max_mel_tokens = max_mel_tokens,
                ref_reset = reset, 
                emo_reset = reset, 
                **kwargs
            )
            resp = requests.post(self._url, json=req.to_dict())
            if resp.status_code != 200:
                raise RuntimeError(f"Synthesis failed (status={resp.status_code}): {resp.text}")
            wav_bytes = resp.content
            audio, _ = sf.read(io.BytesIO(wav_bytes))
            audio = th.from_numpy(audio).to(device=self.device, dtype=self.dtype)
            return audio
        except Exception as e:
            error_detail = get_error_detail(e)
            logger.error(f"{RED}{error_detail}{RESET}")
            raise RuntimeError(f"Single generation failed: {text[:60]}...") from e

    # ------------------------------------------------------------------
    # Streaming synthesis (write to file incrementally — avoids OOM)
    # ------------------------------------------------------------------
    def synthesize_parallel(
        self,
        total_sentences: list,
        sampling_rate: int,
        ori_len: float,
        ori_sr: float = 1.0,
        debug_en: bool = False,
        vocal_path: str = None,
        ref_audio_path: str = None,
    ):
        """Stream synthesised audio directly to a file, batch by batch.
        Identical semantics to :meth:`synthesize` but writes incrementally
        via a background thread to avoid holding the full waveform in memory.
        """

        total_start_time = time.time()
        self.start()
        N = len(total_sentences)
        if debug_en:
            file = open("regeneration_log_indextts.txt", "w")
        _vocal_path = vocal_path if vocal_path is not None else self.vocalpath
        if os.path.isfile(_vocal_path):
            os.remove(_vocal_path)
        total_len = 0

        write_queue = queue.Queue()

        def writer(sf_file):
            while True:
                item = write_queue.get()
                if item is None:
                    break
                sf_file.write(item)

        with sf.SoundFile(_vocal_path, "w", samplerate=sampling_rate, channels=2) as f:
            writer_thread = threading.Thread(target=writer, args=(f,), daemon=True)
            writer_thread.start()
            num_advance = 0
            advance_flag = 0
            prev_end_sample = 0
            for i in range(N):
                resetflag = 1 if i == 0 else 0 
                audio_segment = th.tensor([], dtype = self.dtype, device = self.device)
                texts = total_sentences[i]["chinese"].strip()
                max_mel_tokens = math.ceil((total_sentences[i]["end"]- total_sentences[i]["start"] + 1) * INDEXTTS_SAMPLE_RATE / 256)
                audio = self.synthesize_single(
                    text = texts, 
                    spk_audio_path = ref_audio_path, 
                    speed = 0.85, 
                    max_mel_tokens = max_mel_tokens, 
                    reset = resetflag
                )
                current_length = len(audio)
                target_length = (total_sentences[i]["end"]- total_sentences[i]["start"]) * self.samplerate
                if len(audio) > target_length:
                    real_calibration_factor = (math.ceil(len(audio) / target_length * 100) / 100.0)
                    calibration_factor = min(real_calibration_factor, 1.15)
                    audio_np = audio.squeeze().cpu().to(th.float32).numpy()
                    stretched_audio_np = pyrb.time_stretch(audio_np, self.samplerate, rate=calibration_factor)
                    audio = th.as_tensor(stretched_audio_np, device=self.device, dtype=self.dtype)
                    if calibration_factor < real_calibration_factor:
                        diff_samples = len(audio) - target_length
                        num_advance += int(diff_samples // Advance_SAMPLE)
                        advance_flag = 1
                        if debug_en:
                            content = (
                                f"The {i}th data, "
                                f"Calibration Factor = {calibration_factor}, "
                                f"Target = {target_length}, "
                                f"Original = {current_length}, "
                                f"Updated = {len(audio)}\n"
                            )
                            file.write(content)
                start_sample = round(total_sentences[i]["start"]* self.samplerate)
                silence_samples = max(start_sample - prev_end_sample, 0)
                if advance_flag and silence_samples > Advance_SAMPLE:
                    silence_samples -= Advance_SAMPLE
                    num_advance -= 1
                    advance_flag = 1 if num_advance > 0 else 0
                temp_silence = self.generate_silence(silence_samples)
                audio_segment = th.concatenate([audio_segment, temp_silence])
                prev_end_sample += silence_samples + len(audio)
                del temp_silence
                audio_segment = th.concatenate([audio_segment, audio])

                if sampling_rate != self.samplerate:
                    resampler = self.get_resampler(target_freq=sampling_rate)
                    audio_to_save = resampler(audio_segment.float())
                else:
                    audio_to_save = audio_segment
                total_len += len(audio_to_save)

                if audio_to_save.dim() == 1:
                    audio_to_save = audio_to_save.repeat(2, 1)
                write_queue.put(audio_to_save.cpu().numpy().T)

                # Clean up
                del audio_to_save, audio_segment, audio
                gc.collect()
                if self.device == "cuda":
                    th.cuda.synchronize()
                    th.cuda.empty_cache()

            # Pad to original video length
            target_samples_ori = int(round(ori_len * sampling_rate))
            if int(total_len * ori_sr) < target_samples_ori:
                numzeros = int(round(target_samples_ori / ori_sr)) - int(total_len)
                if numzeros > 0:
                    padding = np.zeros((numzeros, 2), dtype=np.float32)
                    write_queue.put(padding)

            write_queue.put(None)
            writer_thread.join()
        """Sleep after synthesis to free GPU memory for subsequent processing steps"""
        if self.device == "cuda":
            free, _ = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}Before vllm shut down Free: {free / 1024 ** 3:.2f} GB{RESET}")
        self.shutdown()
        if debug_en:
            file.close()

        gc.collect()
        if self.device == "cuda":
            th.cuda.synchronize()
            th.cuda.empty_cache()
            free, _ = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}After vllm shut down Free: {free / 1024 ** 3:.2f} GB{RESET}")
            mem_ava_gb =  free / (1024 ** 3) - 1.0
        total_end_time = time.time()
        hours, minutes, seconds, ms = converttime(total_end_time - total_start_time)
        logger.info(f"{GREEN}Synthesis Time is {hours}h {minutes}m {seconds}s {ms}ms{RESET}")
        return mem_ava_gb

