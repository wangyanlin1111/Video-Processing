# import time
# import requests
# import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# expandable_segments is incompatible with vLLM memory pool, must NOT be set
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Disable all vLLM log/tdqm/auto optimization
os.environ["VLLM_LOGGING_LEVEL"] = "CRITICAL"
os.environ["VLLM_DISABLE_PROGRESS_BARS"] = "1"
os.environ["VLLM_HIDE_CUDA_GRAPHS_PROGRESS"] = "1" 
os.environ["FLASHINFER_AUTOTUNE_QUIET"] = "1"
os.environ["FLASHINFER_LOGGING_LEVEL"] = "CRITICAL"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

# Disable all EngineCore/Autotuner log
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cuda.cudart.*")
warnings.filterwarnings("ignore", message=".*cuda.nvrtc.*")
warnings.filterwarnings("ignore", message=".*flashinfer.*")
warnings.filterwarnings("ignore", message=".*Autotuner.*")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("vllm").setLevel(logging.CRITICAL)
logging.getLogger("vllm.engine").setLevel(logging.INFO)
logging.getLogger("flashinfer").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

import gc
import torch as th
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from commonfunc import converttime

MODEL_NAME ="tencent/HY-MT1.5-1.8B"
VLLM_BATCH_SIZE = 50

RED = "\033[91m"    
GREEN = "\033[92m"  
YELLOW = "\033[93m"  
BLUE = "\033[94m" 
PUPPLE = "\033[95m"     
RESET = "\033[0m"

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

def merge_srt_chinese(english_list, chinese_list):
    """
    Merge english content and chinese content together (Increase length check)
    """
    merged = []
    """Take the shorest length to avoid exception"""
    min_len = min(len(english_list), len(chinese_list))
    if len(english_list) != len(chinese_list):
        logger.warning(f"English sentences count({len(english_list)}) != Chinese sentences count({len(chinese_list)})")

    for eng_dict, chi_text in zip(english_list[:min_len], chinese_list[:min_len]):
        merged_item = {
            "start": eng_dict["start"],
            "end": eng_dict["end"],
            "english": eng_dict["text"],
            "chinese": chi_text.strip() if chi_text else ""
        }
        merged.append(merged_item)
    return merged

def save_as_srt(segments, output_path):
    """
    Save as Bilingual SRT
    """
    if not segments:
        logger.warning("No segments to save into SRT file")
        return
    try:
        if os.path.isfile(output_path):
            os.remove(output_path) 
        with open(output_path, "w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(segments, start=1):
                start_time = format_time(segment["start"])
                end_time = format_time(segment["end"])
                chinese = segment.get("chinese", "").strip()
                english = segment.get("english", "").strip()
                srt_file.write(f"{i}\n")
                srt_file.write(f"{start_time} --> {end_time}\n")
                if chinese:
                    srt_file.write(f"{chinese}\n")
                if english:
                    srt_file.write(f"{english}\n\n")
    except Exception as e:
        """Throw Exception"""
        logger.error(f"{RED}Failed to save SRT file: {e}{RESET}")
        raise

class SubscriptTranslation:
    """
    option 0: Transformer; 1: Ollama; 2: vllm
    """
    def __init__(self, option : int = None, model_path: str = None, gpu_memory_utilization: float = None):
        self.OPTION = option if option is not None else 0
        self.DEVICE = "cuda" if th.cuda.is_available() else "cpu"
        self.MODEL_NAME = model_path if model_path is not None else MODEL_NAME
        init_start_time = time.time()
        if self.OPTION == 0:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=th.bfloat16,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2",
                    device_map=self.DEVICE
                )
            except Exception as e:
                """Throw Exception"""
                logger.error(f"{RED}Failed to initialize Transformer service: {e}{RESET}")
                raise
        elif self.OPTION == 2:
            self.GPU_MENORY_UTILIZATION = gpu_memory_utilization if gpu_memory_utilization is not None else 0.75
        init_end_time = time.time()
        logger.info(f"{GREEN}Subscript Translation Init Time: {init_end_time - init_start_time:.4f}s{RESET}")
        translate_method = "transformer" if self.OPTION == 0 else "vllm"
        if th.cuda.is_available():
            free, total = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}Subscript Translation via {translate_method}: All: {total / 1024 ** 3:.2f} GB, Free: {free / 1024 ** 3:.2f} GB{RESET}")
        

    def translate_via_vllm(self, english_sentences) -> list[str]:
        """
        Translate in batch manner
        """
        try:
            init_start_time = time.time()
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=512,
                stop=["<|end|>", "</s>", "\n\n"],
                repetition_penalty=1.05
            )
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, 
                trust_remote_code=True
            )
            llm = LLM(
                model=MODEL_NAME,
                disable_log_stats=True,
                use_tqdm_on_load=False,
                gpu_memory_utilization=self.GPU_MENORY_UTILIZATION,
                trust_remote_code=True,
                dtype="bfloat16",               
                max_model_len=8192,
                max_num_batched_tokens=8192,
                enforce_eager=False, 
                max_num_seqs=32,
                kv_cache_dtype="fp8",
                quantization=None,
            )
            init_end_time = time.time()
            logger.info(f"{GREEN}Translate via vllm initialize Time: {init_end_time - init_start_time:.4f}s{RESET}")
            """Build Batch prompts"""
            start_time = time.time()
            iteration_num = (len(english_sentences) + VLLM_BATCH_SIZE - 1) // VLLM_BATCH_SIZE
            chinese_sentences = []
            for i in range(iteration_num):
                prompts = []
                startpoint = i * VLLM_BATCH_SIZE
                endpoint = min((i + 1) * VLLM_BATCH_SIZE, len(english_sentences))
                batch_sentences = english_sentences[startpoint:endpoint]
                text_batch = [sentence["text"] for sentence in batch_sentences]
                for text in text_batch:
                    # Truncate text if tokenized prompt would exceed model context
                    max_input_tokens = 8192 - 512  # reserve 512 for output
                    messages = [{"role": "user", "content": f"Translate the following English text into Chinese without any additional explanation and keep the original long sentence structure:\n\n{text}"}]
                    prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    # Safety: truncate from raw text end if prompt still too long
                    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    if len(token_ids) > max_input_tokens:
                        truncated_ids = token_ids[:max_input_tokens]
                        prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)
                    prompts.append(prompt)
                """Batch Translation"""
                outputs = llm.generate(
                    prompts, 
                    sampling_params,
                    use_tqdm=False
                )
                for output in outputs:
                    if output.outputs:
                        chinese_sentences.append(output.outputs[0].text.strip())
                    else:
                        chinese_sentences.append("")
            end_time = time.time()
            logger.info(f"{GREEN}Translate via vllm service Time: {end_time - start_time:.4f}s{RESET}")
            return chinese_sentences
        except Exception as e:
            """Throw Exception"""
            logger.error(f"{RED}Failed to initialize vllm service: {e}{RESET}")
            raise       

    def translate_via_transformer(self, english_sentences) -> list[str]:
        """
        Translate in a one-by-one manner
        """
        chinese_sentences = []
        for sentence in english_sentences:
            text = sentence["text"]
            # messages = [{"role": "user", "content": f"Translate the following segment into Chinese without additional explanation and keep the original long sentence structure.\n\n{text}"}]
            system_prompt = (
                "你是专业的英译中翻译引擎。请严格遵循以下规则：\n"
                "- 精确翻译给定的英文内容，不做任何额外解释。\n"
                "- 严禁在译文中使用任何标点符号，包括逗号、顿号、分号等。\n"
                "- 译文必须是一个完整的长句，中间不能有任何停顿或分隔。\n"
                "- 保持原文的句子结构，但把所有信息整合到一个连贯的句子中。"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            with th.no_grad():
                outputs = self.model.generate(
                    prompt.to(self.DEVICE),
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=False,
                    num_beams=1,                         # Greedy decoding, most stable output
                    repetition_penalty=1.2,         # Slightly suppress repetition, enhances sentence diversity
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            input_length = prompt.shape[1]
            generated_tokens = outputs[0][input_length:]
            translated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            if translated_text:
                chinese_sentences.append(translated_text)
            else:
                chinese_sentences.append("")

        return chinese_sentences
        

    def Subscript_Translation_Srt_Generation(self, english_sentences, output_srt_path:str = None):
        total_start_time = time.time()

        if (self.OPTION == 0):
            chinese_sentences = self.translate_via_transformer(english_sentences)
        elif(self.OPTION == 2):
            chinese_sentences = self.translate_via_vllm(english_sentences)

        """Merge English/Chinese/time info"""
        total_sentences = merge_srt_chinese(english_sentences, chinese_sentences)

        """Save Bilingual SRT"""
        output_srt_path = output_srt_path if output_srt_path is not None else "subscript.srt"
        save_as_srt(total_sentences, output_srt_path)

        """Clear Memory"""
        gc.collect()
        if self.DEVICE == "cuda":
            th.cuda.synchronize()
            th.cuda.empty_cache()
            free, _ = th.cuda.mem_get_info()
            logger.info(f"{YELLOW}After Speech Translation Free: {free / 1024 ** 3:.2f} GB{RESET}")
        total_end_time = time.time()
        hours,minutes,seconds,miliseconds = converttime(total_end_time - total_start_time)
        logger.info(f"{GREEN}Translate and SRT saving Time is {hours} hours, {minutes} minutes, {seconds} seconds and {miliseconds} ms{RESET}")
        return total_sentences
    # def __init__(
    #         self,
    #         src_lang: str = None,
    #         tgt_lang: str = None,
    #         enable_batch: bool = False,
    #         num_worders: int = 1,
    #         output_srt_path: str = None
    # ):

    # self.SRC_LANGUAGE = src_lang if src_lang is not None else "English"
    # self.TGT_LANGUAGE = tgt_lang if tgt_lang is not None else "Chinese"
    # self.ENABLE_BATCH = False if enable_batch is None else enable_batch
    # self.NUM_WORKERS = 1 if num_worders is None else num_worders
    # self.SRT_PATH = "subscript.srt" if num_worders is None else output_srt_path
    # self.URL = "http://localhost:11434/api/generate"

    # def translate_batch(self, english_sentences):
    #     """
    #     Batch translation
    #     """
    #     chinese_sentences = []
    #     with ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as executor:
    #         future_to_idx = {
    #             executor.submit(self.translate_via_ollama, sentence["text"]): idx
    #             for idx, sentence in enumerate(english_sentences)
    #         }

    #         with tqdm(total=len(english_sentences), desc="Batch Translation", unit="Sentence") as pbar:
    #             for future in as_completed(future_to_idx):
    #                 idx = future_to_idx[future]
    #                 try:
    #                     chinese_sentences.append(future.result())
    #                 except Exception as e:
    #                     logger.error(f"Translation the {idx + 1}th sentence fail: {e}")
    #                     chinese_sentences[idx] = ""
    #                 pbar.update(1)
    #     return chinese_sentences

    # def translate_one_by_one(self, english_sentences):
    #     """
    #     One-by-one translation
    #     """
    #     chinese_sentences = []
    #     for i, sentence in enumerate(
    #             tqdm(english_sentences, desc="One-by-one Translation", unit="Sentence")
    #     ):
    #         try:
    #             chinese_sentences.append(self.translate_via_ollama(sentence["text"]))
    #         except Exception as e:
    #             logger.error(f"Translation the {i + 1}th sentence fail: {e}")
    #             chinese_sentences.append("")
    #     return chinese_sentences

    # def translate_via_ollama(self, text: str) -> str:
    #     """
    #     Using HY-MT1.5-1.8B through Ollama API to translate english into chinese
    #     """
    #     """Construct Prompt"""
    #     prompt = f"Translate the following {self.SRC_LANGUAGE} text to {self.TGT_LANGUAGE}: {text}"
    #     """Construct request"""
    #     payload = {
    #         "model": "demonbyron/HY-MT1.5-1.8B:bf16",
    #         "prompt": prompt,
    #         "stream": False,
    #         "options": {
    #             "temperature": 0,
    #         }
    #     }
    #     """Send request"""
    #     response = requests.post(self.URL, json=payload)
    #     if response.status_code == 200:
    #         result = response.json()
    #         return result["response"].strip()
    #     else:
    #         return f"Error: {response.status_code}"

    # def Subscript_Translation_Srt_Generation(self, english_sentences):
    #     total_start_time = time.time()
    #     if self.ENABLE_BATCH and len(english_sentences) > 1:
    #         chinese_sentences = self.translate_batch(english_sentences)
    #     else:
    #         chinese_sentences = self.translate_one_by_one(english_sentences)
    #     """Merge English/Chinese/time info"""
    #     total_sentences = merge_srt_chinese(english_sentences, chinese_sentences)

    #     """Save Bilingual SRT"""
    #     save_as_srt(total_sentences, self.SRT_PATH)
    #     total_end_time = time.time()
    #     logger.info(f"Translate All Time：{total_end_time - total_start_time:.4f}s")
    #     return total_sentences


