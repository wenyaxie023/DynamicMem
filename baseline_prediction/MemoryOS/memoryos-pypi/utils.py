import time
import uuid
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import inspect
from functools import wraps
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List
try:
    from . import prompts # 尝试相对导入
except ImportError:
    import prompts # 回退到绝对导入
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

_USAGE_LOCK = threading.Lock()
_USAGE_RECORDS = []


def _usage_to_dict(raw_usage):
    if hasattr(raw_usage, "model_dump"):
        raw_usage = raw_usage.model_dump(mode="json", by_alias=True)
    elif hasattr(raw_usage, "to_dict"):
        raw_usage = raw_usage.to_dict()
    if isinstance(raw_usage, dict):
        return raw_usage
    return {}


def _record_usage(request_kind, model, usage):
    usage_dict = _usage_to_dict(usage)
    completion_details = _usage_to_dict(
        usage_dict.get("completion_tokens_details") or usage_dict.get("output_tokens_details")
    )
    prompt_tokens = int(usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens") or 0)
    completion_tokens = int(usage_dict.get("completion_tokens") or usage_dict.get("output_tokens") or 0)
    reasoning_tokens = int(usage_dict.get("reasoning_tokens") or completion_details.get("reasoning_tokens") or 0)
    total_tokens = int(usage_dict.get("total_tokens") or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    record = {
        "request_kind": str(request_kind or ""),
        "model": str(model or ""),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "timestamp_unix": time.time(),
    }
    with _USAGE_LOCK:
        _USAGE_RECORDS.append(record)
    return record


def reset_usage_tracker():
    with _USAGE_LOCK:
        _USAGE_RECORDS.clear()


def get_usage_records():
    with _USAGE_LOCK:
        return [dict(x) for x in _USAGE_RECORDS]


def get_usage_summary():
    records = get_usage_records()
    summary = {
        "request_count": len(records),
        "chat_request_count": 0,
        "embedding_request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "by_model": [],
    }
    by_model = {}
    for record in records:
        request_kind = str(record.get("request_kind") or "")
        if request_kind == "chat_completion":
            summary["chat_request_count"] += 1
        elif request_kind == "embedding":
            summary["embedding_request_count"] += 1
        summary["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
        summary["completion_tokens"] += int(record.get("completion_tokens") or 0)
        summary["reasoning_tokens"] += int(record.get("reasoning_tokens") or 0)
        summary["total_tokens"] += int(record.get("total_tokens") or 0)
        key = (request_kind, str(record.get("model") or ""))
        bucket = by_model.setdefault(
            key,
            {
                "request_kind": key[0],
                "model": key[1],
                "request_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        bucket["request_count"] += 1
        bucket["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += int(record.get("completion_tokens") or 0)
        bucket["total_tokens"] += int(record.get("total_tokens") or 0)
    summary["by_model"] = list(by_model.values())
    return summary

def clean_reasoning_model_output(text):
    """
    清理推理模型输出中的<think>标签
    适配推理模型（如o1系列）的输出格式
    """
    if not text:
        return text
    
    import re
    # 移除<think>...</think>标签及其内容
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 清理可能产生的多余空白行
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    # 移除开头和结尾的空白
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

# ---- OpenAI Client ----
class OpenAIClient:
    def __init__(self, api_key, base_url=None, max_workers=5):
        self.api_key = api_key
        self.base_url = base_url if base_url else "https://api.openai.com/v1"
        # The openai library looks for OPENAI_API_KEY and OPENAI_BASE_URL env vars by default
        # or they can be passed directly to the client.
        # For simplicity and explicit control, we'll pass them to the client constructor.
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        print(f"Calling OpenAI API. Model: {model}")
        try:
            request_args = {
                "model": model,
                "messages": messages,
            }
            if model and str(model).lower().startswith("gpt-5"):
                request_args["max_completion_tokens"] = max_tokens
                if temperature is None or float(temperature) == 1.0:
                    request_args["temperature"] = 1
            else:
                request_args["max_tokens"] = max_tokens
                request_args["temperature"] = temperature
            response = self.client.chat.completions.create(**request_args)
            _record_usage("chat_completion", model, getattr(response, "usage", None))
            raw_content = response.choices[0].message.content.strip()
            # 自动清理推理模型的<think>标签
            cleaned_content = clean_reasoning_model_output(raw_content)
            return cleaned_content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            # Fallback or error handling
            return "Error: Could not get response from LLM."

    def chat_completion_async(self, model, messages, temperature=0.7, max_tokens=2000):
        """异步版本的chat_completion"""
        return self.executor.submit(self.chat_completion, model, messages, temperature, max_tokens)

    def batch_chat_completion(self, requests):
        """
        并行处理多个LLM请求
        requests: List of dict with keys: model, messages, temperature, max_tokens
        """
        futures = []
        for req in requests:
            future = self.chat_completion_async(
                model=req.get("model", "gpt-4o-mini"),
                messages=req["messages"],
                temperature=req.get("temperature", 0.7),
                max_tokens=req.get("max_tokens", 2000)
            )
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in batch completion: {e}")
                results.append("Error: Could not get response from LLM.")
        
        return results

    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)

# ---- Parallel Processing Utilities ----
def run_parallel_tasks(tasks, max_workers=3):
    """
    并行执行任务列表
    tasks: List of callable functions
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in parallel task: {e}")
                results.append(None)
        return results

# ---- Basic Utilities ----
def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def generate_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def ensure_directory_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ---- Embedding Utilities ----
_model_cache = {}
_embedding_cache = {}  # 添加embedding缓存

def _get_valid_kwargs(func, kwargs):
    """Helper to filter kwargs for a given function's signature."""
    try:
        sig = inspect.signature(func)
        param_keys = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in param_keys}
    except (ValueError, TypeError):
        # Fallback for functions/methods where signature inspection is not straightforward
        return kwargs

_EMBEDDING_MODEL_ALIASES = {
    "minilm-l6-v2": "all-MiniLM-L6-v2",
    "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
}

def _normalize_model_name(model_name: str) -> str:
    if not model_name:
        return model_name
    name = model_name.strip()
    alias = _EMBEDDING_MODEL_ALIASES.get(name)
    if alias:
        return alias
    alias = _EMBEDDING_MODEL_ALIASES.get(name.lower())
    return alias or model_name

def _infer_embedding_backend(model_name: str) -> str:
    name = model_name.lower()
    if name in {"bm25", "bm25-okapi"}:
        return "bm25"
    if "contriever" in name:
        return "contriever"
    if name.startswith("text-embedding-") or name.startswith("openai/") or name.startswith("litellm/"):
        return "litellm"
    return "sentence-transformers"

def _normalize_embedding_backend(embedding_backend: str, model_name: str) -> str:
    backend = embedding_backend or _infer_embedding_backend(model_name)
    backend = backend.lower()
    if backend in {"sentence_transformers", "sentence-transformer", "st"}:
        return "sentence-transformers"
    if backend in {"openai"}:
        return "litellm"
    if backend in {"hf", "huggingface"}:
        return "contriever"
    return backend

def resolve_embedding_backend(model_name: str, embedding_model_kwargs: Optional[dict] = None) -> str:
    embedding_model_kwargs = embedding_model_kwargs or {}
    return _normalize_embedding_backend(embedding_model_kwargs.get("embedding_backend"), model_name)

def _supports_sentence_transformer_trust_remote_code() -> bool:
    try:
        signature = inspect.signature(SentenceTransformer.__init__)
    except (TypeError, ValueError):
        return False
    if "trust_remote_code" in signature.parameters:
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False

def _should_trust_remote_code(model_name: str) -> bool:
    return "qwen" in model_name.lower()

def _is_unknown_model_type_error(error: Exception) -> bool:
    message = str(error).lower()
    return "does not recognize this architecture" in message or "model type `qwen" in message

def _tokenize_bm25(text: str) -> List[str]:
    import re
    return re.findall(r"[a-z0-9]+", text.lower())

def compute_bm25_scores(corpus_texts: List[str], query_text: str) -> np.ndarray:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "Please install rank_bm25 to use bm25: pip install rank_bm25"
        ) from exc

    tokenized_corpus = [_tokenize_bm25(text) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = _tokenize_bm25(query_text)
    scores = bm25.get_scores(query_tokens)
    return np.array(scores, dtype=np.float32)

def get_embedding(text, model_name="all-MiniLM-L6-v2", use_cache=True, **kwargs):
    """
    获取文本的embedding向量。
    支持多种主流模型，能自动适应不同库的调用方式。
    - SentenceTransformer模型: e.g., 'all-MiniLM-L6-v2', 'Qwen/Qwen3-Embedding-0.6B'
    - FlagEmbedding模型: e.g., 'BAAI/bge-m3'
    - Contriever模型: e.g., 'facebook/contriever'
    - LiteLLM embeddings: e.g., 'text-embedding-3-large', 'Qwen3-Embedding-8B'

    :param text: 输入文本。
    :param model_name: Hugging Face上的模型名称。
    :param use_cache: 是否使用内存缓存。
    :param kwargs: 传递给模型构造函数或encode方法的额外参数。
                   - for Qwen: `model_kwargs`, `tokenizer_kwargs`, `prompt_name="query"`
                   - for BGE-M3: `use_fp16=True`, `max_length=8192`
                   - for LiteLLM: `embedding_backend="litellm"`, `api_key`, `api_base`
    :return: 文本的embedding向量 (numpy array)。
    """
    raw_model_name = model_name
    model_name = _normalize_model_name(model_name)
    kwargs = dict(kwargs)
    embedding_backend = kwargs.pop("embedding_backend", None)
    api_key = kwargs.pop("api_key", None)
    api_base = kwargs.pop("api_base", None)
    backend = _normalize_embedding_backend(embedding_backend, model_name)
    if embedding_backend is None and raw_model_name:
        if "qwen3-embedding-8b" in raw_model_name.lower():
            backend = "litellm"
    if backend == "litellm":
        litellm_model_name = raw_model_name or model_name
        if litellm_model_name and litellm_model_name.lower().startswith("qwen/"):
            litellm_model_name = litellm_model_name.split("/", 1)[1]
        if litellm_model_name and "qwen3-embedding-8b" in litellm_model_name.lower():
            if "/" not in litellm_model_name and not api_base:
                litellm_model_name = "qwen/qwen3-embedding-8b"
        model_name = litellm_model_name

    if backend == "bm25":
        raise RuntimeError("BM25 does not produce embeddings. Use BM25 in retrieval paths instead.")

    cache_kwargs = dict(kwargs)
    if api_base:
        cache_kwargs["api_base"] = api_base
    model_config_key = json.dumps(
        {"model_name": model_name, "backend": backend, **cache_kwargs},
        sort_keys=True,
    )
    
    if use_cache:
        cache_key = f"{model_config_key}::{hash(text)}"
        if cache_key in _embedding_cache:
            return _embedding_cache[cache_key]
    
    # --- Model Loading ---
    model_init_key = json.dumps(
        {
            "model_name": model_name,
            "backend": backend,
            **{k: v for k, v in kwargs.items() if k not in ["batch_size", "max_length"]},
        },
        sort_keys=True,
    )
    if backend in {"sentence-transformers", "contriever"} and model_init_key not in _model_cache:
        print(f"Loading model: {model_name}...")
        if backend == "contriever":
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "Please install torch to use Contriever embeddings."
                ) from exc
            tokenizer_kwargs = kwargs.get("tokenizer_kwargs") or {}
            model_kwargs = kwargs.get("model_kwargs") or {}
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            model = AutoModel.from_pretrained(model_name, **model_kwargs)
            model.eval()
            device = kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            _model_cache[model_init_key] = (tokenizer, model, device)
        elif 'bge-m3' in model_name.lower():
            try:
                from FlagEmbedding import BGEM3FlagModel
                init_kwargs = _get_valid_kwargs(BGEM3FlagModel.__init__, kwargs)
                print(f"-> Using BGEM3FlagModel with init kwargs: {init_kwargs}")
                _model_cache[model_init_key] = BGEM3FlagModel(model_name, **init_kwargs)
            except ImportError as exc:
                raise ImportError(
                    "Please install FlagEmbedding: 'pip install -U FlagEmbedding' to use bge-m3 model."
                ) from exc
        else:
            init_kwargs = dict(kwargs)
            if _should_trust_remote_code(model_name) and _supports_sentence_transformer_trust_remote_code():
                init_kwargs.setdefault("trust_remote_code", True)
            init_kwargs = _get_valid_kwargs(SentenceTransformer.__init__, init_kwargs)
            try:
                print(f"-> Using SentenceTransformer with init kwargs: {init_kwargs}")
                _model_cache[model_init_key] = SentenceTransformer(model_name, **init_kwargs)
            except ImportError as exc:
                raise ImportError(
                    "Please install sentence-transformers: 'pip install -U sentence-transformers' to use this model."
                ) from exc
            except Exception as exc:
                if _should_trust_remote_code(model_name) and _is_unknown_model_type_error(exc):
                    raise RuntimeError(
                        f"SentenceTransformer failed to load '{model_name}'. "
                        "Upgrade 'transformers' and 'sentence-transformers' to a version that supports Qwen3."
                    ) from exc
                raise
            
    # --- Encoding ---
    if backend == "litellm":
        try:
            from litellm import embedding as litellm_embedding
        except ImportError as exc:
            raise ImportError(
                "Please install litellm to use OpenAI-compatible embedding models."
            ) from exc
        request_args = {"model": model_name, "input": [text], "encoding_format": "float"}
        if api_base:
            request_args["api_base"] = api_base
        if api_key:
            request_args["api_key"] = api_key

        safe_request_args = dict(request_args)
        if safe_request_args.get("api_key"):
            safe_request_args["api_key"] = "<redacted>"
        print("request_args: ", safe_request_args)
        response = litellm_embedding(**request_args)
        raw_usage = getattr(response, "usage", None)
        if raw_usage is None and isinstance(response, dict):
            raw_usage = response.get("usage")
        _record_usage("embedding", model_name, raw_usage)
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data")
        embedding = data[0]["embedding"] if data else []
    elif backend == "contriever":
        tokenizer, model, device = _model_cache[model_init_key]
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Please install torch to use Contriever embeddings."
            ) from exc
        encode_kwargs = kwargs.get("encode_kwargs") or {}
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt", **encode_kwargs)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            model_output = model(**inputs)
        token_embeddings = model_output[0]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
        sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
        embedding = sum_embeddings / sum_mask
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1).cpu().numpy()[0]
    else:
        model = _model_cache[model_init_key]
        if 'bge-m3' in model_name.lower():
            encode_kwargs = _get_valid_kwargs(model.encode, kwargs)
            print(f"-> Encoding with BGEM3FlagModel using kwargs: {encode_kwargs}")
            result = model.encode([text], **encode_kwargs)
            embedding = result['dense_vecs'][0]
        else:
            encode_kwargs = _get_valid_kwargs(model.encode, kwargs)
            print(f"-> Encoding with SentenceTransformer using kwargs: {encode_kwargs}")
            embedding = model.encode([text], **encode_kwargs)[0]

    if use_cache:
        cache_key = f"{model_config_key}::{hash(text)}"
        _embedding_cache[cache_key] = embedding
        if len(_embedding_cache) > 10000:
            keys_to_remove = list(_embedding_cache.keys())[:1000]
            for key in keys_to_remove:
                try:
                    del _embedding_cache[key]
                except KeyError:
                    pass
            print("Cleaned embedding cache to prevent memory overflow")
    
    return embedding


def clear_embedding_cache():
    """清空embedding缓存"""
    global _embedding_cache
    _embedding_cache.clear()
    print("Embedding cache cleared")

def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ---- Time Decay Function ----
def compute_time_decay(event_timestamp_str, current_timestamp_str, tau_hours=24):
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t_event = datetime.strptime(event_timestamp_str, fmt)
        t_current = datetime.strptime(current_timestamp_str, fmt)
        delta_hours = (t_current - t_event).total_seconds() / 3600.0
        return np.exp(-delta_hours / tau_hours)
    except ValueError: # Handle cases where timestamp might be invalid
        return 0.1 # Default low recency


# ---- LLM-based Utility Functions ----

def gpt_summarize_dialogs(dialogs, client: OpenAIClient, model="gpt-4o-mini"):
    dialog_text = "\n".join([f"User: {d.get('user_input','')} Assistant: {d.get('agent_response','')}" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.SUMMARIZE_DIALOGS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.SUMMARIZE_DIALOGS_USER_PROMPT.format(dialog_text=dialog_text)}
    ]
    print("Calling LLM to generate topic summary...")
    return client.chat_completion(model=model, messages=messages)

def gpt_generate_multi_summary(text, client: OpenAIClient, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": prompts.MULTI_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.MULTI_SUMMARY_USER_PROMPT.format(text=text)}
    ]
    print("Calling LLM to generate multi-topic summary...")
    response_text = client.chat_completion(model=model, messages=messages)
    try:
        summaries = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse multi-summary JSON: {response_text}")
        summaries = [] # Return empty list or a default structure
    return {"input": text, "summaries": summaries}


def gpt_user_profile_analysis(dialogs, client: OpenAIClient, model="gpt-4o-mini", existing_user_profile="None"):
    """
    Analyze and update user personality profile from dialogs
    结合现有画像和新对话，直接输出更新后的完整画像
    """
    conversation = "\n".join([f"User: {d.get('user_input','')} (Timestamp: {d.get('timestamp', '')})\nAssistant: {d.get('agent_response','')} (Timestamp: {d.get('timestamp', '')})" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.PERSONALITY_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.PERSONALITY_ANALYSIS_USER_PROMPT.format(
            conversation=conversation,
            existing_user_profile=existing_user_profile
        )}
    ]
    print("Calling LLM for user profile analysis and update...")
    result_text = client.chat_completion(model=model, messages=messages)
    return result_text.strip() if result_text else "None"


def gpt_knowledge_extraction(dialogs, client: OpenAIClient, model="gpt-4o-mini"):
    """Extract user private data and assistant knowledge from dialogs"""
    conversation = "\n".join([f"User: {d.get('user_input','')} (Timestamp: {d.get('timestamp', '')})\nAssistant: {d.get('agent_response','')} (Timestamp: {d.get('timestamp', '')})" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.KNOWLEDGE_EXTRACTION_USER_PROMPT.format(
            conversation=conversation
        )}
    ]
    print("Calling LLM for knowledge extraction...")
    result_text = client.chat_completion(model=model, messages=messages)
    
    private_data = "None"
    assistant_knowledge = "None"

    try:
        if "【User Private Data】" in result_text:
            private_data_start = result_text.find("【User Private Data】") + len("【User Private Data】")
            if "【Assistant Knowledge】" in result_text:
                private_data_end = result_text.find("【Assistant Knowledge】")
                private_data = result_text[private_data_start:private_data_end].strip()
                
                assistant_knowledge_start = result_text.find("【Assistant Knowledge】") + len("【Assistant Knowledge】")
                assistant_knowledge = result_text[assistant_knowledge_start:].strip()
            else:
                private_data = result_text[private_data_start:].strip()
        elif "【Assistant Knowledge】" in result_text:
             assistant_knowledge_start = result_text.find("【Assistant Knowledge】") + len("【Assistant Knowledge】")
             assistant_knowledge = result_text[assistant_knowledge_start:].strip()

    except Exception as e:
        print(f"Error parsing knowledge extraction: {e}. Raw result: {result_text}")

    return {
        "private": private_data if private_data else "None", 
        "assistant_knowledge": assistant_knowledge if assistant_knowledge else "None"
    }


# Keep the old function for backward compatibility, but mark as deprecated
def gpt_personality_analysis(dialogs, client: OpenAIClient, model="gpt-4o-mini", known_user_traits="None"):
    """
    DEPRECATED: Use gpt_user_profile_analysis and gpt_knowledge_extraction instead.
    This function is kept for backward compatibility only.
    """
    # Call the new functions
    profile = gpt_user_profile_analysis(dialogs, client, model, known_user_traits)
    knowledge_data = gpt_knowledge_extraction(dialogs, client, model)
    
    return {
        "profile": profile,
        "private": knowledge_data["private"],
        "assistant_knowledge": knowledge_data["assistant_knowledge"]
    }


def gpt_update_profile(old_profile, new_analysis, client: OpenAIClient, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": prompts.UPDATE_PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.UPDATE_PROFILE_USER_PROMPT.format(old_profile=old_profile, new_analysis=new_analysis)}
    ]
    print("Calling LLM to update user profile...")
    return client.chat_completion(model=model, messages=messages)

def gpt_extract_theme(answer_text, client: OpenAIClient, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": prompts.EXTRACT_THEME_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.EXTRACT_THEME_USER_PROMPT.format(answer_text=answer_text)}
    ]
    print("Calling LLM to extract theme...")
    return client.chat_completion(model=model, messages=messages)



# ---- Functions from dynamic_update.py (to be used by Updater class) ----
def check_conversation_continuity(previous_page, current_page, client: OpenAIClient, model="gpt-4o-mini"):
    prev_user = previous_page.get("user_input", "") if previous_page else ""
    prev_agent = previous_page.get("agent_response", "") if previous_page else ""
    
    user_prompt = prompts.CONTINUITY_CHECK_USER_PROMPT.format(
        prev_user=prev_user,
        prev_agent=prev_agent,
        curr_user=current_page.get("user_input", ""),
        curr_agent=current_page.get("agent_response", "")
    )
    messages = [
        {"role": "system", "content": prompts.CONTINUITY_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat_completion(model=model, messages=messages, temperature=0.0, max_tokens=10)
    return response.strip().lower() == "true"

def generate_page_meta_info(last_page_meta, current_page, client: OpenAIClient, model="gpt-4o-mini"):
    current_conversation = f"User: {current_page.get('user_input', '')}\nAssistant: {current_page.get('agent_response', '')}"
    user_prompt = prompts.META_INFO_USER_PROMPT.format(
        last_meta=last_page_meta if last_page_meta else "None",
        new_dialogue=current_conversation
    )
    messages = [
        {"role": "system", "content": prompts.META_INFO_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    return client.chat_completion(model=model, messages=messages, temperature=0.3, max_tokens=100).strip() 
