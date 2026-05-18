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

def clean_reasoning_model_output(text):
    """
    清理推理模型输出中的<think>标签
    """
    if not text:
        return text
    
    import re
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

# ---- OpenAI Client ----
class OpenAIClient:
    def __init__(self, api_key, base_url=None, max_workers=5):
        self.api_key = api_key
        self.base_url = base_url if base_url else "https://api.openai.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        print(f"Calling OpenAI API. Model: {model}")
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw_content = response.choices[0].message.content.strip()
            cleaned_content = clean_reasoning_model_output(raw_content)
            return cleaned_content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "Error: Could not get response from LLM."

    def chat_completion_async(self, model, messages, temperature=0.7, max_tokens=2000):
        return self.executor.submit(self.chat_completion, model, messages, temperature, max_tokens)

    def batch_chat_completion(self, requests):
        futures = [self.chat_completion_async(**req) for req in requests]
        results = [future.result() for future in as_completed(futures)]
        return results

    def shutdown(self):
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
_embedding_cache = {}

def _get_valid_kwargs(func, kwargs):
    try:
        sig = inspect.signature(func)
        param_keys = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in param_keys}
    except (ValueError, TypeError):
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
    model_name = _normalize_model_name(model_name)
    kwargs = dict(kwargs)
    embedding_backend = kwargs.pop("embedding_backend", None)
    api_key = kwargs.pop("api_key", None)
    api_base = kwargs.pop("api_base", None)
    backend = _normalize_embedding_backend(embedding_backend, model_name)

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
            
    if backend == "litellm":
        try:
            from litellm import embedding as litellm_embedding
        except ImportError as exc:
            raise ImportError(
                "Please install litellm to use OpenAI-compatible embedding models."
            ) from exc
        request_args = {"model": model_name, "input": [text]}
        if api_base:
            request_args["api_base"] = api_base
        if api_key:
            request_args["api_key"] = api_key
        response = litellm_embedding(**request_args)
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
            result = model.encode([text], **encode_kwargs)
            embedding = result['dense_vecs'][0]
        else:
            encode_kwargs = _get_valid_kwargs(model.encode, kwargs)
            embedding = model.encode([text], **encode_kwargs)[0]

    if use_cache:
        cache_key = f"{model_config_key}::{hash(text)}"
        _embedding_cache[cache_key] = embedding
    
    return embedding

def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

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
        summaries = []
    return {"input": text, "summaries": summaries}

def extract_keywords_from_multi_summary(text, client: OpenAIClient, model="gpt-4o-mini"):
    """
    Extract keywords using multi-summary analysis instead of separate keyword extraction.
    This is more efficient as the multi-summary already includes keywords for each theme.
    """
    multi_summary_result = gpt_generate_multi_summary(text, client, model)
    all_keywords = []
    
    if multi_summary_result and multi_summary_result.get("summaries"):
        for summary_item in multi_summary_result["summaries"]:
            keywords = summary_item.get("keywords", [])
            all_keywords.extend(keywords)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in all_keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords

def gpt_user_profile_analysis(conversation_str: str, client: OpenAIClient, model="gpt-4o-mini", existing_user_profile="None"):
    """
    Analyze and update user personality profile from a conversation string.
    """
    messages = [
        {"role": "system", "content": prompts.PERSONALITY_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.PERSONALITY_ANALYSIS_USER_PROMPT.format(
            conversation=conversation_str,
            existing_user_profile=existing_user_profile
        )}
    ]
    print("Calling LLM for user profile analysis and update...")
    result_text = client.chat_completion(model=model, messages=messages)
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        print(f"Warning: User profile analysis did not return valid JSON. Content: {result_text}")
        return {"raw_text_profile": result_text}

def gpt_knowledge_extraction(conversation_str: str, client: OpenAIClient, model="gpt-4o-mini"):
    """Extract user private data and assistant knowledge from a conversation string"""
    messages = [
        {"role": "system", "content": prompts.KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.KNOWLEDGE_EXTRACTION_USER_PROMPT.format(
            conversation=conversation_str
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

def gpt_update_profile(old_profile, new_analysis, client: OpenAIClient, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": prompts.UPDATE_PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.UPDATE_PROFILE_USER_PROMPT.format(old_profile=old_profile, new_analysis=new_analysis)}
    ]
    return client.chat_completion(model=model, messages=messages)

def check_conversation_continuity(previous_page, current_page, client: OpenAIClient, model="gpt-4o-mini"):
    if not previous_page or not current_page:
        return False
    
    prompt = prompts.CONTINUITY_CHECK_USER_PROMPT.format(
        prev_user=previous_page.get('user_input', ''),
        prev_agent=previous_page.get('agent_response', ''),
        curr_user=current_page.get('user_input', ''),
        curr_agent=current_page.get('agent_response', '')
    )
    messages = [{"role": "system", "content": prompts.CONTINUITY_CHECK_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    
    response = client.chat_completion(model, messages, temperature=0.0)
    return response.lower() == 'true'

def generate_page_meta_info(last_page_meta, current_page, client: OpenAIClient, model="gpt-4o-mini"):
    new_dialogue = f"User: {current_page.get('user_input', '')}\nAssistant: {current_page.get('agent_response', '')}"
    
    prompt = prompts.META_INFO_USER_PROMPT.format(
        last_meta=last_page_meta or "This is the beginning of the conversation.",
        new_dialogue=new_dialogue
    )
    messages = [{"role": "system", "content": prompts.META_INFO_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    
    return client.chat_completion(model, messages, temperature=0.3) 
