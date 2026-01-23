import os
import json
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Union

from openai import OpenAI
from google import genai


class LLMClient:
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-5-mini",
        *,
        max_workers: int = 4,
        vllm_base_url: str = "http://localhost:8002/v1",
        retry_times: int = 1,
    ):
        self.provider = provider
        self.model_name = model_name
        if retry_times < 0:
            raise ValueError("retry_times must be >= 0")
        self.retry_times = retry_times
        self.vllm_base_url = vllm_base_url

        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=max_workers
        )

        if provider == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif provider == "gemini":
            self.client = genai.Client(
                api_key=os.getenv("GOOGLE_API_KEY")
            )

        elif provider == "vllm":
            self.client = OpenAI(
                base_url=self.vllm_base_url,
                api_key="EMPTY",
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def ask(self, prompt: str, *, response_type: str = "json") -> Dict | str:
        if self._executor is None:
            raise RuntimeError("LLMClient executor is closed.")

        future = self._executor.submit(self._ask_impl, prompt, response_type)
        return future.result()

    def ask_async(self, prompt: str, *, response_type: str = "json") -> Future:
        if self._executor is None:
            raise RuntimeError("LLMClient executor is closed.")
        return self._executor.submit(self._ask_impl, prompt, response_type)

    def ask_many_async(
        self,
        prompts: Iterable[str],
        *,
        response_type: str = "json",
    ) -> List[Future]:
        return [
            self.ask_async(prompt, response_type=response_type)
            for prompt in prompts
        ]

    def collect(self, futures: Iterable[Future]) -> List[Union[Dict, str, Exception]]:
        results: List[Union[Dict, str, Exception]] = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(exc)
        return results

    def ask_many(
        self,
        prompts: Iterable[str],
        *,
        response_type: str = "json",
    ) -> List[Union[Dict, str, Exception]]:
        futures = self.ask_many_async(prompts, response_type=response_type)
        return self.collect(futures)

    # =======================
    # Provider implementations
    # =======================

    def _ask_impl(self, prompt: str, response_type: str) -> Dict | str:
        attempts = self.retry_times + 1
        last_exc: Exception | None = None
        for _ in range(attempts):
            try:
                if self.provider == "openai":
                    response = self.client.responses.create(
                        model=self.model_name,
                        input=prompt,
                    )
                    return self._parse_response(response.output_text, response_type)

                if self.provider == "gemini":
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={"response_mime_type": "application/json"},
                    )
                    return self._parse_response(response.text, response_type)
                if self.provider == "vllm":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                    )
                    text = response.choices[0].message.content
                    return self._parse_response(text, response_type)
        
                raise ValueError(f"Unknown provider: {self.provider}")
            except Exception as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    def _parse_response(self, text: str, response_type: str) -> Dict | str:
        if response_type == "text":
            return text

        if response_type != "json":
            raise ValueError(f"Unknown response_type: {response_type}")

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            extracted = self._extract_json_block(text)
            if extracted is None:
                raise ValueError(f"Invalid JSON response: {text}") from exc
            try:
                return json.loads(extracted)
            except json.JSONDecodeError as nested_exc:
                raise ValueError(f"Invalid JSON response: {text}") from nested_exc

    def _extract_json_block(self, text: str) -> Optional[str]:
        marker = "```"
        start = text.find(marker)
        if start == -1:
            return None
        end = text.find(marker, start + len(marker))
        if end == -1:
            return None
        block = text[start + len(marker):end]
        if block.lstrip().lower().startswith("json"):
            block = block.lstrip()[4:]
        return block.strip() if block.strip() else None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


if __name__ == "__main__":
    llm_openai = LLMClient(
        provider="openai",
        model_name="gpt-5-mini"
    )

    print(
        llm_openai.ask(
            "What is the capital of France?"
            , response_type="text"
        )
    )

    llm_gemini = LLMClient(
        provider="gemini",
        model_name="gemini-2.5-flash-lite"
    )

    print(
        llm_gemini.ask(
            "What is the capital of Japan?"
            , response_type="text"
        )
    )

    llm_vllm = LLMClient(
        provider="vllm",
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    )

    print(
        llm_vllm.ask(
            "What is the capital of China?"
            , response_type="text"
        )
    )
