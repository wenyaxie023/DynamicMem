import os
import json
from typing import Dict

from openai import OpenAI
from google import genai


class LLMClient:
    def __init__(self, provider="openai", model_name="gpt-5-mini"):
        self.provider = provider
        self.model_name = model_name

        if provider == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

        elif provider == "gemini":
            self.client = genai.Client(
                api_key=os.getenv("GOOGLE_API_KEY")
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def ask(self, prompt: str) -> Dict:
        if self.provider == "openai":
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
            )

            return json.loads(response.output_text)

        elif self.provider == "gemini":
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            return json.loads(response.text)



if __name__ == "__main__":
    llm_openai = LLMClient(
        provider="openai",
        model_name="gpt-5-mini"
    )

    print(
        llm_openai.ask(
            "What is the capital of France? Answer in JSON with key 'capital'."
        )
    )

    llm_gemini = LLMClient(
        provider="gemini",
        model_name="gemini-2.5-flash-lite"
    )

    print(
        llm_gemini.ask(
            "What is the capital of Japan? Answer in JSON with key 'capital'."
        )
    )
