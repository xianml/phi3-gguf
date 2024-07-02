import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated
from llama_cpp import Llama
from typing import AsyncGenerator

MAX_TOKENS = 1024
SYS_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.If you don't know the answer to a question, please don't share false information
"""

@bentoml.service()
class Phi3:
    
    def __init__(self) -> None:
        self.llm = Llama.from_pretrained(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="*q4.gguf",
            verbose=False
        )
    
    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> str:
        response = self.llm.create_chat_completion(
            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]
    
    @bentoml.api
    async def generate_stream(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        response = self.llm.create_chat_completion(
            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            try:
                yield chunk["choices"][0]["delta"]["content"]
            except KeyError:
                yield ""

if __name__ == "__main__":
    phi3 = Phi3()
    response = phi3.llm.create_chat_completion(
            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": "Explain superconductors like I'm five years old"}
            ],
            max_tokens=256,
            stream=True,
        )
    for chunk in response:
        try:
            print(chunk["choices"][0]["delta"]["content"], end='', flush=True)
        except KeyError:
            pass