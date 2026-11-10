from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ..config import Settings


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...

    def stream(self, system: str, user: str,
               json_mode: bool = False) -> Iterator[str]: ...


def get_llm(settings: Settings) -> LLMClient:
    """Factory on settings.llm_provider. mock is the default offline path."""
    provider = (settings.llm_provider or "mock").lower()
    if provider == "bedrock":
        from .bedrock import BedrockLLM
        return BedrockLLM(settings.aws_region, settings.llm_model,
                          settings.aws_access_key_id, settings.aws_secret_access_key)
    if provider == "openai":
        from .openai import OpenAILLM
        return OpenAILLM(settings.openai_api_key, settings.openai_model,
                         base_url=settings.openai_base_url,
                         timeout=getattr(settings, "llm_timeout", 60.0))
    from .mock import MockLLM
    return MockLLM()
