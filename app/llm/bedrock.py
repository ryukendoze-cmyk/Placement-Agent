import logging

import boto3

logger = logging.getLogger(__name__)


def _chunk_words(text: str, size: int = 6):
    """Split ``text`` into whitespace-delimited chunks of roughly ``size`` words."""
    words = text.split()
    for i in range(0, len(words), size):
        piece = " ".join(words[i:i + size])
        if i + size < len(words):
            piece += " "
        yield piece


class BedrockLLM:
    def __init__(self, region: str, model: str, access_key: str | None = None, secret_key: str | None = None):
        kwargs = {"region_name": region}
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
        self._c = boto3.client("bedrock-runtime", **kwargs)
        self._m = model

    def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        req = {
            "modelId": self._m,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"maxTokens": 2000, "temperature": float(temperature)},
        }
        if json_mode:
            req["additionalModelRequestFields"] = {"output_format": "json"}
        r = self._c.converse(**req)
        return r["output"]["message"]["content"][0]["text"]

    def stream(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: float = 0.0,
        **kwargs,
    ):
        """Stream via ``converse_stream``; falls back to chunked ``complete()``."""
        req = {
            "modelId": self._m,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"maxTokens": 2000, "temperature": float(temperature)},
        }
        if json_mode:
            req["additionalModelRequestFields"] = {"output_format": "json"}
        try:
            resp = self._c.converse_stream(**req)
            for event in resp.get("stream", []):
                block = event.get("contentBlockDelta", {}).get("delta", {})
                text = block.get("text")
                if text:
                    yield text
        except Exception as e:
            logger.warning("Bedrock streaming failed (%s); falling back to chunked complete().", e)
            yield from _chunk_words(
                self.complete(system, user, json_mode=json_mode, temperature=temperature)
            )
