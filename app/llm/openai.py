import logging
import time

logger = logging.getLogger(__name__)

def _client_factory(api_key: str, base_url: str | None = None, timeout: float = 300.0):
    import openai  # lazily - only when provider=openai and package installed
    kwargs = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def _chunk_words(text: str, size: int = 6):
    """Split ``text`` into whitespace-delimited chunks of roughly ``size`` words."""
    words = text.split()
    for i in range(0, len(words), size):
        piece = " ".join(words[i:i + size])
        if i + size < len(words):
            piece += " "
        yield piece


class OpenAILLM:
    def __init__(self, api_key: str | None, model: str = "gpt-4o-mini", base_url: str | None = None, timeout: float = 300.0):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self._api_key = api_key
        self._client = None
        self._m = model
        self._base_url = base_url
        self._timeout = timeout

    def _get_client(self):
        if self._client is None:
            kwargs = {}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            try:
                self._client = _client_factory(self._api_key, timeout=self._timeout, **kwargs)
            except TypeError:
                try:
                    self._client = _client_factory(self._api_key, **kwargs)
                except TypeError:
                    self._client = _client_factory(self._api_key)
        return self._client

    def complete(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.0) -> str:
        kwargs = {
            "model": self._m,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        max_retries = 3
        backoff_sec = 2.0
        for attempt in range(max_retries):
            try:
                resp = self._get_client().chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                err_msg = str(e).lower()
                if json_mode and "response_format" in err_msg:
                    kwargs.pop("response_format", None)
                    continue
                is_rate_limit = any(term in err_msg for term in ["429", "rate limit", "queue is full", "too many requests"])
                if is_rate_limit and attempt < max_retries - 1:
                    sleep_time = backoff_sec * (attempt + 1)
                    logger.warning("LLM proxy rate limited (attempt %d/%d). Backing off %.1fs: %s", attempt + 1, max_retries, sleep_time, e)
                    time.sleep(sleep_time)
                    continue
                raise
        return ""

    def stream(self, system: str, user: str, json_mode: bool = False):
        """Stream the completion token-by-token via the OpenAI streaming API.

        Falls back to chunking the buffered ``complete()`` response when the
        streaming call fails (e.g. proxy without ``stream=True`` support), so
        SSE consumers always receive a full answer.
        """
        kwargs = {
            "model": self._m,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "stream": True,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._get_client().chat.completions.create(**kwargs)
            for event in resp:
                try:
                    delta = event.choices[0].delta.content or ""
                except (AttributeError, IndexError, TypeError):
                    continue
                if delta:
                    yield delta
        except Exception as e:
            logger.warning("OpenAI streaming failed (%s); falling back to chunked complete().", e)
            full = self.complete(system, user, json_mode=json_mode)
            yield from _chunk_words(full)
