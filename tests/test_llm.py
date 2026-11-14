import json

from app.config import Settings
from app.llm.base import get_llm
from app.llm.bedrock import BedrockLLM
from app.llm.mock import MockLLM
from app.llm.openai import OpenAILLM


def _settings(**overrides):
    base = dict(llm_provider="mock", llm_model="m", aws_region="us-east-1",
                aws_access_key_id=None, aws_secret_access_key=None,
                openai_api_key=None, openai_model="gpt-4o-mini", openai_base_url=None)
    base.update(overrides)
    return Settings(**base)


def test_factory_returns_mock_by_default():
    llm = get_llm(_settings())
    assert isinstance(llm, MockLLM)


def test_factory_returns_bedrock_for_bedrock_provider(monkeypatch):
    import app.llm.bedrock as bedrock_mod
    class FakeClient:
        def converse(self, **kw):
            return {"output": {"message": {"content": [{"text": "hi"}]}}}
    monkeypatch.setattr(bedrock_mod.boto3, "client", lambda *a, **k: FakeClient())
    llm = get_llm(_settings(llm_provider="bedrock"))
    assert isinstance(llm, BedrockLLM)


def test_factory_returns_openai_for_openai_provider(monkeypatch):
    llm = get_llm(_settings(llm_provider="openai", openai_api_key="sk-test"))
    assert isinstance(llm, OpenAILLM)


def test_mock_llm_json_mode_returns_parseable_json():
    llm = MockLLM()
    out = llm.complete(system="TASK: plan_narrative", user='{"gaps": ["git"]}', json_mode=True)
    parsed = json.loads(out)
    assert "narrative" in parsed or "text" in parsed


def test_mock_llm_text_mode_returns_text():
    llm = MockLLM()
    out = llm.complete(system="greet", user="world", json_mode=False)
    assert isinstance(out, str) and out


def test_mock_evaluate_answer_scores_by_keywords():
    llm = MockLLM()
    rich = json.loads(llm.complete(
        system="TASK: evaluate_answer",
        user=json.dumps({"answer": "I use lists dicts and generators often",
                         "keywords": ["list", "dict", "generator"]}),
        json_mode=True))
    thin = json.loads(llm.complete(
        system="TASK: evaluate_answer",
        user=json.dumps({"answer": "x", "keywords": ["list", "dict", "generator"]}),
        json_mode=True))
    assert 0 <= rich["score"] <= 100
    assert "feedback" in rich
    assert rich["score"] > thin["score"]


def test_mock_interview_feedback_json():
    llm = MockLLM()
    out = llm.complete(
        system="TASK: interview_feedback",
        user=json.dumps({"score": 65, "per_skill": {"python": 80, "git": 40}}),
        json_mode=True)
    parsed = json.loads(out)
    assert "feedback" in parsed
    assert parsed["feedback"].strip()


def test_mock_task_marker_space_and_inline_forms_equivalent():
    llm = MockLLM()
    user = json.dumps({"gaps": ["git"], "total_hours": 8})
    spaced = llm.complete("TASK: plan_narrative", user, json_mode=True)
    inline = llm.complete("TASK:plan_narrative", user, json_mode=True)
    assert spaced == inline


def test_mock_fallback_when_no_task_marker():
    out = MockLLM().complete(system="you are helpful", user="hello", json_mode=True)
    assert json.loads(out) == {"echo": "hello"}


def test_factory_falls_back_to_mock_on_unknown_provider():
    llm = get_llm(_settings(llm_provider="totally-unknown"))
    assert isinstance(llm, MockLLM)


def test_bedrock_complete_builds_converse_request(monkeypatch):
    import app.llm.bedrock as bedrock_mod
    captured = {}

    class FakeClient:
        def converse(self, **kw):
            captured.update(kw)
            return {"output": {"message": {"content": [{"text": "resp"}]}}}

    monkeypatch.setattr(bedrock_mod.boto3, "client", lambda *a, **k: FakeClient())
    llm = BedrockLLM("us-east-1", "amazon.nova-lite-v1:0")
    out = llm.complete(system="sys", user="usr", json_mode=True)
    assert out == "resp"
    assert captured["modelId"] == "amazon.nova-lite-v1:0"
    assert captured["system"][0]["text"] == "sys"
    assert captured["messages"][0]["content"][0]["text"] == "usr"
    assert captured["inferenceConfig"]["maxTokens"] == 2000
    assert captured["additionalModelRequestFields"]["output_format"] == "json"

    captured.clear()
    llm.complete(system="s", user="u", json_mode=False)
    assert "additionalModelRequestFields" not in captured
    assert captured["inferenceConfig"] == {"maxTokens": 2000, "temperature": 0.0}


def test_openai_complete_builds_chat_request(monkeypatch):
    import app.llm.openai as openai_mod

    captured = {}

    class FakeMsg:
        content = "ai-reply"

    class FakeChoice:
        message = FakeMsg()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(openai_mod, "_client_factory", lambda key: FakeClient())
    llm = OpenAILLM("sk-test", "gpt-4o-mini")
    out = llm.complete(system="s", user="u", json_mode=True)
    assert out == "ai-reply"
    assert captured["model"] == "gpt-4o-mini"


def test_openai_with_custom_base_url(monkeypatch):
    import app.llm.openai as openai_mod
    captured = {}

    class FakeClient:
        pass

    def fake_factory(key, base_url=None):
        captured["key"] = key
        captured["base_url"] = base_url
        return FakeClient()

    monkeypatch.setattr(openai_mod, "_client_factory", fake_factory)
    llm = OpenAILLM("sk-test", "qwen-3.8-27b", base_url="https://bharatcode.ai/api/model/v1")
    _ = llm._get_client()
    assert captured["key"] == "sk-test"
    assert captured["base_url"] == "https://bharatcode.ai/api/model/v1"


def test_factory_passes_openai_base_url():
    settings = _settings(llm_provider="openai", openai_api_key="sk-test",
                         openai_model="qwen-3.8-27b", openai_base_url="https://bharatcode.ai/api/model/v1")
    llm = get_llm(settings)
    assert isinstance(llm, OpenAILLM)
    assert llm._base_url == "https://bharatcode.ai/api/model/v1"
    assert llm._m == "qwen-3.8-27b"
