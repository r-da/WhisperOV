"""Tests for the faster-whisper (CUDA) backend and the backend selection (faster_whisper and openvino are faked)."""

import sys
import types

import numpy as np
import pytest

from whisper_ov import cli, whisper_ct2


class _FakeSegment:
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end


@pytest.fixture
def ct2_calls(monkeypatch):
    calls: list[dict] = []

    class FakeWhisperModel:
        def __init__(self, model_path, **kwargs):
            calls.append({"init": kwargs, "model_path": model_path})

        def transcribe(self, audio, **kwargs):
            calls.append({"transcribe": kwargs})
            segments = [_FakeSegment(" Ciao,", 0.0, 1.5), _FakeSegment(" come stai?", 1.5, 3.0)]
            return iter(segments), None

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(whisper_ct2, "_resolve_model_path", lambda model_id, cache_dir: "/fake/ct2")
    return calls


def _run(**kwargs):
    return whisper_ct2.transcribe(np.zeros(16000, dtype=np.float32), **kwargs)


def _init_kwargs(calls):
    return next(c["init"] for c in calls if "init" in c)


def _gen_kwargs(calls):
    return next(c["transcribe"] for c in calls if "transcribe" in c)


def test_segments_and_text(ct2_calls):
    result = _run()
    assert result["text"] == "Ciao, come stai?"
    assert result["segments"] == [
        {"text": "Ciao,", "start": 0.0, "end": 1.5},
        {"text": "come stai?", "start": 1.5, "end": 3.0},
    ]


@pytest.mark.parametrize("language", ["it", "<|it|>"])
def test_language_passed_as_plain_code(ct2_calls, language):
    _run(language=language)
    assert _gen_kwargs(ct2_calls)["language"] == "it"


def test_prompts_unset_by_default(ct2_calls):
    _run()
    assert "initial_prompt" not in _gen_kwargs(ct2_calls)
    assert "hotwords" not in _gen_kwargs(ct2_calls)


@pytest.mark.parametrize("blank", ["", " ", "\n\t "])
def test_blank_prompts_ignored(ct2_calls, blank):
    _run(initial_prompt=blank, hotwords=blank)
    assert "initial_prompt" not in _gen_kwargs(ct2_calls)
    assert "hotwords" not in _gen_kwargs(ct2_calls)


def test_prompts_are_stripped(ct2_calls):
    _run(initial_prompt=" Ciao. ", hotwords="  Buongiorno.\n")
    assert _gen_kwargs(ct2_calls)["initial_prompt"] == "Ciao."
    assert _gen_kwargs(ct2_calls)["hotwords"] == "Buongiorno."


@pytest.mark.parametrize(
    ("model_id", "compute_type", "expected"),
    [
        ("openai/whisper-large-v3-turbo", "auto", "auto"),
        ("openai/whisper-large-v3-turbo-int8", "auto", "int8_float16"),
        ("openai/whisper-large-v3-turbo-int8", "float32", "float32"),
    ],
)
def test_compute_type(ct2_calls, model_id, compute_type, expected):
    _run(model_id=model_id, compute_type=compute_type)
    assert _init_kwargs(ct2_calls)["compute_type"] == expected


def test_int4_not_supported():
    assert not whisper_ct2.supports_model("openai/whisper-large-v3-int4")
    with pytest.raises(ValueError, match="INT4"):
        whisper_ct2._get_ct2_repo_id("openai/whisper-large-v3-int4")


def test_repo_ids():
    assert whisper_ct2._get_ct2_repo_id("openai/whisper-tiny") == "Systran/faster-whisper-tiny"
    assert whisper_ct2._get_ct2_repo_id("openai/whisper-large-v2") == "Systran/faster-whisper-large-v2"
    # INT8 variants share the fp16 repo
    assert whisper_ct2._get_ct2_repo_id("openai/whisper-large-v3-int8") == "Systran/faster-whisper-large-v3"


@pytest.mark.parametrize(
    ("device", "expected"),
    [("CUDA", 0), ("cuda:1", 1), ("CPU", None), ("GPU.1", None), ("AUTO", None)],
)
def test_parse_device(device, expected):
    assert whisper_ct2.parse_device(device) == expected


def test_parse_device_invalid():
    with pytest.raises(ValueError):
        whisper_ct2.parse_device("CUDA:x")


@pytest.fixture
def backends(monkeypatch):
    """Record which backend _transcribe_with_backend picks; CUDA availability is controlled per test."""
    calls: list[tuple[str, dict]] = []

    def fake_ct2(audio, **kwargs):
        calls.append(("cuda", kwargs))
        if state["ct2_fails"]:
            raise RuntimeError("cuDNN not found")
        return {"text": "cuda", "segments": []}

    def fake_ov(audio, **kwargs):
        calls.append(("openvino", kwargs))
        return {"text": "openvino", "segments": []}

    state = {"ct2_fails": False, "cuda_devices": [{"index": 0, "compute_types": ["float16"]}]}
    monkeypatch.setattr(whisper_ct2, "transcribe", fake_ct2)
    monkeypatch.setattr(whisper_ct2, "cuda_devices", lambda: state["cuda_devices"])
    monkeypatch.setattr(cli, "_transcribe", fake_ov)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    return calls, state


def _select(device, model="openai/whisper-tiny"):
    audio = np.zeros(16000, dtype=np.float32)
    return cli._transcribe_with_backend(audio, device=device, model=model, compute_type="auto")


def test_auto_prefers_cuda(backends):
    calls, _ = backends
    assert _select("auto")["text"] == "cuda"
    assert [name for name, _ in calls] == ["cuda"]


def test_auto_without_cuda_uses_openvino(backends):
    calls, state = backends
    state["cuda_devices"] = []
    assert _select("AUTO")["text"] == "openvino"
    assert calls[0][1]["device"] == "AUTO"


def test_auto_int4_uses_openvino(backends):
    calls, _ = backends
    assert _select("AUTO", model="openai/whisper-large-v3-int4")["text"] == "openvino"
    assert [name for name, _ in calls] == ["openvino"]


def test_auto_falls_back_when_cuda_fails(backends):
    calls, state = backends
    state["ct2_fails"] = True
    assert _select("AUTO")["text"] == "openvino"
    assert [name for name, _ in calls] == ["cuda", "openvino"]


def test_auto_without_extra_uses_openvino(backends, monkeypatch):
    calls, _ = backends
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    assert _select("AUTO")["text"] == "openvino"
    assert [name for name, _ in calls] == ["openvino"]


def test_explicit_cuda_does_not_fall_back(backends):
    calls, state = backends
    state["ct2_fails"] = True
    state["cuda_devices"] = [{"index": 0, "compute_types": []}, {"index": 1, "compute_types": []}]
    with pytest.raises(RuntimeError, match="cuDNN"):
        _select("CUDA:1")
    assert calls == [("cuda", calls[0][1])]
    assert calls[0][1]["device_index"] == 1


@pytest.mark.parametrize(("device", "visible"), [("CUDA", 0), ("CUDA:1", 1)])
def test_explicit_cuda_device_not_found(backends, device, visible):
    calls, state = backends
    state["cuda_devices"] = state["cuda_devices"][:visible]
    with pytest.raises(RuntimeError, match=f"sees {visible} CUDA device"):
        _select(device)
    assert calls == []


def test_explicit_cuda_without_extra(backends, monkeypatch):
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="uv sync --extra cuda"):
        _select("CUDA")


def test_openvino_devices_unchanged(backends):
    calls, _ = backends
    assert _select("gpu")["text"] == "openvino"
    assert calls[0][1]["device"] == "GPU"
