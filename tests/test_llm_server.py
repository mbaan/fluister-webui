import signal

import app.llm_server as mod
from app.llm_server import LlamaServer


def _srv(**kw):
    return LlamaServer(
        enabled=kw.pop("enabled", True),
        repo=kw.pop("repo", None),
        file=kw.pop("file", None),
        token=kw.pop("token", None),
        model_path=kw.pop("model_path", None),
        port=kw.pop("port", 8080),
        ctx=kw.pop("ctx", 8192),
        health_timeout=kw.pop("health_timeout", 1),
    )


def _no_spawn(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("spawned")
    monkeypatch.setattr(mod.subprocess, "Popen", boom)


def test_build_command_has_coresidence_flags(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    s = _srv(model_path=str(gguf), port=9000, ctx=4096)
    cmd = s.build_command()
    assert cmd[0].endswith("llama-server")
    assert "-m" in cmd and str(gguf) in cmd
    assert "--cpu-moe" in cmd
    assert cmd[cmd.index("--port") + 1] == "9000"
    assert cmd[cmd.index("-c") + 1] == "4096"
    assert cmd[cmd.index("-ctk") + 1] == "q8_0"
    assert cmd[cmd.index("-ctv") + 1] == "q8_0"


def test_disabled_never_spawns(monkeypatch):
    _no_spawn(monkeypatch)
    s = _srv(enabled=False)
    s.start()
    assert s.available is False


def test_no_model_configured_never_spawns(monkeypatch):
    _no_spawn(monkeypatch)
    s = _srv()  # no repo/file, no override
    s.start()
    assert s.available is False


def test_missing_override_file_never_spawns(tmp_path, monkeypatch):
    _no_spawn(monkeypatch)
    s = _srv(model_path=str(tmp_path / "nope.gguf"))
    s.start()
    assert s.available is False


def test_hf_repo_resolves_into_model_path(tmp_path, monkeypatch):
    # hf_hub_download is imported lazily inside _resolve_model from huggingface_hub.
    calls = {}

    def fake_dl(repo, file, token=None):
        calls["args"] = (repo, file, token)
        return "/cache/hub/resolved.gguf"

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(mod, "_port_in_use", lambda host, port: True)  # bail before spawn
    _no_spawn(monkeypatch)

    s = _srv(repo="org/model-GGUF", file="m-Q4_K_M.gguf", token="tok")
    s.start()
    assert calls["args"] == ("org/model-GGUF", "m-Q4_K_M.gguf", "tok")
    assert s.model_path == "/cache/hub/resolved.gguf"  # resolved path stored for -m
    assert s.available is False  # stale-port guard stopped it before spawn


def test_local_override_wins_over_repo(tmp_path, monkeypatch):
    gguf = tmp_path / "local.gguf"
    gguf.write_bytes(b"x")

    def boom(*a, **k):
        raise AssertionError("hf_hub_download should not be called when override is set")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    monkeypatch.setattr(mod, "_port_in_use", lambda host, port: True)
    _no_spawn(monkeypatch)

    s = _srv(model_path=str(gguf), repo="org/model-GGUF", file="m.gguf")
    s.start()
    assert s.model_path == str(gguf)


def test_stale_port_never_spawns(tmp_path, monkeypatch):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(mod, "_port_in_use", lambda host, port: True)
    _no_spawn(monkeypatch)
    s = _srv(model_path=str(gguf))
    s.start()
    assert s.available is False


class _FakeProc:
    def __init__(self):
        self.pid = 4321
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self._alive = False
        return 0


def test_stop_signals_process_group(monkeypatch):
    s = _srv()
    s._proc = _FakeProc()
    s.available = True
    killed = []
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    s.stop()
    assert (4321, signal.SIGTERM) in killed
    assert s.available is False
    assert s._proc is None
