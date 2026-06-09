import signal

import app.llm_server as mod
from app.llm_server import LlamaServer


def _srv(tmp_path, **kw):
    return LlamaServer(
        enabled=kw.pop("enabled", True),
        model_path=kw.pop("model", None),
        port=kw.pop("port", 8080),
        ctx=kw.pop("ctx", 8192),
        health_timeout=kw.pop("health_timeout", 1),
    )


def test_build_command_has_coresidence_flags(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    cmd = _srv(tmp_path, model=str(gguf), port=9000, ctx=4096).build_command()
    assert cmd[0].endswith("llama-server")
    assert "-m" in cmd and str(gguf) in cmd
    assert "--cpu-moe" in cmd
    assert cmd[cmd.index("--port") + 1] == "9000"
    assert cmd[cmd.index("-c") + 1] == "4096"
    assert cmd[cmd.index("-ctk") + 1] == "q8_0"
    assert cmd[cmd.index("-ctv") + 1] == "q8_0"


def _no_spawn(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("spawned")
    monkeypatch.setattr(mod.subprocess, "Popen", boom)


def test_disabled_never_spawns(tmp_path, monkeypatch):
    _no_spawn(monkeypatch)
    s = _srv(tmp_path, enabled=False)
    s.start()
    assert s.available is False


def test_missing_model_never_spawns(tmp_path, monkeypatch):
    _no_spawn(monkeypatch)
    s = _srv(tmp_path, model=str(tmp_path / "nope.gguf"))
    s.start()
    assert s.available is False


def test_stale_port_never_spawns(tmp_path, monkeypatch):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(mod, "_port_in_use", lambda host, port: True)
    _no_spawn(monkeypatch)
    s = _srv(tmp_path, model=str(gguf))
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


def test_stop_signals_process_group(tmp_path, monkeypatch):
    s = _srv(tmp_path)
    s._proc = _FakeProc()
    s.available = True
    killed = []
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    s.stop()
    assert (4321, signal.SIGTERM) in killed
    assert s.available is False
    assert s._proc is None
