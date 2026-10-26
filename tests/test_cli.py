"""Tests for the cli.main entry point (os._exit is faked)."""

import pytest

from whisper_ov import cli


class _Exited(Exception):
    pass


@pytest.fixture
def exit_codes(monkeypatch):
    codes = []

    def fake_exit(code):
        codes.append(code)
        raise _Exited

    monkeypatch.setattr(cli.os, "_exit", fake_exit)
    return codes


@pytest.mark.parametrize(("argv", "code"), [(["models"], 0), (["no-such-command"], 2)])
def test_windows_exits_without_teardown(monkeypatch, exit_codes, argv, code):
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.sys, "argv", ["whisper-ov", *argv])
    with pytest.raises(_Exited):
        cli.main()
    assert exit_codes == [code]


def test_other_platforms_exit_normally(monkeypatch, exit_codes):
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.sys, "argv", ["whisper-ov", "models"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0
    assert exit_codes == []
