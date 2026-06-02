from __future__ import annotations

import sys
from io import BytesIO, TextIOWrapper
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BROWSER_HARNESS_SRC = REPO_ROOT / "skills" / "browser-harness" / "src"
if str(BROWSER_HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(BROWSER_HARNESS_SRC))

from browser_harness import _ipc  # noqa: E402
from browser_harness import helpers  # noqa: E402
from browser_harness import run  # noqa: E402


def test_sanitize_lone_surrogates_in_nested_response() -> None:
    result = _ipc._sanitize_surrogates(
        {
            "plain": "hello",
            "low": "abc\udc80def",
            "high": "abc\ud800def",
            "items": [1, "x\udcff", {"k\udca2": "v\udcad"}],
        }
    )

    assert result == {
        "plain": "hello",
        "low": "abc\ufffddef",
        "high": "abc\ufffddef",
        "items": [1, "x\ufffd", {"k\ufffd": "v\ufffd"}],
    }


def test_run_sanitizes_script_before_exec_compile() -> None:
    code = run._replace_surrogates("print('abc\udcaedef')")

    compile(code, "<browser-harness-stdin>", "exec")
    assert code == "print('abc\ufffddef')"


def test_run_reconfigures_stdin_to_utf8() -> None:
    stream = TextIOWrapper(BytesIO("李秉寰".encode("utf-8")), encoding="cp936", errors="replace")

    run._configure_text_stream(stream)

    assert stream.read() == "李秉寰"


def test_ipc_request_sanitizes_json_decoded_surrogates() -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.sent = b""
            self.chunks = [b'{"result":{"value":"abc\\udc80def"}}\n']

        def sendall(self, data: bytes) -> None:
            self.sent += data

        def recv(self, size: int) -> bytes:
            return self.chunks.pop(0)

    sock = FakeSocket()
    result = _ipc.request(sock, None, {"method": "Runtime.evaluate"})

    assert sock.sent == b'{"method": "Runtime.evaluate"}\n'
    assert result == {"result": {"value": "abc\ufffddef"}}


def test_page_info_sanitizes_surrogates_from_second_json_decode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        helpers,
        "_runtime_evaluate",
        lambda expression: '{"url":"https://example.test","title":"abc\\udc80def"}',
    )
    monkeypatch.setattr(helpers, "_send", lambda req: {"dialog": None})

    assert helpers.page_info() == {
        "url": "https://example.test",
        "title": "abc\ufffddef",
    }
