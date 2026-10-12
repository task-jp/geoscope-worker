"""worker._AsyncProgress: 進捗の送信がスキャンのスレッドを止めないこと。"""
import os
import tempfile
import threading
import time

os.environ.setdefault("MODELS_DIR", tempfile.mkdtemp())
os.environ.setdefault("DATASETS_DIR", tempfile.mkdtemp())

import worker  # noqa: E402


def test_call_does_not_wait_for_send():
    release = threading.Event()
    sent = []

    def slow_send(p, msg):
        release.wait(5)
        sent.append((p, msg))

    prog = worker._AsyncProgress(slow_send)
    t0 = time.monotonic()
    for i in range(100):
        prog(i / 100, f"m{i}")
    assert time.monotonic() - t0 < 0.5      # 送信 (5 秒止まる) を待たない
    release.set()
    prog.close()
    # 1 件目は送信中、残りは最新の値だけが送られる
    assert sent[-1] == (0.99, "m99")
    assert len(sent) <= 3


def test_close_sends_latest_and_survives_errors():
    sent = []

    def send(p, msg):
        if msg == "bad":
            raise RuntimeError("network down")
        if msg == "cancel":
            raise worker.JobCancelled("x")
        sent.append(msg)

    prog = worker._AsyncProgress(send)
    prog(0.1, "bad")
    time.sleep(0.1)
    prog(0.2, "cancel")
    time.sleep(0.1)
    prog(1.0, "last")
    prog.close()
    assert sent == ["last"]


def test_session_is_per_thread():
    a = worker._session()
    assert worker._session() is a
    other = []
    t = threading.Thread(target=lambda: other.append(worker._session()))
    t.start(); t.join()
    assert other[0] is not a
