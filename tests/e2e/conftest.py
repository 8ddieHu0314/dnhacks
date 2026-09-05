"""Session-scoped fixtures for the Live tab end-to-end test.

Two things need to exist before the test can run:

1. The real app (app.py's `demo` object), launched as a subprocess on a free
   TCP port that is never 7860 (7860 is reserved for a human running
   `python app.py` at the same time). We poll GET / until it answers 200,
   since model load can be slow.
2. A headless Playwright chromium browser fed a fake webcam clip via
   Chrome's `--use-file-for-fake-video-capture` flag, with camera permission
   pre-granted for the app's origin so no permission prompt blocks the test.

Note on networkidle: Gradio keeps a persistent SSE/heartbeat connection open
to the server for its queue, so `page.goto(..., wait_until="networkidle")`
never resolves (verified against a throwaway Gradio app during harness
development: it timed out even with no webcam streaming in progress). Tests
use `wait_until="load"` plus an explicit wait for a known selector instead.
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FAKE_CAM_Y4M = os.path.join(FIXTURES_DIR, "fake_cam.y4m")
SOURCE_IMAGE = os.path.join(REPO_ROOT, "test_images", "bare_hands_wrench_1.jpg")
FFMPEG = "/Users/gabemeredith/anaconda3/bin/ffmpeg"


def _free_port() -> int:
    """Ask the OS for a free TCP port, never 7860 (the real dev server)."""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port != 7860:
            return port


def _wait_for_http_ok(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_err = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - keep polling on any error
            last_err = exc
        time.sleep(1.0)
    raise RuntimeError(
        f"App never returned 200 from {url} within {timeout_s:.0f} s "
        f"(last error: {last_err!r}). Model load can be slow; check the "
        f"app's stdout printed above for a traceback."
    )


@pytest.fixture(scope="session")
def fake_cam_y4m():
    """Build tests/e2e/fixtures/fake_cam.y4m once per session if missing.

    Source: test_images/bare_hands_wrench_1.jpg (bare hands holding a
    wrench), looped into a 6 s / 5 fps yuv420p clip so BARE_HANDS and
    TOOL_IN_BARE_HAND have something to fire on once the debounce window
    passes.
    """
    if os.path.exists(FAKE_CAM_Y4M):
        return FAKE_CAM_Y4M

    os.makedirs(FIXTURES_DIR, exist_ok=True)
    if not os.path.exists(SOURCE_IMAGE):
        pytest.skip(f"source image missing, cannot build fake cam clip: {SOURCE_IMAGE}")
    if not os.path.exists(FFMPEG):
        pytest.skip(f"ffmpeg missing, cannot build fake cam clip: {FFMPEG}")

    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-loop", "1", "-i", SOURCE_IMAGE,
        "-t", "6", "-r", "5",
        "-vf", "scale=640:-2,format=yuv420p",
        "-pix_fmt", "yuv420p",
        FAKE_CAM_Y4M,
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    assert os.path.exists(FAKE_CAM_Y4M), "ffmpeg reported success but output file is missing"
    return FAKE_CAM_Y4M


@pytest.fixture(scope="session")
def live_app_server():
    """Launch app.demo as a subprocess on a free port, wait for GET / to
    return 200 (up to 90 s, model load is slow), yield the base URL, then
    terminate the process."""
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    launch_code = (
        "import app, os; "
        "app.demo.launch(server_name='127.0.0.1', server_port={port}, "
        "share=False, allowed_paths=[app.OUTPUT_DIR], prevent_thread_lock=False)"
    ).format(port=port)

    proc = subprocess.Popen(
        [sys.executable, "-c", launch_code],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        try:
            _wait_for_http_ok(base_url + "/", timeout_s=90.0)
        except Exception:
            if proc.poll() is not None and proc.stdout:
                print("---- app server output (process exited early) ----")
                print(proc.stdout.read())
            raise
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


@pytest.fixture(scope="session")
def browser_context(live_app_server, fake_cam_y4m):
    """A headless chromium context fed the fake webcam clip, with camera
    permission pre-granted for the app's origin so no permission prompt
    blocks streaming."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-video-capture={fake_cam_y4m}",
            ],
        )
        context = browser.new_context()
        context.grant_permissions(["camera"], origin=live_app_server)
        try:
            yield context
        finally:
            context.close()
            browser.close()
