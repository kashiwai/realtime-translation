"""
ローカルWeb UI サーバ。
既存の翻訳エンジン(audio_io / backend_deepl / fish_tts)を再利用し、
ブラウザから デバイス選択 / 言語ペア / APIキー / リアルタイム調整 を設定して
開始・停止できるようにする。認識結果と訳文はWebSocketでライブ配信する。

起動:  ./serve.sh   (= python server.py)
ブラウザ: http://127.0.0.1:8765
"""
import asyncio
import json
import os
import threading
from pathlib import Path

from aiohttp import web, WSMsgType
import sounddevice as sd

import config
import audio_io
from fish_tts import FishTTSSession

HERE = Path(__file__).parent
SETTINGS_FILE = HERE / "settings.local.json"   # ローカル専用(.gitignore済み)
STATIC_DIR = HERE / "webui"

# ── 設定の既定値(config.pyから引き継ぎ) ─────────────────────
DEFAULT_SETTINGS = {
    "fish_api_key": config.FISH_API_KEY,
    "deepl_api_key": config.DEEPL_API_KEY,
    "deepl_free": config.DEEPL_FREE,
    "a": {"source": config.A_SOURCE, "target": config.A_TARGET,
          "in_dev": config.A_INPUT_DEVICE, "out_dev": config.A_OUTPUT_DEVICE, "voice": config.A_VOICE_ID},
    "b": {"source": config.B_SOURCE, "target": config.B_TARGET,
          "in_dev": config.B_INPUT_DEVICE, "out_dev": config.B_OUTPUT_DEVICE, "voice": config.B_VOICE_ID},
    "vad_max_speech_ms": config.VAD_MAX_SPEECH_MS,
    "vad_min_silence_ms": config.VAD_MIN_SILENCE_MS,
    "vad_rms_floor": config.VAD_RMS_FLOOR,
}


def load_settings():
    s = json.loads(json.dumps(DEFAULT_SETTINGS))  # deep copy
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text())
            for k, v in saved.items():
                if isinstance(v, dict) and isinstance(s.get(k), dict):
                    s[k].update(v)
                else:
                    s[k] = v
        except Exception as e:
            print("settings load error:", e)
    return s


def save_settings(s):
    SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2))


def apply_to_config(s):
    """設定値を config モジュールへ反映(エンジンは config を参照するため)。"""
    config.FISH_API_KEY = s["fish_api_key"]
    config.DEEPL_API_KEY = s["deepl_api_key"]
    config.DEEPL_FREE = bool(s["deepl_free"])
    config.VAD_MAX_SPEECH_MS = int(s["vad_max_speech_ms"])
    config.VAD_MIN_SILENCE_MS = int(s["vad_min_silence_ms"])
    config.VAD_RMS_FLOOR = float(s["vad_rms_floor"])


def list_devices():
    inputs, outputs = [], []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            inputs.append({"index": i, "name": d["name"]})
        if d["max_output_channels"] > 0:
            outputs.append({"index": i, "name": d["name"]})
    return {"inputs": inputs, "outputs": outputs}


def make_backend(source, target):
    from backend_deepl import DeepLBackend
    return DeepLBackend(source, target)


class Engine:
    """経路A/Bの起動・停止と、認識/翻訳イベントのコールバック配信を管理。"""

    def __init__(self, emit):
        self.emit = emit
        self._tasks = []
        self._segmenters = []
        self._stop = None
        self.running = False
        self.mode = "both"

    async def start(self, settings, mode):
        if self.running:
            return
        apply_to_config(settings)
        self._stop = asyncio.Event()
        routes = []
        if mode in ("both", "a"):
            r = dict(settings["a"]); r["tag"] = "A"; routes.append(r)
        if mode in ("both", "b"):
            r = dict(settings["b"]); r["tag"] = "B"; routes.append(r)
        self.mode = mode
        self.running = True
        self._tasks = [asyncio.create_task(self._run_route(r)) for r in routes]
        self.emit({"type": "status", "running": True, "mode": mode})

    async def stop(self):
        if not self.running:
            return
        self._stop.set()
        for sg in self._segmenters:
            try:
                sg.stop()
            except Exception:
                pass
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks, self._segmenters = [], []
        self.running = False
        self.emit({"type": "status", "running": False})

    async def _run_route(self, route):
        tag = route["tag"]
        player = audio_io.PcmPlayer(output_device=route["out_dev"] or None)
        player.start()
        try:
            segmenter = audio_io.SpeechSegmenter(input_device=route["in_dev"] or None)
        except Exception as e:
            self.emit({"type": "error", "tag": tag, "message": f"入力デバイス初期化失敗: {e}"})
            return
        self._segmenters.append(segmenter)

        loop = asyncio.get_running_loop()
        seg_queue: asyncio.Queue = asyncio.Queue(maxsize=8)

        def feed():
            try:
                for seg in segmenter.segments():
                    if self._stop.is_set():
                        break
                    def _put(s=seg):
                        if seg_queue.full():
                            try:
                                seg_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                        seg_queue.put_nowait(s)
                    loop.call_soon_threadsafe(_put)
            except Exception as e:
                loop.call_soon_threadsafe(
                    lambda: self.emit({"type": "error", "tag": tag, "message": f"入力エラー: {e}"}))

        threading.Thread(target=feed, daemon=True).start()
        backend = make_backend(route["source"], route["target"])
        self.emit({"type": "route", "tag": tag,
                   "source": route["source"], "target": route["target"],
                   "in_dev": route["in_dev"] or "既定", "out_dev": route["out_dev"] or "既定"})
        try:
            async with backend, FishTTSSession(on_audio=player.feed, voice_id=route["voice"] or None) as tts:
                while not self._stop.is_set():
                    try:
                        seg = await asyncio.wait_for(seg_queue.get(), timeout=0.3)
                    except asyncio.TimeoutError:
                        continue
                    try:
                        src, dst = await backend.process_verbose(seg)
                    except Exception as e:
                        self.emit({"type": "error", "tag": tag, "message": str(e)})
                        continue
                    if dst:
                        self.emit({"type": "line", "tag": tag, "src": src, "dst": dst})
                        await tts.speak(dst)
        except asyncio.CancelledError:
            pass
        finally:
            player.stop()


# ── HTTP / WebSocket ────────────────────────────────────────
class App:
    def __init__(self):
        self.settings = load_settings()
        self.clients = set()
        self.loop = None
        self.engine = Engine(self._emit)

    def _emit(self, payload):
        # 各WSクライアントへ配信(asyncコンテキストから呼ばれる)
        dead = []
        for ws in self.clients:
            try:
                asyncio.create_task(ws.send_json(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def index(self, request):
        return web.FileResponse(STATIC_DIR / "index.html")

    async def get_devices(self, request):
        return web.json_response(list_devices())

    async def get_config(self, request):
        return web.json_response(self.settings)

    async def post_config(self, request):
        data = await request.json()
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(self.settings.get(k), dict):
                self.settings[k].update(v)
            else:
                self.settings[k] = v
        save_settings(self.settings)
        apply_to_config(self.settings)
        return web.json_response({"ok": True})

    async def post_start(self, request):
        data = await request.json()
        mode = data.get("mode", "both")
        await self.engine.start(self.settings, mode)
        return web.json_response({"ok": True, "running": True})

    async def post_stop(self, request):
        await self.engine.stop()
        return web.json_response({"ok": True, "running": False})

    async def get_status(self, request):
        return web.json_response({"running": self.engine.running, "mode": self.engine.mode})

    async def ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)
        await ws.send_json({"type": "status", "running": self.engine.running, "mode": self.engine.mode})
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self.clients.discard(ws)
        return ws

    def build(self):
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/api/devices", self.get_devices)
        app.router.add_get("/api/config", self.get_config)
        app.router.add_post("/api/config", self.post_config)
        app.router.add_post("/api/start", self.post_start)
        app.router.add_post("/api/stop", self.post_stop)
        app.router.add_get("/api/status", self.get_status)
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_static("/static/", STATIC_DIR)
        return app


def main():
    a = App()
    app = a.build()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    print(f"\n  リアルタイム翻訳 Web UI:  http://{host}:{port}\n  停止: Ctrl+C\n")
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
