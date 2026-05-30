"""
Fish Audio WebSocket TTS クライアント。
wss://api.fish.audio/v1/tts/live にMessagePackで接続。

プロトコル(公式AsyncAPI準拠):
  送信: start → (text / flush)* → stop
  受信: audio* → finish
セッションを張りっぱなしにして、翻訳が確定した文を text→flush で投げると、
確定ぶんから順次 audio が返る = 同時通訳のテンポで喋らせられる。
"""
import asyncio
import ormsgpack
import websockets

import config

WSS_URL = "wss://api.fish.audio/v1/tts/live"


class FishTTSSession:
    def __init__(self, on_audio, voice_id=None, model=None, sample_rate=None):
        """
        on_audio: PCM(s16le) bytes を受け取るコールバック (例: PcmPlayer.feed)
        """
        self.on_audio = on_audio
        self.voice_id = voice_id if voice_id is not None else config.FISH_VOICE_ID
        self.model = model or config.FISH_TTS_MODEL
        self.sample_rate = sample_rate or config.TTS_SAMPLE_RATE
        self._ws = None
        self._recv_task = None

    async def __aenter__(self):
        headers = {
            "Authorization": f"Bearer {config.FISH_API_KEY}",
            "model": self.model,
        }
        self._ws = await websockets.connect(
            WSS_URL, additional_headers=headers, max_size=None
        )
        request = {
            "text": "",
            "format": "pcm",
            "sample_rate": self.sample_rate,
            "latency": config.FISH_TTS_LATENCY,
            "temperature": 0.7,
            "normalize": True,
        }
        if self.voice_id:
            request["reference_id"] = self.voice_id
        await self._ws.send(ormsgpack.packb({"event": "start", "request": request}))
        self._recv_task = asyncio.create_task(self._receiver())
        return self

    async def _receiver(self):
        try:
            async for msg in self._ws:
                data = ormsgpack.unpackb(msg)
                ev = data.get("event")
                if ev == "audio":
                    self.on_audio(data["audio"])     # s16le PCM bytes
                elif ev == "finish":
                    break
        except websockets.ConnectionClosed:
            pass

    async def speak(self, text: str):
        """翻訳テキストを送って即時生成させる。"""
        if not text.strip():
            return
        await self._ws.send(ormsgpack.packb({"event": "text", "text": text}))
        await self._ws.send(ormsgpack.packb({"event": "flush"}))

    async def __aexit__(self, *exc):
        try:
            await self._ws.send(ormsgpack.packb({"event": "stop"}))
            if self._recv_task:
                await asyncio.wait_for(self._recv_task, timeout=10)
        except Exception:
            pass
        finally:
            await self._ws.close()
