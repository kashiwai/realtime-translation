"""
双方向リアルタイム通訳。経路A(相手→自分)と経路B(自分→相手)を同時に走らせる。

使い方:
  python run.py --list-devices            # デバイス名を確認(configに転記)
  python run.py                           # 双方向(A+B)
  python run.py --mode a                  # 経路Aだけ(相手→自分)
  python run.py --mode b                  # 経路Bだけ(自分→相手)

言語や声・デバイスは config.py の A_* / B_* で設定。
相手が英語以外(仏/中/韓)のときは A_SOURCE / B_TARGET を変えるだけ。
"""
import argparse
import asyncio
import threading

import config
import audio_io
from fish_tts import FishTTSSession


def make_backend(name, source, target):
    if name == "deepl":
        from backend_deepl import DeepLBackend
        return DeepLBackend(source, target)
    elif name == "seamless":
        from backend_seamless import SeamlessBackend
        return SeamlessBackend(source, target)
    raise ValueError(f"unknown backend: {name}")


class Route:
    def __init__(self, tag, source, target, backend, in_dev, out_dev, voice):
        self.tag = tag
        self.source = source
        self.target = target
        self.backend = backend
        self.in_dev = in_dev
        self.out_dev = out_dev
        self.voice = voice


async def run_route(route: Route):
    player = audio_io.PcmPlayer(output_device=route.out_dev)
    player.start()
    segmenter = audio_io.SpeechSegmenter(input_device=route.in_dev)

    loop = asyncio.get_running_loop()
    seg_queue: asyncio.Queue = asyncio.Queue(maxsize=8)

    # VAD(同期ジェネレータ)を別スレッドで回し、満杯時は最古を捨てて最新を入れる
    # → 翻訳が詰まっても遅延が雪だるま化しない
    def feed_segments():
        for seg in segmenter.segments():
            def _put(s=seg):
                if seg_queue.full():
                    try:
                        seg_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                seg_queue.put_nowait(s)
            loop.call_soon_threadsafe(_put)

    threading.Thread(target=feed_segments, daemon=True).start()

    backend = make_backend(route.backend, route.source, route.target)
    print(f"[{route.tag}] {route.source}→{route.target} ({route.backend})  "
          f"in='{route.in_dev or '既定'}' out='{route.out_dev or '既定'}'")

    async with backend, FishTTSSession(on_audio=player.feed, voice_id=route.voice) as tts:
        while True:
            seg = await seg_queue.get()
            try:
                text = await backend.process(seg)
            except Exception as e:
                print(f"[{route.tag}] ! error: {e}")
                continue
            if text:
                print(f"[{route.tag}] ▶ {text}")
                await tts.speak(text)


async def main_async(mode):
    routes = []
    if mode in ("both", "a"):
        routes.append(Route("A 相手→自分", config.A_SOURCE, config.A_TARGET,
                            config.A_BACKEND, config.A_INPUT_DEVICE,
                            config.A_OUTPUT_DEVICE, config.A_VOICE_ID))
    if mode in ("both", "b"):
        routes.append(Route("B 自分→相手", config.B_SOURCE, config.B_TARGET,
                            config.B_BACKEND, config.B_INPUT_DEVICE,
                            config.B_OUTPUT_DEVICE, config.B_VOICE_ID))
    print("Ctrl+C で停止\n")
    await asyncio.gather(*(run_route(r) for r in routes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["both", "a", "b"], default="both")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        audio_io.list_devices()
        return
    try:
        asyncio.run(main_async(args.mode))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
