"""
真の逐次(同時通訳)パイプライン。エンジンを2系統から選べる:

  --engine whisper : whisper_streaming(LocalAgreement) → DeepL翻訳 → Fish TTS  [商用OK]
  --engine seamless: SeamlessStreaming(S2TT/EMMA policy) → Fish TTS            [非商用・個人のみ]

whisperは部品の組み合わせ。seamlessはモデルが語順の待ち判断を内蔵するぶん日↔英に強い。
どちらも「確定したテキストだけ」をFishに流すので言い直しが起きない。

使い方:
  python run_streaming.py --list-devices
  python run_streaming.py --engine seamless --source en --target ja
  python run_streaming.py --engine whisper  --source en --target ja --model medium

双方向: 本スクリプトを別の入出力デバイス/言語でもう1つ起動(経路B)。
  ※モデル2本は重い。GPU機推奨。CPUのみなら経路Bは run.py(VAD版)併用が現実的。

ライセンス: whisper=商用OK / seamless=非商用(個人のみ)。
"""
import argparse
import asyncio
import queue

import sounddevice as sd

import config
import audio_io
from fish_tts import FishTTSSession


async def pipeline(args):
    in_dev = audio_io._resolve_device(config.A_INPUT_DEVICE, "input")
    player = audio_io.PcmPlayer(output_device=config.A_OUTPUT_DEVICE)
    player.start()

    # ── エンジン構築 ──────────────────────────────
    deepl = None
    if args.engine == "whisper":
        from streaming_asr import StreamingASR, CommitBuffer
        engine = StreamingASR(args.source, model=args.model, min_chunk_sec=args.min_chunk)
        commit = CommitBuffer(max_chars=args.max_chars)
        from backend_deepl import DeepLBackend
        deepl_ctx = DeepLBackend(args.source, args.target)
    else:  # seamless: 聞く+訳すが一体。翻訳ステップ不要。
        from backend_seamless_streaming import SeamlessStreamingS2TT
        engine = SeamlessStreamingS2TT(args.source, args.target)
        commit = None
        deepl_ctx = None

    raw_q: queue.Queue = queue.Queue()

    def cb(indata, frames, t, status):
        raw_q.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=16000, channels=1, dtype="float32",
        blocksize=int(0.2 * 16000), device=in_dev, callback=cb,
    )

    print(f"[逐次:{args.engine}] {args.source}→{args.target}  "
          f"in='{config.A_INPUT_DEVICE or '既定'}'  (Ctrl+C で停止)")

    loop = asyncio.get_running_loop()

    async def run_loop(tts, deepl):
        with stream:
            while True:
                got = False
                while not raw_q.empty():
                    chunk = raw_q.get_nowait()
                    if args.engine == "whisper":
                        engine.insert(chunk)
                    else:
                        # seamless: insertが確定訳文を都度返す
                        text = await loop.run_in_executor(None, engine.insert, chunk)
                        if text:
                            print(f"  ▶ {text}")
                            await tts.speak(text)
                    got = True
                if not got:
                    await asyncio.sleep(0.05)

                if args.engine == "whisper":
                    confirmed = await loop.run_in_executor(None, engine.poll)
                    if confirmed:
                        commit.add(confirmed)
                    for chunk_text in commit.pop_ready():
                        translated = await deepl.translate(chunk_text)
                        if translated:
                            print(f"  {chunk_text} → {translated}")
                            await tts.speak(translated)

    if args.engine == "whisper":
        async with deepl_ctx as deepl, \
                   FishTTSSession(on_audio=player.feed, voice_id=config.A_VOICE_ID) as tts:
            await run_loop(tts, deepl)
    else:
        async with FishTTSSession(on_audio=player.feed, voice_id=config.A_VOICE_ID) as tts:
            await run_loop(tts, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["whisper", "seamless"], default="seamless")
    p.add_argument("--source", default=config.A_SOURCE)
    p.add_argument("--target", default=config.A_TARGET)
    p.add_argument("--model", default="medium", help="whisper用: tiny/base/small/medium/large-v3")
    p.add_argument("--min-chunk", type=float, default=1.0, help="whisper用: ASRを進める最小音声長(秒)")
    p.add_argument("--max-chars", type=int, default=60, help="whisper用: 強制翻訳の文字数(語順安定つまみ)")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        audio_io.list_devices()
        return
    try:
        asyncio.run(pipeline(args))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
