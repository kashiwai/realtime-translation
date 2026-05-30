"""
オーディオ入出力。
- 入力: マイク/ループバックを16kHzで取り込み、silero-vadで「発話のかたまり」に切る
- 出力: Fish TTSから来たPCM(s16le)を再生キュー経由でスピーカー/仮想デバイスへ

VADで区切ることで「相手が一息ついた瞬間」に翻訳を走らせる = 同時通訳のテンポを作る。
"""
import queue
import threading
import numpy as np
import sounddevice as sd

import config


def list_devices():
    print(sd.query_devices())


def _resolve_device(name, kind):
    """デバイス名(部分一致)をindexへ。空ならNone(既定デバイス)。"""
    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower():
            max_ch = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
            if max_ch > 0:
                return i
    raise ValueError(f"{kind} device not found: {name}")


class SpeechSegmenter:
    """マイク入力をVADで発話単位に切り出し、float32 16k mono の numpy を yield する。"""

    def __init__(self, input_device=None):
        from silero_vad import load_silero_vad, VADIterator
        self.model = load_silero_vad()
        self.vad = VADIterator(
            self.model,
            threshold=config.VAD_THRESHOLD,
            sampling_rate=config.INPUT_SAMPLE_RATE,
            min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
            speech_pad_ms=config.VAD_SPEECH_PAD_MS,
        )
        self._win = 512  # silero-vadは16kで512サンプル単位
        dev = input_device if input_device is not None else config.INPUT_DEVICE
        self._device = _resolve_device(dev, "input")
        self._q = queue.Queue()
        self._buf = []          # 現在の発話バッファ
        self._collecting = False
        self._stop = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass  # オーバーフロー等は無視して継続
        self._q.put(indata[:, 0].copy())

    def segments(self):
        """発話セグメントを順次返すジェネレータ。"""
        stream = sd.InputStream(
            samplerate=config.INPUT_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self._win,
            device=self._device,
            callback=self._callback,
        )
        min_speech = int(config.VAD_MIN_SPEECH_MS / 1000 * config.INPUT_SAMPLE_RATE)
        max_speech = int(config.VAD_MAX_SPEECH_MS / 1000 * config.INPUT_SAMPLE_RATE)
        with stream:
            carry = np.empty(0, dtype="float32")
            buf_len = 0  # 収集中バッファのサンプル数
            while not self._stop.is_set():
                chunk = self._q.get()
                carry = np.concatenate([carry, chunk])
                # 512サンプルずつVADへ
                while len(carry) >= self._win:
                    frame = carry[:self._win]
                    carry = carry[self._win:]
                    if self._collecting:
                        self._buf.append(frame)
                        buf_len += len(frame)
                    res = self.vad(frame, return_seconds=False)
                    seg = None
                    if res is not None and "start" in res:
                        self._collecting = True
                        self._buf = [frame]
                        buf_len = len(frame)
                    end_now = res is not None and "end" in res
                    if self._collecting and not end_now and buf_len >= max_speech:
                        # 区切りが来なくても強制カット(発話は継続扱い)
                        seg = np.concatenate(self._buf)
                        self._buf = []
                        buf_len = 0
                    elif end_now and self._collecting:
                        self._collecting = False
                        seg = np.concatenate(self._buf) if self._buf else np.empty(0, "float32")
                        self._buf = []
                        buf_len = 0
                    if seg is None or len(seg) < min_speech:
                        continue
                    # 音量ゲート: 無音/雑音はASRに渡さない(幻聴防止)
                    rms = float(np.sqrt(np.mean(seg.astype("float32") ** 2)))
                    if rms < config.VAD_RMS_FLOOR:
                        continue
                    yield seg

    def stop(self):
        self._stop.set()


class PcmPlayer:
    """s16le PCM bytes を受け取り、別スレッドで連続再生する。"""

    def __init__(self, output_device=None):
        dev = output_device if output_device is not None else config.OUTPUT_DEVICE
        self._device = _resolve_device(dev, "output")
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def feed(self, pcm_bytes: bytes):
        self._q.put(pcm_bytes)

    def _run(self):
        with sd.RawOutputStream(
            samplerate=config.TTS_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=self._device,
        ) as out:
            while not self._stop.is_set():
                try:
                    data = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if data:
                    out.write(np.frombuffer(data, dtype="int16"))

    def stop(self):
        self._stop.set()


def float_to_wav_bytes(seg: np.ndarray, sr: int = config.INPUT_SAMPLE_RATE) -> bytes:
    """float32[-1,1] の numpy を WAV(16bit) のバイト列へ。Fish ASRへ渡す用。"""
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
