"""
ストリーミングASR (真の逐次の心臓部)。

whisper_streaming (ufal) の LocalAgreement-2 を使い、話している最中の音声から
"確定した(commitした)テキスト断片" だけを逐次返す。
→ 確定ぶんしか翻訳・発話に回さないので、TTSの「喋り始めたら戻せない」問題を回避。

依存:
  pip install faster-whisper librosa
  git clone https://github.com/ufal/whisper_streaming   # whisper_online.py を import可能に
  (より低レイテンシな後継 https://github.com/ufal/SimulStreaming も差し替え可)

Mac(Apple Silicon)の注意:
  faster-whisper(CTranslate2)はMPS非対応 → CPU動作。large-v3はCPUだと重いので
  STREAMING_ASR_MODEL は medium 程度を推奨。CUDA機があれば large-v3 + GPU が最良。
"""
import numpy as np

# ISO639-1 → whisper言語コード(そのまま2文字でOK: en/fr/zh/ko/ja)
_WHISPER_LANG = {"en": "en", "fr": "fr", "zh": "zh", "ko": "ko", "ja": "ja"}


class StreamingASR:
    SR = 16000

    def __init__(self, lang, model="medium", min_chunk_sec=1.0, compute_type="int8"):
        from whisper_online import FasterWhisperASR, OnlineASRProcessor
        self.asr = FasterWhisperASR(
            lan=_WHISPER_LANG.get(lang, lang),
            modelsize=model,
            # CPUなら compute_type="int8"、CUDAなら "float16" 等
        )
        try:
            self.asr.model.compute_type = compute_type  # バックエンド差異を吸収(任意)
        except Exception:
            pass
        self.online = OnlineASRProcessor(self.asr)
        self.min_chunk = int(min_chunk_sec * self.SR)
        self._buf = np.empty(0, dtype=np.float32)

    def insert(self, audio: np.ndarray):
        """マイク等から来た生の16k float32を貯める。"""
        self._buf = np.concatenate([self._buf, audio])

    def poll(self) -> str:
        """min_chunk たまったら認識を1回進め、新たに"確定"したテキストを返す。
        まだ確定なし or データ不足なら ''。"""
        if len(self._buf) < self.min_chunk:
            return ""
        self.online.insert_audio_chunk(self._buf)
        self._buf = np.empty(0, dtype=np.float32)
        _beg, _end, text = self.online.process_iter()
        return text or ""

    def finish(self) -> str:
        """残バッファを吐き出してセッション終了。"""
        if len(self._buf):
            self.online.insert_audio_chunk(self._buf)
            self._buf = np.empty(0, dtype=np.float32)
        _beg, _end, text = self.online.finish()
        return text or ""


class CommitBuffer:
    """確定テキスト断片を溜め、節〜文のまとまりで翻訳へ流す。
    日↔英の語順崩れを抑えるための"意図的な少しの待ち(decalage)"を担う層。
    """
    import re as _re
    _END = _re.compile(r"[。．.!?！？]")

    def __init__(self, max_chars=60):
        self.buf = ""
        self.max = max_chars

    def add(self, text: str):
        self.buf += text

    def pop_ready(self):
        """翻訳に回せる(=区切りの付いた)かたまりをリストで返す。"""
        out = []
        while True:
            m = self._END.search(self.buf)
            if m:
                out.append(self.buf[:m.end()].strip())
                self.buf = self.buf[m.end():]
            elif len(self.buf) >= self.max:
                out.append(self.buf.strip())
                self.buf = ""
            else:
                break
        return [s for s in out if s]

    def flush(self):
        s = self.buf.strip()
        self.buf = ""
        return s
