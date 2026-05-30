"""
個人用バックエンド: Meta Seamless (S2TT)。
音声セグメント → 翻訳テキストを1モデルで直接生成 (認識+翻訳が一体)。
※ weightsは CC-BY-NC-4.0 (非商用)。個人利用のみ。商用は backend_deepl を使うこと。

ここではVADで区切ったセグメントごとに seamless_communication の
Translator.predict(task="s2tt") を呼ぶ「セミストリーミング」方式。
真の逐次ストリーミング(EMMA policy)に発展させたい場合は README 参照。
"""
import asyncio
import numpy as np
import torch

import config

# ISO639-1 → Seamlessの言語コード(ISO639-3 / 独自)。
_SEAMLESS = {"en": "eng", "fr": "fra", "zh": "cmn", "ko": "kor", "ja": "jpn"}


class SeamlessBackend:
    name = "seamless"

    def __init__(self, source=None, target=None, model_name="seamlessM4T_v2_large"):
        self.source = source or config.SOURCE_LANG
        self.target = target or config.TARGET_LANG
        self.model_name = model_name
        self._translator = None
        self._device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    async def __aenter__(self):
        # ロードは重いのでスレッドへ逃がす
        await asyncio.to_thread(self._load)
        return self

    def _load(self):
        from seamless_communication.inference import Translator
        dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._translator = Translator(
            self.model_name,
            vocoder_name_or_card=None,   # S2TTは音声合成不要(テキストだけ欲しい)
            device=torch.device(self._device),
            dtype=dtype,
        )

    async def __aexit__(self, *exc):
        self._translator = None

    def _predict(self, seg: np.ndarray) -> str:
        wav = torch.from_numpy(seg).to(self._device).unsqueeze(0)  # [1, T]
        text_out, _ = self._translator.predict(
            input=wav,
            task_str="s2tt",
            tgt_lang=_SEAMLESS[self.target],
            src_lang=_SEAMLESS.get(self.source),
        )
        return str(text_out[0]).strip()

    async def process(self, seg) -> str:
        return await asyncio.to_thread(self._predict, seg)
