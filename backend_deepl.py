"""
商用バックエンド: Fish Audio ASR + DeepL 翻訳。
音声セグメント(float32 16k) → 認識 → 翻訳 → テキストを返す。

ASR:  POST https://api.fish.audio/v1/asr  (multipart/form-data)
翻訳: POST https://api(.|-free.)deepl.com/v2/translate
"""
import asyncio
import aiohttp

import config
import audio_io

FISH_ASR_URL = "https://api.fish.audio/v1/asr"

# ISO639-1 → DeepLの言語コード。DeepLは中国語をZH、出力先は地域指定可。
_DEEPL_SOURCE = {"en": "EN", "fr": "FR", "zh": "ZH", "ko": "KO", "ja": "JA"}
_DEEPL_TARGET = {"en": "EN-US", "fr": "FR", "zh": "ZH", "ko": "KO", "ja": "JA"}
# Fish ASRのlanguageヒント(認識精度向上)。Noneでも自動判定する。
_FISH_LANG = {"en": "en", "fr": "fr", "zh": "zh", "ko": "ko", "ja": "ja"}


class DeepLBackend:
    name = "deepl"

    def __init__(self, source=None, target=None):
        self.source = source or config.SOURCE_LANG
        self.target = target or config.TARGET_LANG
        self._session = None
        self._deepl_host = (
            "https://api-free.deepl.com" if config.DEEPL_FREE else "https://api.deepl.com"
        )

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    async def _asr(self, seg) -> str:
        wav = audio_io.float_to_wav_bytes(seg)
        form = aiohttp.FormData()
        form.add_field("audio", wav, filename="seg.wav", content_type="audio/wav")
        form.add_field("language", _FISH_LANG.get(self.source, ""))
        form.add_field("ignore_timestamps", "true")  # 短文の遅延を抑える
        headers = {"Authorization": f"Bearer {config.FISH_API_KEY}"}
        async with self._session.post(FISH_ASR_URL, data=form, headers=headers) as r:
            r.raise_for_status()
            data = await r.json()
            return (data.get("text") or "").strip()

    async def _translate(self, text: str) -> str:
        headers = {"Authorization": f"DeepL-Auth-Key {config.DEEPL_API_KEY}"}
        payload = {
            "text": [text],
            "source_lang": _DEEPL_SOURCE.get(self.source),
            "target_lang": _DEEPL_TARGET.get(self.target, "JA"),
        }
        async with self._session.post(
            f"{self._deepl_host}/v2/translate", json=payload, headers=headers
        ) as r:
            r.raise_for_status()
            data = await r.json()
            return data["translations"][0]["text"].strip()

    async def process(self, seg) -> str:
        """発話セグメント → 翻訳テキスト('' なら破棄)。(VAD版バックエンド用)"""
        text = await self._asr(seg)
        if not text:
            return ""
        return await self._translate(text)

    async def translate(self, text: str) -> str:
        """確定テキストの翻訳のみ。逐次(ストリーミング)パイプラインから使う。"""
        if not text.strip():
            return ""
        return await self._translate(text)

    async def process_verbose(self, seg):
        """発話セグメント → (認識した原文, 翻訳文)。Web UIで両方表示する用。"""
        src = await self._asr(seg)
        if not src:
            return "", ""
        return src, await self._translate(src)
