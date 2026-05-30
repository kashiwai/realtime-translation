"""
共通設定。環境変数 or ここを直接編集。
言語コードは ISO 639-1 (en, fr, zh, ko, ja)。各バックエンドが必要な形式へ変換する。

【双方向の考え方】
  経路A: 相手 → 自分   (相手の言語を聞いて、自分の言語の吹き替えを自分が聞く)
  経路B: 自分 → 相手   (自分の言語を話して、相手の言語の吹き替えを相手に届ける)
  2経路を別々の仮想オーディオデバイスに分離するのがループバック対策の肝(README参照)。
"""
import os

# ── APIキー ─────────────────────────────────────────────
FISH_API_KEY = os.environ.get("FISH_API_KEY", "")          # https://fish.audio/app/api-keys
DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY", "")        # DeepL API Pro/Free
DEEPL_FREE = os.environ.get("DEEPL_FREE", "0") == "1"      # Freeプランなら "1"

# ── Fish TTS 共通 ───────────────────────────────────────
FISH_TTS_MODEL = os.environ.get("FISH_TTS_MODEL", "s1")    # s1 / s2-pro
FISH_TTS_LATENCY = "low"                                    # low / balanced / normal
TTS_SAMPLE_RATE = 44100                                     # pcm出力サンプルレート

# ═══════════════════════════════════════════════════════
#  経路A : 相手 → 自分 （相手の声を訳して自分が聞く）
# ═══════════════════════════════════════════════════════
A_SOURCE = os.environ.get("A_SOURCE", "en")    # 相手の言語 en/fr/zh/ko/ja
A_TARGET = os.environ.get("A_TARGET", "ja")    # 自分が聞きたい言語
A_BACKEND = os.environ.get("A_BACKEND", "deepl")        # deepl(商用) / seamless(個人)
A_INPUT_DEVICE  = os.environ.get("A_INPUT_DEVICE",  "BlackHole 2ch")  # Meetの音が入る仮想入力
A_OUTPUT_DEVICE = os.environ.get("A_OUTPUT_DEVICE", "")              # 自分のヘッドホン(空=既定)
A_VOICE_ID = os.environ.get("A_VOICE_ID", "")           # 吹き替え声のreference_id(任意)

# ═══════════════════════════════════════════════════════
#  経路B : 自分 → 相手 （自分の声を訳して相手に届ける）
# ═══════════════════════════════════════════════════════
B_SOURCE = os.environ.get("B_SOURCE", "ja")    # 自分の言語
B_TARGET = os.environ.get("B_TARGET", "en")    # 相手の言語
B_BACKEND = os.environ.get("B_BACKEND", "deepl")
B_INPUT_DEVICE  = os.environ.get("B_INPUT_DEVICE",  "")                 # 自分の物理マイク(空=既定)
B_OUTPUT_DEVICE = os.environ.get("B_OUTPUT_DEVICE", "BlackHole 16ch")   # Meetのマイク入力に指定する別系統
B_VOICE_ID = os.environ.get("B_VOICE_ID", "")           # 自分の声クローンID。入れると自分の声で相手言語を喋る

# 後方互換(単方向デフォルト)。run.py から経路別に明示指定するため通常未使用。
SOURCE_LANG = A_SOURCE
TARGET_LANG = A_TARGET
FISH_VOICE_ID = A_VOICE_ID
INPUT_DEVICE = A_INPUT_DEVICE
OUTPUT_DEVICE = A_OUTPUT_DEVICE

# ── オーディオI/O 共通パラメータ ─────────────────────────
INPUT_SAMPLE_RATE = 16000                                   # ASR/Seamlessとも16k前提
VAD_THRESHOLD = 0.6                                         # silero-vad 発話確率しきい値(高=誤検出減)
VAD_MIN_SILENCE_MS = 600                                    # この無音で発話終了とみなす
VAD_MIN_SPEECH_MS = 400                                     # これ未満の発話は無視(ノイズ除去)
VAD_SPEECH_PAD_MS = 150                                     # 前後パディング
VAD_RMS_FLOOR = float(os.environ.get("VAD_RMS_FLOOR", "0.012"))  # この音量未満の区間はASRに渡さない(幻聴防止)
