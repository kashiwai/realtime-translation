# meet_interpreter — 双方向リアルタイム音声通訳 (Meet / Zoom / Teams 共通)

会議ツールの「外側」で音声を出し入れするので、**Meet / Zoom / Teams どれでも同じ仕組み**で動く。
ツール側はマイク/スピーカーを仮想デバイスに指定するだけ。

```
聞く+訳す → 喋る の2経路を同時実行:

経路A (相手→自分)  Meetの音 →[仮想入力]→ 認識→翻訳 → Fish TTS → 自分のヘッドホン
経路B (自分→相手)  物理マイク → 認識→翻訳 → Fish TTS →[別の仮想出力]→ Meetのマイク
```

- 認識+翻訳: **DeepL**(商用) または **Seamless S2TT**(個人・非商用)
- 発話: **Fish Audio** WebSocket TTS (声質もここで指定)

---

## 1. セットアップ (macOS)

### 1-1. 仮想オーディオを2系統用意 (ループバック対策の肝)
経路Aと経路Bで**別々の仮想デバイス**を使わないと、自分の翻訳音声を自分が再翻訳する無限ループが起きる。
BlackHole の 2ch版 と 16ch版 を両方入れて分離する:

```bash
brew install blackhole-2ch
brew install blackhole-16ch
```

- 経路A の入力 = `BlackHole 2ch`   (Meetの音声をここへ流す)
- 経路B の出力 = `BlackHole 16ch`  (Meetのマイク入力にこれを指定)
- 経路B の入力 = 物理マイク (既定)   ← Meetの音と混ざらないので安全
- 経路A の出力 = 自分のヘッドホン (既定)

### 1-2. Python依存
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 個人用(Seamless)を使う場合のみ追加:
pip install torch
pip install git+https://github.com/facebookresearch/seamless_communication.git
```

### 1-3. APIキー
```bash
export FISH_API_KEY="xxxx"
export DEEPL_API_KEY="xxxx"
# DeepL Freeプランなら: export DEEPL_FREE=1
```

---

## 2. 会議ツール側の設定 (Meet/Zoom/Teams 共通)

各ツールの音声設定で:
- **マイク**   → `BlackHole 16ch`   (= 経路Bが吹き替えを流し込む先)
- **スピーカー** → `BlackHole 2ch`   (= 経路Aが相手の声を拾う先)

さらに「自分でも相手の生声を聞きたい」場合は、macOSの **Audio MIDI設定** で
`BlackHole 2ch + 自分のヘッドホン` をまとめた**複数出力装置**を作り、それをスピーカーに指定する。

---

## 3. デバイス名を確認して config に転記

```bash
python run.py --list-devices
```
表示された名前(部分一致でOK)を `config.py` の `A_INPUT_DEVICE` などに設定。

---

## 4. 起動

```bash
python run.py            # 双方向(A+B同時)
python run.py --mode a   # 相手→自分 だけ
python run.py --mode b   # 自分→相手 だけ (配線チェックに便利)
```

---

## 5. よくある調整

| やりたいこと | 設定 |
|---|---|
| 相手が仏/中/韓のとき | `config.py` の `A_SOURCE` と `B_TARGET` を fr/zh/ko に |
| 自分の声で相手言語を喋らせる | Fishで自分の声をクローン → そのID を `B_VOICE_ID` に |
| 個人検証でSeamlessを試す | `A_BACKEND=seamless` (※非商用ライセンス。個人利用のみ) |
| 商用(自社ツール)で使う | A/B とも `deepl`。Seamlessは使わない |
| 訳の出が遅い/細切れ | `config.py` の `VAD_MIN_SILENCE_MS` を調整(大=まとまり重視/小=速さ重視) |
| 二重に訳される | 経路A入力と経路B出力が**同じ仮想デバイスになっていないか**確認 |

---

## 5.5 真の逐次モード (同時通訳) — run_streaming.py

`run.py` はVADで「話し終わり」を待つ逐次通訳。
`run_streaming.py` は**話している最中から訳し始める**真の同時通訳。

**仕組み**: ストリーミングASR(whisper_streaming, LocalAgreement-2)が、音声を食わせ続けながら
"確定した(commitした)テキスト"だけを逐次出力 → 節〜文のまとまり(CommitBuffer)で DeepL → Fish TTS。
確定ぶんしか喋らせないので「言い直し」が起きない。

```bash
# 追加依存(商用OK):
pip install faster-whisper librosa
git clone https://github.com/ufal/whisper_streaming   # whisper_online.py をPYTHONPATHに

python run_streaming.py --source en --target ja --model medium
```

**遅延と語順の調整つまみ**:
| つまみ | 効果 |
|---|---|
| `--min-chunk`(秒) | 小さい=低遅延だが高負荷。大きい=安定だが遅延増 |
| `--max-chars` | 区切りが来なくても強制翻訳する長さ。日↔英の語順崩れを抑える"意図的な待ち" |
| `--model` | Mac(CPU)は medium 推奨。CUDA機なら large-v3 |

**Macの現実**: faster-whisperはMPS非対応→CPU動作。large-v3は重いので medium から。
双方向の逐次は重い(Whisper2本)ので、GPU機推奨。CPUのみなら経路Bは run.py(VAD版)併用が現実的。

**個人ルートで真の逐次を極めるなら**: Seamless**Streaming**(SimulEval / EMMA policy)が
逐次同時通訳用に設計されており語順差に強い。`--engine seamless` で使う。

```bash
pip install git+https://github.com/facebookresearch/seamless_communication.git
pip install simuleval fairseq2
python run_streaming.py --engine seamless --source en --target ja
```

**注意(重要)**:
- weightsは **CC-BY-NC-4.0 = 非商用**。個人のMeet理解用途のみ。自社ツールは whisper/DeepL を使う。
- 公式CLI `streaming_evaluate` は**ファイルベースでマイク非対応**。リアルタイムは
  SimulEvalエージェントを states 管理で回す必要があり、API名は版で変わる。
- **最短で逐次の質を体感する手順**: 先に公式demoを動かす →
  `cd seamless_communication/demo && pip install -r requirements.txt && python app.py`
  (※demoは内蔵TTSで喋るのでFishの声にはならない。逐次の質だけ確認する用)
- `backend_seamless_streaming.py` の `_build_agent` / `insert` が動かない場合は、
  demo/app.py の「音声をpushして確定テキストをpopする箇所」に合わせて調整する。

---

## 6. ライセンス注意 (重要)

- **DeepL / Fish Audio API** … 各社の利用規約に従えば商用OK。自社ツールはこちら。
- **Seamless (weights)** … CC-BY-NC-4.0 = **非商用のみ**。個人のMeet理解用途に限定すること。
- 上記は参考情報。最終判断は各社の規約原文で確認を(作者は法律家ではありません)。

## 構成ファイル
```
config.py           設定(言語/デバイス/声/APIキー)
audio_io.py         マイク入力+VAD分割 / PCM再生
fish_tts.py         Fish WebSocket TTS クライアント
backend_deepl.py    Fish ASR + DeepL翻訳 (商用)
backend_seamless.py Seamless S2TT (個人・非商用)
run.py              双方向2経路の並行実行 (VAD版=逐次通訳)
streaming_asr.py    ストリーミングASR + コミット戦略 (whisper, 真の逐次)
backend_seamless_streaming.py  SeamlessStreaming S2TT (個人・非商用, 真の逐次)
run_streaming.py    真の逐次パイプライン (--engine whisper / seamless)
```
