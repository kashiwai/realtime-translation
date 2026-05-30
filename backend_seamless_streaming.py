"""
SeamlessStreaming (S2TT) を SimulEval エージェントで逐次に回すラッパー。
真の同時通訳: モデルが EMMA 系ポリシーで「待つ/今訳す」を判断するので、
語順の違う日↔英でも自然な区切りで訳文が出る(whisper+翻訳より語順に強い)。

────────────────────────────────────────────────────────────
【ライセンス】weights は CC-BY-NC-4.0 = 非商用。個人利用のみ。
            自社ツール(商用)では使わないこと → そちらは backend_deepl。
【版依存・最重要】
  SeamlessStreaming のリアルタイム states 回し API は
  simuleval / fairseq2 / seamless_communication の版で関数・クラス名が変わる。
  公式のリアルタイム実装は demo/app.py が唯一の実例。
  下の _build_agent / insert / poll が動かない場合は、demo/app.py の
  「音声を push して確定テキストを pop する箇所」に合わせて調整すること。
────────────────────────────────────────────────────────────

依存:
  pip install git+https://github.com/facebookresearch/seamless_communication.git
  pip install simuleval fairseq2
  # 初回実行時に seamless_streaming_unity / seamless_streaming_monotonic_decoder を自動DL
"""
import numpy as np
import torch

# ISO639-1 → Seamlessの3文字コード
_LANG3 = {"en": "eng", "fr": "fra", "zh": "cmn", "ko": "kor", "ja": "jpn"}


class SeamlessStreamingS2TT:
    SR = 16000

    def __init__(self, source, target, device=None):
        self.source = _LANG3[source]
        self.target = _LANG3[target]
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self._agent = None
        self._states = None
        self._build_agent()

    # ── 版依存ゾーン(ここを demo/app.py に合わせる) ──────────────
    def _build_agent(self):
        """SeamlessStreaming S2T エージェントを構築。
        公式CLI streaming_evaluate が内部で使うのと同じ:
          unity  : seamless_streaming_unity
          decoder: seamless_streaming_monotonic_decoder
        """
        from simuleval.utils.arguments import cli_argument_list  # noqa
        from seamless_communication.streaming.agents.seamless_streaming_s2t import (
            SeamlessStreamingS2TVADAgent,   # ← 版により名称が異なる場合あり
        )
        # SimulEvalのargparse経由でエージェントを組むのが公式流儀。
        # 下は典型値。tgt_lang や decision policy 閾値(--decision-threshold)で
        # 遅延⇔精度を調整できる(大=待つ=精度/語順◎・遅延↑)。
        args = [
            "--source-segment-size", "320",          # ms 単位の入力チャンク
            "--device", self.device,
            "--dtype", "fp16" if self.device != "cpu" else "fp32",
            "--tgt-lang", self.target,
            "--task", "s2tt",
        ]
        parser_ns = SeamlessStreamingS2TVADAgent.build_argument_parser().parse_args(args)
        self._agent = SeamlessStreamingS2TVADAgent.from_args(parser_ns)
        self._states = self._agent.build_states()

    def _make_segment(self, audio: np.ndarray, finished=False):
        from simuleval.data.segments import SpeechSegment
        return SpeechSegment(
            content=audio.astype("float32"),
            sample_rate=self.SR,
            finished=finished,
            tgt_lang=self.target,
        )

    def _pop_text(self, output_segment) -> str:
        """出力セグメントから確定テキストを取り出す。版差吸収用に分離。"""
        if output_segment is None:
            return ""
        content = getattr(output_segment, "content", None)
        if not content:
            return ""
        if isinstance(content, (list, tuple)):
            return " ".join(map(str, content)).strip()
        return str(content).strip()
    # ──────────────────────────────────────────────────────────

    def insert(self, audio: np.ndarray):
        """マイクから来た16k float32を push し、確定した訳文があれば返す。"""
        seg = self._make_segment(audio)
        out = self._agent.pushpop(seg, self._states)  # states管理は版依存。要なら self._agentのpush/popに分割
        return self._pop_text(out)

    def poll(self) -> str:
        """run_streaming のループ互換用。insert で都度返すのでここは空。"""
        return ""

    def finish(self) -> str:
        out = self._agent.pushpop(self._make_segment(np.empty(0, "float32"), finished=True),
                                  self._states)
        return self._pop_text(out)
