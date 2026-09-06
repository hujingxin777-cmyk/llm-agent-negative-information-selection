from __future__ import annotations

"""决策结果到论文指标的编码函数。"""

from typing import Iterable, Optional

from .schemas import DecisionOutput, EncodedMetrics


def encode_metrics(
    decision: DecisionOutput,
    previous_nci_comp: Optional[int] = None,
    group_negative_ratio: Optional[float] = None,
    negative_signal_share: Optional[float] = None,
) -> EncodedMetrics:
    cb = int(decision.credible_choice == "negative")
    sb = int(decision.share_choice == "negative")
    eb = int(decision.comment_choice == "support_negative")
    like_negative = int((decision.like_choice or "") == "negative")
    nsb = int(decision.final_choice == "negative")
    nci_comp = cb + sb + eb + like_negative
    scm = None if previous_nci_comp is None else abs(nci_comp - previous_nci_comp)
    gcr = None if group_negative_ratio is None else max(group_negative_ratio, 1 - group_negative_ratio)
    ssa = None if negative_signal_share is None else (negative_signal_share - 0.5) * 2
    return EncodedMetrics(
        nsb=nsb,
        cb=cb,
        sb=sb,
        eb=eb,
        nci_comp=nci_comp,
        scm=scm,
        gcr=gcr,
        ssa=ssa,
    )


def group_consensus_ratio(votes: Iterable[int]) -> float:
    """计算群体一致性比率（负向占比与正向占比的较大值）。"""
    votes = list(votes)
    if not votes:
        return 0.0
    negative_ratio = sum(votes) / len(votes)
    return max(negative_ratio, 1 - negative_ratio)
