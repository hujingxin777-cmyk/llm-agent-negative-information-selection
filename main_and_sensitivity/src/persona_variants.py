from __future__ import annotations

"""Persona 版本集：支持 legacy 与修订版人设切换。"""

from copy import deepcopy
from typing import Any, Dict

LEGACY_PERSONAS = {
    "profiles": {
        "baseline_open": "",
        "risk_sensitive": "你对潜在健康风险较为敏感，更关注可能被忽视的隐患。",
        "authority_trusting": "你通常更相信专家和正式来源，会优先考虑权威证据。",
        "ordinary_user": "你是普通社交平台用户，不预设固定立场，通常会根据当前最显眼、最能打动的信息形成相对判断。",
    },
    "behavior_texts": {
        "baseline_open": "",
        "risk_sensitive": "你会更注意排队拥挤、恢复缓慢、风险被低估这类线索，面对不确定性时更容易提高警惕。",
        "authority_trusting": "你会更看重医院、疾控、专家和正式通报中的线索，不太愿意因零散个体经验立刻下重结论。",
        "ordinary_user": "你不会刻意坚持某一类信息来源，也不会先验偏向哪一边；你更像普通用户，会根据当前最显眼、最具体、最容易打动你的线索形成一个相对判断。",
    },
    "behavior_profiles": {
        "baseline_open": {
            "negative_risk_gain": 1.0,
            "positive_relief_gain": 1.0,
            "engagement_gain": 1.0,
            "observe_decay": 1.0,
            "majority_sensitivity": 1.0,
        },
        "risk_sensitive": {
            "negative_risk_gain": 1.35,
            "positive_relief_gain": 0.75,
            "engagement_gain": 1.10,
            "observe_decay": 0.85,
            "majority_sensitivity": 1.15,
        },
        "authority_trusting": {
            "negative_risk_gain": 0.85,
            "positive_relief_gain": 1.20,
            "engagement_gain": 0.90,
            "observe_decay": 1.00,
            "majority_sensitivity": 0.90,
        },
        "ordinary_user": {
            "negative_risk_gain": 1.00,
            "positive_relief_gain": 1.00,
            "engagement_gain": 0.75,
            "observe_decay": 1.20,
            "majority_sensitivity": 0.70,
        },
    },
}

REVISED_2026_04_PERSONAS = {
    "profiles": {
        "baseline_open": "你是普通社交平台用户，不预设固定立场，通常会根据当前最显眼、最能打动你的信息形成一个相对判断。",
        "main_baseline_neutral": "你是主实验中的中性普通用户，不预设风险或可控立场。你会关注具体风险经验，也会重视正式应对和恢复线索；少数个案不能自动代表整体，但反复出现的恢复慢、复查观察和症状拖延线索也不会被简单视为无关。",
        "risk_sensitive": "你对潜在健康风险较为敏感，会注意病程拖长、反复发热、交叉感染、资源压力、复诊增加和可能被忽视的隐患。风险敏感不等于默认相信坏消息；当风险线索具体、连续、相互印证时，你会提高警觉，看到具体的缓解证据时也会下调担忧。",
        "authority_trusting": "你通常更重视来源清楚、表述克制、证据连续的信息，会优先考虑医生、医院、学校、社区和正式来源中的可核验线索。权威信任不等于自动选择安抚叙事；如果风险经验多次独立出现，并且与可核验事实互相印证，你也会提高警觉。",
        "ordinary_user": "你是普通社交平台用户，没有特别固定的立场。你会参考当前最显眼、最具体、最贴近日常经验的信息，但也会保留一点观望，不会仅因讨论里某一边声音更密集就立刻推到极端判断。",
    },
    "behavior_texts": {
        "baseline_open": "你不会刻意坚持某一类信息来源，也不会先验偏向哪一边；你更像普通用户，会根据当前最显眼、最具体、最容易打动你的线索形成一个相对判断。",
        "main_baseline_neutral": "你是主实验的中性普通基线。你会比较风险线索与可控线索的具体性、代表性和一致性；少数具体个案会提高注意但不会自动代表整体；正式应对和恢复路径会降低担忧，但恢复慢、复诊观察、症状反复等重复线索也需要被纳入判断。",
        "risk_sensitive": "你会额外检查恢复慢、反复、交叉感染、复诊和资源压力等线索的具体性、连续性与代表性。具体风险会让你更谨慎，但如果正式应对、恢复路径或缓解线索同样具体，你也会保留可控判断。",
        "authority_trusting": "你会更看重信息来源是否清楚、证据是否连续、表述是否克制，而不是简单把个体经验或安抚说法当成结论。正式应对线索会缓冲担忧；多条独立风险经验与报道事实互相印证时，也会让你转向更谨慎。",
        "ordinary_user": "你没有明显的预设立场，会参考当前最显眼、最具体、最容易打动你的线索形成相对判断。你会参考别人怎么说，也会受讨论氛围影响，但需要看到多个具体且相互补充的线索，才会从观望转向强烈判断。",
    },
    "behavior_profiles": {
        "baseline_open": {
            "negative_risk_gain": 1.0,
            "positive_relief_gain": 1.0,
            "engagement_gain": 1.0,
            "observe_decay": 1.0,
            "majority_sensitivity": 1.0,
        },
        "main_baseline_neutral": {
            "negative_risk_gain": 1.07,
            "positive_relief_gain": 0.94,
            "engagement_gain": 1.00,
            "observe_decay": 1.00,
            "majority_sensitivity": 1.00,
        },
        "risk_sensitive": {
            "negative_risk_gain": 1.06,
            "positive_relief_gain": 0.96,
            "engagement_gain": 1.00,
            "observe_decay": 1.00,
            "majority_sensitivity": 1.00,
        },
        "authority_trusting": {
            "negative_risk_gain": 1.00,
            "positive_relief_gain": 1.00,
            "engagement_gain": 0.95,
            "observe_decay": 1.00,
            "majority_sensitivity": 0.92,
        },
        "ordinary_user": {
            "negative_risk_gain": 1.00,
            "positive_relief_gain": 1.00,
            "engagement_gain": 0.98,
            "observe_decay": 1.00,
            "majority_sensitivity": 0.98,
        },
    },
}

REVISED_2026_04_GENERAL_PUBLIC_PERSONAS = deepcopy(REVISED_2026_04_PERSONAS)
REVISED_2026_04_GENERAL_PUBLIC_PERSONAS["profiles"]["ordinary_user"] = REVISED_2026_04_PERSONAS["profiles"][
    "ordinary_user"
]
REVISED_2026_04_GENERAL_PUBLIC_PERSONAS["behavior_texts"]["ordinary_user"] = REVISED_2026_04_PERSONAS[
    "behavior_texts"
]["ordinary_user"]
REVISED_2026_04_GENERAL_PUBLIC_PERSONAS["behavior_profiles"]["ordinary_user"] = {
    "negative_risk_gain": 1.00,
    "positive_relief_gain": 1.00,
    "engagement_gain": 0.95,
    "observe_decay": 1.05,
    "majority_sensitivity": 0.82,
}
PERSONA_NAME_ALIASES = {
    "neutral_prudent": "ordinary_user",
}

PERSONA_VARIANTS = {
    "legacy": LEGACY_PERSONAS,
    "revised_2026_04": REVISED_2026_04_PERSONAS,
    "revised_2026_04_general_public": REVISED_2026_04_GENERAL_PUBLIC_PERSONAS,
}

_CURRENT_VARIANT = "legacy"


def set_persona_variant(variant_name: str | None) -> str:
    """设置当前运行使用的人设版本。"""
    global _CURRENT_VARIANT
    if not variant_name:
        _CURRENT_VARIANT = "legacy"
        return _CURRENT_VARIANT
    if variant_name not in PERSONA_VARIANTS:
        raise ValueError(f"未知 persona variant: {variant_name}")
    _CURRENT_VARIANT = variant_name
    return _CURRENT_VARIANT


def get_persona_variant() -> str:
    """返回当前激活的人设版本名。"""
    return _CURRENT_VARIANT


def get_persona_behavior_text(persona_name: str) -> str:
    """读取指定 persona 的行为文本。"""
    variant = PERSONA_VARIANTS.get(_CURRENT_VARIANT, LEGACY_PERSONAS)
    canonical_name = PERSONA_NAME_ALIASES.get(persona_name, persona_name)
    return variant["behavior_texts"].get(canonical_name, "")


def get_persona_behavior_profile(persona_name: str) -> dict[str, float]:
    """读取指定 persona 的行为参数画像。"""
    variant = PERSONA_VARIANTS.get(_CURRENT_VARIANT, LEGACY_PERSONAS)
    baseline = variant["behavior_profiles"]["baseline_open"]
    canonical_name = PERSONA_NAME_ALIASES.get(persona_name, persona_name)
    return deepcopy(variant["behavior_profiles"].get(canonical_name, baseline))


def apply_persona_variant_to_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """将所选 variant 的 profile 覆盖进 config.personas。"""
    variant_name = str(config.get("persona_runtime", {}).get("variant", _CURRENT_VARIANT))
    variant = PERSONA_VARIANTS.get(variant_name)
    if not variant:
        return config

    merged = deepcopy(config)
    existing = {item["name"]: item for item in merged.get("personas", [])}
    for persona_name, profile in variant["profiles"].items():
        item = dict(existing.get(persona_name, {"name": persona_name}))
        item["profile"] = profile
        existing[persona_name] = item

    ordered_names = list(variant["profiles"].keys())
    merged["personas"] = [existing[name] for name in ordered_names]
    return merged
