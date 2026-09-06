from __future__ import annotations

"""Prompt 组装：P1/P2 静态提示词与 P3 时间步提示词。"""

import re
import hashlib
import random
from typing import Any, Dict

from .environments import get_social_block

_SOCIAL_LABEL_PATTERN = re.compile(r"^\s*【([^】]+)】\s*")
_SOCIAL_LIKE_PATTERN = re.compile(r"👍\s*\d+\s*")


def build_system_message(persona_profile: str) -> str:
    """构造 system 提示词（普通用户视角 + 人设附加）。"""
    base = (
        "你正在参与一项关于信息选择的研究模拟。\n"
        "你扮演的是普通社交平台用户，不是专家、记者或事实核查员。\n"
        "请基于给定材料做有限信息下的快速判断，不要解释实验设计本身。\n"
        "你的任务是比较两种相互竞争的叙事，并做出相对选择，而不是等待绝对确定性。\n"
        "请像真实用户那样反应，不要写成助手式、百科式或过度平衡的分析。\n"
        "普通用户可能会被提示中给出的具体经历、风险提醒、正式信息或可见评论打动，不预设哪一类天然更重要。\n"
        "你没有固定立场，也不默认更相信官方安抚说法或更相信风险提醒说法。\n"
        "不要自行补充材料外的个人经历、朋友圈说法、新闻背景或现实记忆。\n"
        "请不要为了显得谨慎而机械保持观望；如果两边都说得通，请仍然选出你相对更接近的一边。"
    )
    if persona_profile and persona_profile.strip():
        return base + "\n" + f"你的个性设定：{persona_profile}"
    return base


def _choice_instruction_block(condition_name: str) -> str:
    """通用选择规则块，保证三平台口径一致。"""
    common = (
        "判断规则：\n"
        "1. 这是有限信息下的相对判断任务，不要求你百分之百确定。\n"
        "2. 如果两种叙事都各有理由，请根据你此刻更愿意采纳、转述或公开表态的一边做选择。\n"
        "3. 只有在你认为两种叙事几乎无法区分、没有明显相对优势时，credible_choice 才可使用 neutral。\n"
        "4. share_choice 和 like_choice 只有在你明确不愿传播任一叙事时才可填 none。\n"
        "5. like_choice 表示你会点赞或顶起哪一边的观点，不等于情绪积极；你也可以点赞一条提醒风险、表达担忧或说现实压力更大的评论。\n"
        "6. comment_choice 应反映你公开表达时更偏向支持哪一边；只有你会明确表达观望时才填 neutral。\n"
        "7. 真实用户会同时参考材料中的具体经历、正式来源、风险提醒和安抚性信号，不要因为一边更像官方表述或更像风险提示，就机械默认选它。\n"
        "8. 报道1和报道2只是展示顺序标签，先出现不代表更权威、更可信或更推荐；每次都要比较两篇内容本身。\n"
        "9. 如果你的人设对风险更敏感，应理解为更仔细检查风险成本，而不是默认选择负向；"
        "具体缓解证据会降低担忧，但不会自动抵消恢复慢、复诊、夜间反复或资源压力等多点风险线索。\n"
        "10. 严禁使用材料外经验作为理由，包括“我身边的人/我看到的现实情况/朋友圈/新闻里/网上还有人说”等未给出的信息；只能引用报道、给定评论和当前可见线程。"
    )
    if condition_name == "independent":
        return (
            common
            + "\n11. P1 是独立材料判断，只能使用背景和两篇报道；不要引用或想象身边朋友、朋友圈、网友评论、学校讨论、门诊见闻等材料外信息。"
        )
    if condition_name == "low_feedback":
        return (
            common
            + "\n11. P2 可以受给定评论和点赞影响，但只能使用本提示中列出的评论；不要自行补充其他网友、朋友圈或现实经历。"
            + "\n12. P2 应先基于报道材料形成基础判断，再把可见评论和点赞作为增量线索；评论点赞数接近时，只说明没有压倒性多数。"
            + "\n13. 评论材料不是事实结论；请比较评论内容是否具体、是否与报道材料互相印证，而不是只看评论数量。"
            + "\n14. 如果你的人设更信任权威或正式来源，匿名个案通常只是增量线索；但当个案评论与报道里的明确事实相互印证时，也可以提高警觉。"
        )
    return (
        common
        + "\n11. P3 可以受当前公开线程影响，但只能使用本提示中给出的报道、初始只读评论和当前可见线程；不要自行补充外部经历。"
        + "\n12. P3 的公开线程会影响你，但不是投票命令；如果你的人设、报道材料和可见线程之间存在张力，可以保留不同方向的判断。"
        + "\n13. P3 是高阶互动，不是对初始评论的复读或投票；当初始只读评论和最近线程明显偏向同一边时，也要检查另一边是否存在具体、可核验、可讨论的线索。"
        + "\n14. 具体风险线索和具体可控线索都需要被认真比较：不要把退热后咳嗽拖久、复诊、班级请假、夜间反复、门诊压力等笼统写成“模糊提醒”，也不要把规范治疗、学校处置、分诊随访、多数恢复等笼统写成“空泛安抚”；如果某一边不足以改变判断，请说明不足在哪里。"
    )


def _static_persona_softening_block(
    condition_name: str,
    persona_name: str,
    seed: str | None = None,
    chat_atmosphere: str = "none",
) -> str:
    """P1/P2-only persona reminder. It describes tendencies without forcing outcomes."""
    if condition_name not in {"independent", "low_feedback"}:
        return ""
    if persona_name == "risk_sensitive":
        return (
            "静态阶段人格提醒：风险敏感型会更早注意潜在风险成本，但不是默认选择风险叙事。"
            "请比较风险线索与可控线索的具体性、连续性和代表性；"
            "具体缓解证据会降低担忧，但不会自动抵消恢复慢、复诊、夜间反复、传播或资源压力等多点风险线索。"
        )
    if persona_name == "authority_trusting":
        return (
            "静态阶段人格提醒：权威信任型表示你更重视医生、医院、学校、社区等正式应对和可核验来源。"
            "普通评论中的恢复慢、反复或门诊压力可以作为提醒，但不能单独压过明确的治疗路径、学校处置或常规恢复线索。"
            "只有当风险线索来自多个独立来源，并且与正式/可核验事实相互印证时，你才会提高风险警觉。"
        )
    if persona_name == "main_baseline_neutral":
        return (
            "静态阶段人格提醒：你是主实验的中性普通基线，不预设风险或可控立场。"
            "请比较两边线索的具体性、代表性和一致性；少数个案可以作为提醒，但不能自动外推为整体趋势。"
            "正式应对、常规治疗、学校处置和恢复路径是可控线索，但不应单独替代对风险线索的判断；"
            "恢复慢、请假变化、复诊观察、咳嗽拖延或症状反复等线索如果更具体、更连续，也应被纳入判断。"
            "如果评论只是重复报道中已经出现的低强度观察或常规随访提醒，不要把它重复计权成新的强风险证据。"
        )
    if persona_name in {"ordinary_user", "baseline_open", "neutral_prudent"}:
        return (
            "静态阶段人格提醒：普通用户会受显眼线索影响，但不会只由单条评论或单个措辞决定。"
            "如果两边都有具体线索，请选择你此刻更接近的一边，而不是把所有字段都推到同一极端。"
            "普通用户不会天然认为正式可控线索一定更重要；退热后恢复慢、请假变化、复诊安排、"
            "夜间反复等贴近日常的风险经验，如果和报道互相印证，也会影响你的判断。"
        )
    return ""


def _render_reports(
    scenario: Dict[str, Any],
    report_mapping: Dict[str, str],
) -> tuple[str, str]:
    """按 report mapping 把叙事文本映射到“报道1/报道2”。"""
    report_1_key = report_mapping["report_1"]
    report_2_key = report_mapping["report_2"]
    narrative_map = {
        "positive": scenario["positive_narrative"],
        "negative": scenario["negative_narrative"],
    }
    return narrative_map[report_1_key], narrative_map[report_2_key]


def _render_p3_static_reference(social_block: str) -> str:
    """Render P2 social text without like counts for optional non-thread context."""
    lines: list[str] = []
    for raw_line in (social_block or "").splitlines():
        text = raw_line.strip()
        if not text:
            continue
        text = _SOCIAL_LABEL_PATTERN.sub("", text)
        text = _SOCIAL_LIKE_PATTERN.sub("", text).strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines) if lines else "无"


def _shuffle_social_block(social_block: str, seed: str | None, base_seed: int = 42) -> str:
    """Deterministically shuffle visible comments and relabel them 评论1..n."""
    lines = [line.strip() for line in (social_block or "").splitlines() if line.strip()]
    if not seed or len(lines) <= 1:
        return social_block
    rng_seed = int(hashlib.sha256(f"{base_seed}|{seed}".encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(rng_seed)
    shuffled = list(lines)
    rng.shuffle(shuffled)
    relabeled: list[str] = []
    for index, line in enumerate(shuffled, start=1):
        line = _SOCIAL_LABEL_PATTERN.sub("", line).strip()
        relabeled.append(f"【评论{index}】{line}")
    return "\n".join(relabeled)


def _few_shot_examples(condition_name: str) -> str:
    """按平台返回 few-shot 输出示例。"""
    independent_examples = (
        "输出示例（只展示 JSON 格式，不暗示报道编号、正负方向或你应该选择哪一边）：\n"
        "示例1：\n"
        '{\n'
        '  "final_choice": "report_1",\n'
        '  "credible_choice": "report_1",\n'
        '  "share_choice": "none",\n'
        '  "comment_choice": "neutral",\n'
        '  "like_choice": "none",\n'
        '  "reason": "只看当前材料，报道1的具体线索比报道2更能解释背景里的情况。"\n'
        '}\n'
        "示例2：\n"
        '{\n'
        '  "final_choice": "report_2",\n'
        '  "credible_choice": "report_2",\n'
        '  "share_choice": "none",\n'
        '  "comment_choice": "neutral",\n'
        '  "like_choice": "none",\n'
        '  "reason": "在不补充外部经验的情况下，报道2给出的材料证据更具体。"\n'
        '}'
    )
    static_examples = (
        "输出示例（只展示 JSON 格式，不暗示报道编号、正负方向或你应该选择哪一边）：\n"
        "示例1：\n"
        '{\n'
        '  "final_choice": "report_1",\n'
        '  "credible_choice": "report_1",\n'
        '  "share_choice": "report_1",\n'
        '  "comment_choice": "support_report_1",\n'
        '  "like_choice": "report_1",\n'
        '  "reason": "报道1里更具体、更连贯的线索让我此刻更愿意相信这一边。"\n'
        '}\n'
        "示例2：\n"
        '{\n'
        '  "final_choice": "report_2",\n'
        '  "credible_choice": "report_2",\n'
        '  "share_choice": "report_2",\n'
        '  "comment_choice": "support_report_2",\n'
        '  "like_choice": "report_2",\n'
        '  "reason": "报道2和当前可见评论里的线索更能解释我看到的材料。"\n'
        '}'
    )
    p3_examples = (
        "输出示例（只展示 JSON 格式和行动类型，不暗示报道编号、正负方向或你应该选择哪一边）：\n"
        "示例1：\n"
        '{\n'
        '  "planned_action": "post_comment",\n'
        '  "public_comment": "我觉得报道1里这组线索更具体一点。",\n'
        '  "reply_target": "",\n'
        '  "final_choice": "report_1",\n'
        '  "credible_choice": "report_1",\n'
        '  "share_choice": "report_1",\n'
        '  "comment_choice": "support_report_1",\n'
        '  "like_choice": "report_1",\n'
        '  "reason": "我主要是被报道1和最近讨论里更具体的线索推向这一边。"\n'
        '}\n'
        "示例2：\n"
        '{\n'
        '  "planned_action": "reply",\n'
        '  "public_comment": "前面有人提到恢复节奏，我更在意的是这种情况会不会反复。",\n'
        '  "reply_target": "t2_e1",\n'
        '  "final_choice": "report_1",\n'
        '  "credible_choice": "report_1",\n'
        '  "share_choice": "none",\n'
        '  "comment_choice": "support_report_1",\n'
        '  "like_choice": "report_1",\n'
        '  "reason": "已有楼层提到与我判断相关的具体点，我直接回复补充，而不是重复新建楼层。"\n'
        '}\n'
        "示例3：\n"
        '{\n'
        '  "planned_action": "like",\n'
        '  "public_comment": "",\n'
        '  "reply_target": "t2_e1",\n'
        '  "final_choice": "report_2",\n'
        '  "credible_choice": "report_2",\n'
        '  "share_choice": "none",\n'
        '  "comment_choice": "neutral",\n'
        '  "like_choice": "report_2",\n'
        '  "reason": "我只是轻度认同这条评论对报道2的解释，暂时没有新的补充。"\n'
        '}\n'
        "示例4：\n"
        '{\n'
        '  "planned_action": "observe",\n'
        '  "public_comment": "",\n'
        '  "reply_target": "",\n'
        '  "final_choice": "report_1",\n'
        '  "credible_choice": "report_1",\n'
        '  "share_choice": "none",\n'
        '  "comment_choice": "neutral",\n'
        '  "like_choice": "report_1",\n'
        '  "reason": "虽然讨论很热，但我目前更相信报道1里更清楚的那组线索。"\n'
        '}'
    )
    if condition_name == "high_interaction":
        return p3_examples
    if condition_name == "independent":
        return independent_examples
    return static_examples


def _reason_focus_hint(seed_index: int) -> str:
    """为重复轮次提供理由角度扰动，减少模板化复读。"""
    hints = [
        "你这次尽量比较“恢复较快”和“恢复拖长”哪一种描述更能说服你。",
        "你这次尽量比较“医院应对能力”和“现实就医压力”哪一边更能说服你。",
        "你这次尽量比较“材料里哪类具体线索更充分”而不是只复述抽象判断。",
        "你这次尽量比较“风险提醒”和“缓解证据”哪一种在材料中更具体。",
        "你这次尽量比较“正式来源、个体案例和材料措辞”哪一类线索更能影响你。",
    ]
    return hints[seed_index % len(hints)]


def build_user_prompt(
    config: Dict[str, Any],
    scenario: Dict[str, Any],
    condition_name: str,
    chat_atmosphere: str,
    report_mapping: Dict[str, str],
    persona_behavior_text: str = "",
    persona_name: str = "",
    reason_focus_hint: str = "",
    social_shuffle_seed: str | None = None,
) -> str:
    """构建 P1/P2 用户提示词。"""
    social_block = get_social_block(config, condition_name, chat_atmosphere, scenario.get("name"))
    social_block = _shuffle_social_block(social_block, social_shuffle_seed)
    report_1_text, report_2_text = _render_reports(scenario, report_mapping)
    if condition_name == "independent":
        task_frame = (
            "请把自己当作普通用户第一次刷到这组信息。"
            "你不会长时间查证，而是会根据直观印象做一次快速相对判断。"
            "如果需要立刻给出一个更接近的判断，请选择你此刻更偏向的一边。"
            "这一步没有评论、点赞、朋友圈或额外现实经验，不能自行补充材料外线索。"
        )
    elif condition_name == "low_feedback":
        task_frame = (
            "请把自己当作正在刷社交平台的普通用户。"
            "你会先根据报道内容形成基础印象，再受到显眼评论/点赞线索影响，并很快形成一个偏向。"
            "即使心里仍有保留，你通常也会形成一个更接近的倾向，而不是长期悬置。"
            "这一步只能使用当前给出的评论和点赞线索，不能自行补充其他网友或朋友圈经验。"
        )
    else:
        task_frame = "请把自己当作刷到这组内容的普通用户，并在有限注意力下做出一次实际选择。"

    parts = [
        "任务：\n" + task_frame,
        "背景信息：\n" + scenario["background"],
        "报道1：\n" + report_1_text,
        "报道2：\n" + report_2_text,
        _choice_instruction_block(condition_name),
    ]
    if persona_behavior_text:
        parts.append("你的反应方式：\n" + persona_behavior_text)
    softening_block = _static_persona_softening_block(
        condition_name,
        persona_name,
        social_shuffle_seed,
        chat_atmosphere,
    )
    if softening_block:
        parts.append(softening_block)
    if social_block:
        parts.append("可见社交材料：\n" + social_block)
        parts.append(
            "补充说明：这些评论和点赞只是你此刻刷到的少量可见内容，不代表平台整体多数或事实结论。"
            "它们不会继续更新，也没有回复链；请把它们作为报道材料之外的增量线索，而不是替代报道材料。"
        )
    if reason_focus_hint:
        parts.append("本次理由组织提示：\n" + reason_focus_hint)
    parts.append(
        "非重复要求：\n"
        "1. 不要总是重复同一类理由模板，例如反复只说“整体可控”或“风险可能更高”。\n"
        "2. 如果你连续多次做同方向选择，也请尽量换一个你最看重的证据角度。\n"
        "3. `reason` 要简短自然，但应尽量点出你这一次最关键的线索。\n"
        "4. `reason` 不能写本提示没有给出的个人经历、朋友经历、朋友圈内容或现实新闻。"
    )
    parts.append(_few_shot_examples(condition_name))
    parts.append(
        "额外要求：请像真实用户那样给出一个最终偏向。你不需要绝对确信，只需要给出更接近的一边。"
        "如果两边接近，请按你此刻更愿意相信或转述的一边作答，而不是把全部字段都填成中立或 none。"
        "不要因为某一边语气更平静或更吓人，就直接默认它更可信。\n"
        "\n请只输出一个 JSON 对象，不要输出额外解释。\n"
        "JSON 字段如下：\n"
        '{\n'
        '  "final_choice": "report_1|report_2",\n'
        '  "credible_choice": "report_1|neutral|report_2",\n'
        '  "share_choice": "report_1|none|report_2",\n'
        '  "comment_choice": "support_report_1|neutral|support_report_2",\n'
        '  "like_choice": "report_1|none|report_2",\n'
        '  "reason": "一句话说明你此刻为什么更偏向这一边，尽量引用你真正看到的线索"\n'
        '}'
    )
    return "\n\n".join(parts)


def build_p3_timestep_prompt(
    config: Dict[str, Any],
    scenario: Dict[str, Any],
    persona_profile: str,
    persona_behavior_text: str,
    chat_atmosphere: str,
    report_mapping: Dict[str, str],
    timestep: int,
    visible_context: str,
    thread_summary: str,
    agent_state_text: str,
    reason_focus_hint: str = "",
    social_shuffle_seed: str | None = None,
    agent_persona_name: str = "",
) -> str:
    """构建 P3 单时间步提示词（含可见上下文与动作约束）。"""
    social_block = get_social_block(config, "high_interaction", chat_atmosphere, scenario.get("name"))
    social_block = _shuffle_social_block(social_block, social_shuffle_seed)
    static_reference_cfg = config.get("p3", {}).get("static_reference", {})
    include_static_reference = bool(static_reference_cfg.get("enabled", True))
    static_reference_block = _render_p3_static_reference(social_block) if include_static_reference else ""
    report_1_text, report_2_text = _render_reports(scenario, report_mapping)
    parts = [
        "你正在参与一个社交平台上的话题讨论线程，形式接近论坛/推特话题区。",
    ]
    if persona_profile:
        parts.append(f"你的个性设定：{persona_profile}")
    if persona_behavior_text:
        parts.append(f"你的反应方式：{persona_behavior_text}")
    if agent_persona_name == "risk_sensitive":
        parts.append(
            "互动中的人格提醒：风险敏感型更容易注意恢复慢、复诊、症状反复、班级传播、"
            "家庭照护压力或门诊压力等线索。可控信息可以降低担忧，反复出现且相互印证的"
            "风险经验也会影响判断。"
        )
    if agent_persona_name == "authority_trusting":
        parts.append(
            "互动中的人格提醒：权威信任型更重视医生、医院、学校、社区和正式应对信息，"
            "通常会把匿名个案视为增量线索。若多个独立用户反复提到恢复慢、症状反复、"
            "复诊、就医压力或家庭照护压力，并且这些经验与报道中的可核验事实相互印证，"
            "也可以提高风险警觉。"
        )
    if agent_persona_name == "ordinary_user":
        parts.append(
            "互动中的人格提醒：普通用户会参考当前讨论中更具体、更贴近日常经验的线索，"
            "但通常不会只因为单条风险提醒或同一方向的评论数量变多就直接形成极端判断。"
            "若风险与可控线索都能看到，请比较它们是否来自多个独立来源、是否相互印证，"
            "可以保留观望、轻度提醒或中等强度倾向。"
        )
    if reason_focus_hint:
        parts.append(f"本次理由组织提示：{reason_focus_hint}")
    parts.extend(
        [
            f"当前时间步：{timestep}",
            f"背景信息：{scenario['background']}",
            f"报道1：{report_1_text}",
            f"报道2：{report_2_text}",
            "你的当前状态：\n" + agent_state_text,
            (
                "补充参考材料（来自 P2 可见社交材料，已去掉点赞热度）：\n"
                + static_reference_block
                if include_static_reference
                else "P2 评论已作为 P3 初始可见评论出现在当前线程中；这些 seed 初始评论只供阅读，不作为可回复、点赞或转发的互动目标。"
            ),
            "到目前为止的讨论摘要：\n" + thread_summary,
            "当前对你可见的最新讨论：\n" + visible_context,
            _choice_instruction_block("high_interaction"),
            "补充要求：\n"
            "1. 你是普通用户，不是专家或事实核查员。\n"
            "2. 你只能基于当前看到的有限上下文来反应，不要假设自己知道全部历史。\n"
            "3. 你的公开表达可以受最近讨论内容、上一轮自己的判断和多数意见感知影响。\n"
            "4. 行动方式说明：post_comment 表示单独发表一条新评论并建立新楼层；reply 表示回复当前 P3 公开线程里已出现的 agent 评论；like 表示点赞/认同/顶起当前 P3 公开线程里的某条观点但不额外发言；repost 表示转发/放大当前 P3 公开线程里的其中一边；observe 表示暂时不公开发言，但你心里仍然会形成一个相对倾向。\n"
            "5. 如果你选择 observe，请不要写公开评论，但仍然要如实填写你这一刻更接近哪一边，以及为什么。\n"
            "6. 如果你只是轻度认同某条评论、但没有新的补充角度，可以选择 like；认同负向风险提醒时也可以 like。\n"
            "7. 如果当前已有 agent 发言与你的判断相关，优先考虑 reply、like 或 observe；只有你确实要提出一个尚未出现的新角度时，才用 post_comment 新建楼层。\n"
            "8. 如果你只是读完后形成了心里判断，但还不想公开参与，更自然的动作是 observe；真实论坛里多数用户会先看、少数用户才发言。\n"
            "9. 如果你选择发帖或回复，请写一句简短、自然、像社交平台用户会说的话，不要写成正式说明。\n"
            "10. 不要机械重复最近两三条评论的原句；如果你与他们立场一致，也请补充一个新的理由、细节或观察角度。\n"
            "11. 如果你发现自己只能重复别人刚说过的话，可以选择 like 或 observe；但如果能说出新的具体线索，就不要只点赞。\n"
            "12. 不要把自己的理由写成抽象空话，尽量点出这一步真正影响你的线索。\n"
            "13. reply_target 只能填写“当前对你可见的最新讨论”里 id=... 对应的 agent 生成 event_id；seed_1、seed_2 这类初始评论只读，不能作为 reply/like/repost 目标。\n"
            "14. 如果当前可见内容主要是 seed 初始评论，你需要用 post_comment 生成自己的新评论楼层；如果你是在回应 agent 已生成的评论，请选择 reply。\n"
            "15. 公开发言应使用自然论坛语言，例如“前面有人提到……”“我看到有人说……”“有些家长反馈……”，不要机械复述编号。\n"
            "16. 如果当前可见线程已经连续多条都在重复可控/安心判断，而你仍看到报道或 seed 中有恢复慢、复诊、反复、班级请假或照护压力等具体线索，可以用新楼层提出这个风险疑问；这不等于编造外部经历，而是把已给材料中的另一边带入讨论。\n"
            "17. 如果当前可见线程已经连续多条都在重复风险判断，而你仍看到正式应对、治疗路径、学校处置或多数恢复等具体线索，也可以提出可控线索。\n"
            "18. 你的“当前潜在负向倾向”和“当前风险唤起水平”是进入本轮讨论前形成的私有状态；首轮建楼时应优先表达这个私有状态。若潜在负向倾向低于0.46，不要只因为某条 seed 风险评论就直接转成强负向；若在0.46到0.54之间，可以用观望、轻度提醒或轻度可控的方式发新楼层。",
            "19. 第一轮可以自主建立新楼层。从第二轮开始，如果已有 agent 楼层可以承接你的补充、质疑、支持或经验对照，应优先 reply；不要为了重复相似观点继续并列建楼。reply 只能指向 agent 生成的动态楼层，不能指向 seed。",
            _few_shot_examples("high_interaction"),
            "请只输出一个 JSON 对象，不要输出额外解释。\n"
            "JSON 字段如下：\n"
            '{\n'
            '  "planned_action": "post_comment|reply|like|repost|observe",\n'
            '  "public_comment": "如果发帖或回复，请写一句简短评论；否则可留空",\n'
            '  "reply_target": "如果 planned_action=reply/like/repost，优先填写你在可见上下文中看到的 id=... 对应 event_id；否则可留空",\n'
            '  "final_choice": "report_1|report_2",\n'
            '  "credible_choice": "report_1|neutral|report_2",\n'
            '  "share_choice": "report_1|none|report_2",\n'
            '  "comment_choice": "support_report_1|neutral|support_report_2",\n'
            '  "like_choice": "report_1|none|report_2",\n'
            '  "reason": "一句话说明你在这一时间步为什么这样判断或行动，尽量引用你真正看到的线索"\n'
            '}',
        ]
    )
    return "\n\n".join(parts)


def build_prompt_preview(
    config: Dict[str, Any],
    scenario_name: str,
    condition_name: str,
    persona_name: str,
    chat_atmosphere: str,
    report_mapping: Dict[str, str],
    repeat_index: int = 1,
) -> str:
    """生成 system + user 的预览文本（便于人工审阅）。"""
    scenarios = {item["name"]: item for item in config["scenarios"]}
    personas = {item["name"]: item for item in config["personas"]}
    scenario = scenarios[scenario_name]
    persona = personas[persona_name]
    return (
        "=== SYSTEM ===\n"
        + build_system_message(persona["profile"])
        + "\n\n=== USER ===\n"
        + build_user_prompt(
            config,
            scenario,
            condition_name,
            chat_atmosphere,
            report_mapping,
            persona_behavior_text="",
            persona_name=persona_name,
            reason_focus_hint=_reason_focus_hint(repeat_index - 1),
            social_shuffle_seed=f"preview|{scenario_name}|{condition_name}|{persona_name}|{chat_atmosphere}|{repeat_index}",
        )
    )
