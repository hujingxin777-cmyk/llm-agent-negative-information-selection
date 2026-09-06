from __future__ import annotations

"""实验运行期核心数据结构定义。"""

from dataclasses import dataclass
from typing import Optional


ConditionName = str
ChatAtmosphere = str


@dataclass(frozen=True)
class Scenario:
    """单个议题场景：背景 + 两种竞争叙事。"""

    name: str
    background: str
    positive_narrative: str
    negative_narrative: str


@dataclass(frozen=True)
class Persona:
    """角色人设（用于系统提示词）。"""

    name: str
    profile: str


@dataclass(frozen=True)
class PlatformSpec:
    """平台条件元信息（是否动态、是否有回复链）。"""

    name: str
    label: str
    dynamic: bool
    has_reply_chain: bool


@dataclass(frozen=True)
class ExperimentCell:
    """实验单元（最小可运行组合）。"""

    experiment_family: str
    scenario_name: str
    condition_name: ConditionName
    persona_name: str
    chat_atmosphere: ChatAtmosphere
    repeat_index: int
    max_timestep: int


@dataclass(frozen=True)
class DecisionOutput:
    """模型决策输出（P1/P2 通用）。"""

    final_choice: str
    credible_choice: str
    share_choice: str
    comment_choice: str
    like_choice: Optional[str] = None
    reason: Optional[str] = None
    reason_signal: Optional[str] = None


@dataclass(frozen=True)
class P3DecisionOutput(DecisionOutput):
    """P3 额外输出：动作计划、公开评论与回复目标。"""

    planned_action: str = "observe"
    public_comment: Optional[str] = None
    reply_target: Optional[str] = None


@dataclass(frozen=True)
class DiscussionAgent:
    """线程参与者定义。"""

    agent_id: str
    persona_name: str
    activity_level: str
    base_activity_score: float = 0.5


@dataclass(frozen=True)
class ThreadEvent:
    """线程中的单条事件（发帖/回复/点赞/转发/观察）。"""

    timestep: int
    event_id: str
    actor_id: str
    action_type: str
    content: str
    reply_to: Optional[str] = None
    signal_polarity: Optional[str] = None
    salience_score: float = 0.0
    hyperedge_id: Optional[str] = None


@dataclass(frozen=True)
class HyperedgeSnapshot:
    """某时间步下单栋评论楼的聚合状态。"""

    hyperedge_id: str
    root_event_id: str
    root_actor_id: str
    root_content: str
    latest_event_id: str
    latest_timestep: int
    member_count: int
    event_count: int
    dynamic_positive_events: int
    dynamic_negative_events: int
    dynamic_neutral_events: int
    dynamic_negative_share: Optional[float]
    dynamic_hps: float
    dynamic_hcs: float
    dynamic_hvs: float


@dataclass(frozen=True)
class ThreadSnapshot:
    """某时间步对 agent 可见的线程快照。"""

    timestep: int
    visible_events: list[ThreadEvent]
    visible_context: str
    visible_hyperedges: list[HyperedgeSnapshot]
    visible_hyperedge_ids: list[str]
    visible_hyperedge_summary: str = ""
    dominant_hyperedge_id: Optional[str] = None
    negative_visibility_share: Optional[float] = None


@dataclass
class AgentDynamicState:
    """P3 agent 的局部动态状态。"""

    agent_id: str
    last_choice: str = "neutral"
    last_final_choice: str = "neutral"
    last_action: str = "observe"
    last_reason: Optional[str] = None
    last_public_comment: Optional[str] = None
    last_reply_target: Optional[str] = None
    perceived_majority: str = "unclear"
    risk_arousal: float = 0.5
    engagement_tendency: float = 0.5
    latent_neg_tendency: float = 0.5
    fatigue: float = 0.0
    notification_score: float = 0.0
    silence_steps: int = 0
    active_count: int = 0
    was_ever_active: bool = False
    participation_state: str = "dormant"
    cooldown_steps_remaining: int = 0
    visible_negative_pressure: float = 0.5
    dominant_hyperedge_id: Optional[str] = None
    recent_choice_history: list[str] | None = None
    recent_focus_history: list[str] | None = None
    last_visible_hyperedge_ids: list[str] | None = None
    participated_hyperedge_ids: list[str] | None = None


@dataclass(frozen=True)
class SimulationTraceRow:
    """模拟预览导出的轨迹行。"""

    experiment_family: str
    scenario_name: str
    condition_name: str
    persona_name: str
    chat_atmosphere: str
    repeat_index: int
    timestep: int
    event_id: str
    actor_id: str
    action_type: str
    reply_to: Optional[str]
    hyperedge_id: Optional[str]
    signal_polarity: Optional[str]
    content: str


@dataclass(frozen=True)
class EncodedMetrics:
    """由选择结果编码得到的指标集合。"""

    nsb: float
    cb: int
    sb: int
    eb: int
    nci_comp: int
    scm: Optional[float] = None
    gcr: Optional[float] = None
    ssa: Optional[float] = None


@dataclass(frozen=True)
class StaticRunRow:
    """P1/P2 单次调用落盘行。"""

    experiment_family: str
    scenario_name: str
    condition_name: str
    persona_name: str
    chat_atmosphere: str
    report_1_polarity: str
    report_2_polarity: str
    repeat_index: int
    timestep: int
    success: bool
    error_message: Optional[str]
    final_choice: str
    credible_choice: str
    share_choice: str
    comment_choice: str
    like_choice: Optional[str]
    reason: Optional[str]
    reason_signal: Optional[str]
    nsb: Optional[float]
    cb: Optional[int]
    sb: Optional[int]
    eb: Optional[int]
    nci_comp: Optional[int]
    raw_response: str


@dataclass(frozen=True)
class P3RunRow:
    """P3 每事件决策落盘行。"""

    experiment_family: str
    scenario_name: str
    condition_name: str
    persona_name: str
    agent_persona_name: str
    chat_atmosphere: str
    report_1_polarity: str
    report_2_polarity: str
    repeat_index: int
    timestep: int
    event_id: str
    actor_id: str
    action_type: str
    success: bool
    error_message: Optional[str]
    reply_to: Optional[str]
    hyperedge_id: Optional[str]
    visible_hyperedge_ids: str
    activation_score: Optional[float]
    signal_polarity: Optional[str]
    content: str
    final_choice: str
    credible_choice: str
    share_choice: str
    comment_choice: str
    like_choice: Optional[str]
    reason: Optional[str]
    reason_signal: Optional[str]
    nsb: Optional[int]
    cb: Optional[int]
    sb: Optional[int]
    eb: Optional[int]
    nci_comp: Optional[int]
    scm: Optional[float]
    gcr: Optional[float]
    ssa: Optional[float]
    is_stop_step: bool
    stop_reason: Optional[str]
    raw_response: str


@dataclass(frozen=True)
class P3AgentPanelRow:
    """P3 每时间步的 agent 状态面板。"""

    experiment_family: str
    scenario_name: str
    condition_name: str
    persona_name: str
    agent_persona_name: str
    chat_atmosphere: str
    repeat_index: int
    timestep: int
    agent_id: str
    is_active: bool
    participation_state: str
    activation_score: Optional[float]
    active_rank: Optional[int]
    visible_hyperedge_ids: str
    visible_hyperedge_count: int
    dominant_hyperedge_id: Optional[str]
    visible_negative_pressure: Optional[float]
    last_choice: str
    last_final_choice: str
    latent_neg_tendency: float
    perceived_majority: str
    risk_arousal: float
    engagement_tendency: float
    fatigue: float
    notification_score: float
    silence_steps: int
    active_count: int
    was_ever_active: bool
