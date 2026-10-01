"""LangGraph 智能体：主 Agent + 6 个工具 + 条件分支 + 人工确认 + 失败重试 + 状态持久化。

图结构（与答辩用的执行轨迹一一对应）：

  parse_intent ─┬─ scan_stream ──────────────────────────────────────────┐
                └─ fetch_sensor → predict_failure → [风险分支]           │
                                                     ├─ 低风险 → low_risk_report
                                                     └─ 高风险 → explain_failure
                                                                  → retrieve_manual
                                                                  → simulate_decision
                                                                  → propose
                                                                  → human_review（人工确认）
                                                                       ├─ 同意 → create_work_order
                                                                       └─ 拒绝 → handover_human
                                                                  → finalize
  每个工具节点后接一个 retry 守卫：工具失败 → retry 节点 → 回到原节点（最多 N 次）→ handle_error
"""

from __future__ import annotations

import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import planner
from .config import AgentParams
from .tools import Toolbox, evidence_text

try:  # LangGraph >= 0.2.57 的人机协同中断原语
    from langgraph.types import Command, interrupt  # noqa: F401

    HAS_INTERRUPT = True
except Exception:  # pragma: no cover - 兼容旧版本
    HAS_INTERRUPT = False


class AgentState(TypedDict, total=False):
    question: str
    intent: str
    planner: str
    lot_id: str
    scan: dict
    sensor: dict
    risk: dict
    explanation: dict
    evidence: dict
    decision: dict
    proposal: dict
    approval: dict
    work_order: dict
    report: str
    llm_used: bool
    stage: str
    last_error: str
    failed_node: str
    retry_target: str
    retry_counts: dict
    trace: list
    finished: bool


TOOL_NODES = [
    "fetch_sensor",
    "predict_failure",
    "explain_failure",
    "retrieve_manual",
    "simulate_decision",
    "create_work_order",
]


def build_graph(toolbox: Toolbox, params: AgentParams | None = None):
    """构建并编译 LangGraph。传入 Toolbox 后返回可调用的 compiled graph。"""

    params = params or toolbox.params
    max_retries = params.max_retries

    # ------------------------------------------------------------ 节点实现
    def parse_intent_node(state: AgentState) -> dict:
        parsed = planner.parse_intent(state["question"], toolbox.lots)
        return {
            "intent": parsed["intent"],
            "lot_id": parsed["lot_id"],
            "planner": parsed["planner"],
            "stage": "意图解析",
            "retry_counts": {},
            "last_error": "",
            "trace": list(toolbox.trace),
        }

    def scan_stream_node(state: AgentState) -> dict:
        result = toolbox.scan_recent(window=300)
        if result.get("n_high_risk"):
            result["suggested_lot"] = result["high_risk_lots"][0]["lot_id"]
        return {
            "scan": result,
            "lot_id": result.get("suggested_lot", state.get("lot_id", "")),
            "stage": "批次巡检",
            "trace": list(toolbox.trace),
        }

    def fetch_sensor_node(state: AgentState) -> dict:
        try:
            data = toolbox.get_sensor_data(state["lot_id"])
            return {"sensor": data, "last_error": "", "stage": "读取工艺信号",
                    "trace": list(toolbox.trace)}
        except Exception as exc:  # noqa: BLE001
            return {"last_error": str(exc), "failed_node": "fetch_sensor",
                    "trace": list(toolbox.trace)}

    def predict_node(state: AgentState) -> dict:
        try:
            data = toolbox.predict_failure(state["lot_id"])
            return {"risk": data, "last_error": "", "stage": "风险预测",
                    "trace": list(toolbox.trace)}
        except Exception as exc:  # noqa: BLE001
            return {"last_error": str(exc), "failed_node": "predict_failure",
                    "trace": list(toolbox.trace)}

    def explain_node(state: AgentState) -> dict:
        try:
            data = toolbox.explain_failure(state["lot_id"])
            return {"explanation": data, "last_error": "", "stage": "异常归因",
                    "trace": list(toolbox.trace)}
        except Exception as exc:  # noqa: BLE001
            return {"last_error": str(exc), "failed_node": "explain_failure",
                    "trace": list(toolbox.trace)}

    def retrieve_node(state: AgentState) -> dict:
        try:
            expl = state.get("explanation") or {}
            stations = [s["station"] for s in expl.get("station_ranking", [])][:2] or ["工序站"]
            high_risk = bool((state.get("risk") or {}).get("is_high"))
            queries = toolbox.kb.aspect_queries(stations, high_risk=high_risk)
            data = toolbox.search_maintenance_manual(queries)
            return {"evidence": data, "last_error": "", "stage": "检索维修依据",
                    "trace": list(toolbox.trace)}
        except Exception as exc:  # noqa: BLE001
            return {"last_error": str(exc), "failed_node": "retrieve_manual",
                    "trace": list(toolbox.trace)}

    def simulate_node(state: AgentState) -> dict:
        try:
            risk = (state.get("risk") or {}).get("risk")
            data = toolbox.simulate_maintenance_decision(state["lot_id"], risk=risk)
            return {"decision": data, "last_error": "", "stage": "模拟维保决策",
                    "trace": list(toolbox.trace)}
        except Exception as exc:  # noqa: BLE001
            return {"last_error": str(exc), "failed_node": "simulate_decision",
                    "trace": list(toolbox.trace)}

    def propose_node(state: AgentState) -> dict:
        risk = state.get("risk") or {}
        decision = state.get("decision") or {}
        expl = state.get("explanation") or {}
        ev = (state.get("evidence") or {}).get("evidence", [])
        stations = [s["station"] for s in expl.get("station_ranking", [])]
        station = expl.get("suspected_station") or (stations[0] if stations else "工序站A")
        best_id = decision.get("recommended", "A")
        best = next((o for o in decision.get("options", []) if o["id"] == best_id), None)
        best_name = best["name"] if best else ""

        if best_id == "A":
            action = f"立即停机检修：对{station}执行零点校验、泄漏检查与部件点检"
        elif best_id == "B":
            action = f"延时检修：{best_name}，其间加严抽检并准备备件"
        else:
            action = f"维持生产并加严抽检，若{station}风险继续上升则立即停机检修"

        p = float(risk.get("risk", 0.0))
        thr = float(risk.get("threshold", 0.5))
        priority = "高" if p >= max(0.6, thr) else ("中" if p >= thr else "低")

        spare_candidates = {
            "密封件": ["密封", "泄漏", "O 型圈", "门封"],
            "质量流量控制器(MFC)": ["MFC", "流量"],
            "射频匹配网络部件": ["射频", "匹配", "反射功率"],
            "静电吸盘(ESC)": ["静电吸盘", "ESC"],
            "温度传感器/热电偶": ["温度传感器", "热电偶", "零点校验"],
        }
        ev_text = " ".join(e["text"] for e in ev)
        spares = [name for name, kws in spare_candidates.items() if any(k in ev_text for k in kws)]

        proposal = {
            "lot_id": state["lot_id"],
            "station": station,
            "priority": priority,
            "action": action,
            "owner": f"{station}维护组",
            "spare_parts": "、".join(spares),
            "repair_hours": toolbox.business.repair_hours,
            "risk": round(p, 4),
            "evidence_summary": evidence_text(ev, k=2) or "预测模型输出",
            "options": decision.get("options", []),
        }
        return {"proposal": proposal, "last_error": "", "stage": "形成建议",
                "trace": list(toolbox.trace)}

    def low_risk_node(state: AgentState) -> dict:
        risk = state.get("risk") or {}
        proposal = {
            "lot_id": state["lot_id"],
            "station": "-",
            "priority": "低",
            "action": "维持正常生产，纳入趋势观察，下一班次复核",
            "owner": "-",
            "spare_parts": "",
            "repair_hours": 0.0,
            "risk": round(float(risk.get("risk", 0.0)), 4),
            "evidence_summary": "风险概率低于阈值，未触发工单",
            "options": [],
        }
        return {"proposal": proposal, "stage": "低风险放行", "trace": list(toolbox.trace)}

    def review_node(state: AgentState) -> dict:
        if state.get("approval"):
            return {}
        payload = {
            "type": "approval_request",
            "lot_id": state.get("lot_id"),
            "proposal": state.get("proposal"),
            "decision": state.get("decision"),
        }
        if HAS_INTERRUPT:
            decision = interrupt(payload)
            if not isinstance(decision, dict):
                decision = {"approved": bool(decision)}
            decision.setdefault("operator", "值班工程师")
            return {"approval": decision}
        return {"approval": {"approved": None, "operator": "待确认"}}

    def create_work_order_node(state: AgentState) -> dict:
        prop = state.get("proposal") or {}
        risk = state.get("risk") or {}
        try:
            data = toolbox.create_work_order(
                lot_id=state["lot_id"],
                station=prop.get("station", "-"),
                priority=prop.get("priority", "中"),
                action=prop.get("action", ""),
                owner=prop.get("owner", "设备维护组"),
                spare_parts=prop.get("spare_parts", ""),
                risk=risk.get("risk", 0.0),
                evidence=prop.get("evidence_summary", ""),
            )
            return {"work_order": data["work_order"], "last_error": "",
                    "stage": "生成工单", "trace": list(toolbox.trace)}
        except Exception as exc:  # noqa: BLE001
            return {"last_error": str(exc), "failed_node": "create_work_order",
                    "trace": list(toolbox.trace)}

    def handover_node(state: AgentState) -> dict:
        return {
            "stage": "人工接管",
            "proposal": {**(state.get("proposal") or {}),
                         "action": "工程师拒绝自动派工，转人工处理并记录原因"},
            "trace": list(toolbox.trace),
        }

    def retry_node(state: AgentState) -> dict:
        target = state.get("failed_node") or "fetch_sensor"
        counts = dict(state.get("retry_counts") or {})
        counts[target] = counts.get(target, 0) + 1
        return {
            "retry_target": target,
            "retry_counts": counts,
            "last_error": "",
            "stage": f"重试 {target}（第 {counts[target]} 次）",
            "trace": list(toolbox.trace),
        }

    def handle_error_node(state: AgentState) -> dict:
        return {
            "stage": "异常终止",
            "report": (
                "工具连续失败，已停止自动执行并转人工处理。"
                f"最后一次错误：{state.get('last_error')}"
            ),
            "finished": True,
            "trace": list(toolbox.trace),
        }

    def finalize_node(state: AgentState) -> dict:
        report, used_llm = planner.compose_report(state)
        approval = state.get("approval") or {}
        stage = "完成（人工接管）" if approval.get("approved") is False else "完成"
        return {"report": report, "llm_used": used_llm, "stage": stage, "finished": True,
                "trace": list(toolbox.trace)}

    # -------------------------------------------------------------- 路由函数
    def guard(node: str, ok_target: str):
        def route(state: AgentState) -> str:
            if not state.get("last_error"):
                return ok_target
            counts = state.get("retry_counts") or {}
            if counts.get(node, 0) < max_retries:
                return "retry"
            return "handle_error"

        return route

    def route_after_intent(state: AgentState) -> str:
        return "scan_stream" if state.get("intent") == "scan" else "fetch_sensor"

    def route_by_risk(state: AgentState) -> str:
        risk = state.get("risk") or {}
        return "explain_failure" if risk.get("is_high") else "low_risk_report"

    def route_retry(state: AgentState) -> str:
        return state.get("retry_target", "fetch_sensor")

    def route_approval(state: AgentState) -> str:
        approval = state.get("approval") or {}
        return "create_work_order" if approval.get("approved") else "handover_human"

    # ------------------------------------------------------------ 构图
    g = StateGraph(AgentState)
    g.add_node("parse_intent", parse_intent_node)
    g.add_node("scan_stream", scan_stream_node)
    g.add_node("route_risk", lambda state: {})
    g.add_node("fetch_sensor", fetch_sensor_node)
    g.add_node("predict_failure", predict_node)
    g.add_node("explain_failure", explain_node)
    g.add_node("retrieve_manual", retrieve_node)
    g.add_node("simulate_decision", simulate_node)
    g.add_node("propose", propose_node)
    g.add_node("low_risk_report", low_risk_node)
    g.add_node("human_review", review_node)
    g.add_node("create_work_order", create_work_order_node)
    g.add_node("handover_human", handover_node)
    g.add_node("retry", retry_node)
    g.add_node("handle_error", handle_error_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "parse_intent")
    g.add_conditional_edges(
        "parse_intent",
        route_after_intent,
        {"scan_stream": "scan_stream", "fetch_sensor": "fetch_sensor"},
    )
    g.add_edge("scan_stream", "finalize")

    for node, ok_target in [
        ("fetch_sensor", "predict_failure"),
        ("predict_failure", "route_risk"),
        ("explain_failure", "retrieve_manual"),
        ("retrieve_manual", "simulate_decision"),
        ("simulate_decision", "propose"),
        ("create_work_order", "finalize"),
    ]:
        mapping = {"retry": "retry", "handle_error": "handle_error", ok_target: ok_target}
        g.add_conditional_edges(node, guard(node, ok_target), mapping)

    g.add_conditional_edges(
        "route_risk",
        route_by_risk,
        {"explain_failure": "explain_failure", "low_risk_report": "low_risk_report"},
    )
    g.add_edge("low_risk_report", "finalize")
    g.add_edge("propose", "human_review")
    g.add_conditional_edges(
        "human_review",
        route_approval,
        {"create_work_order": "create_work_order", "handover_human": "handover_human"},
    )
    g.add_edge("handover_human", "finalize")
    g.add_conditional_edges("retry", route_retry, {n: n for n in TOOL_NODES})
    g.add_edge("handle_error", END)
    g.add_edge("finalize", END)

    return g.compile(checkpointer=MemorySaver())


def new_thread(toolbox: Toolbox) -> dict:
    toolbox.reset_trace()
    return {"configurable": {"thread_id": f"zhixi-{uuid.uuid4().hex[:8]}"}}


def invoke(toolbox: Toolbox, graph, question: str, config: dict | None = None) -> dict:
    config = config or new_thread(toolbox)
    return graph.invoke({"question": question, "trace": []}, config)


def resume(graph, config: dict, payload: dict) -> dict:
    """人工确认后继续执行（LangGraph 中断恢复）。"""

    if not HAS_INTERRUPT:
        raise RuntimeError("当前 langgraph 版本不支持 interrupt，请升级到 0.2.57 以上")
    return graph.invoke(Command(resume=payload), config)


def snapshot(graph, config: dict) -> dict:
    state = graph.get_state(config)
    values = dict(state.values) if state and state.values else {}
    return {"values": values, "next": list(state.next) if state else []}
