"""意图理解与自然语言生成。

设计原则：**LLM 可选，链路必通**。
配置了 OPENAI_API_KEY / DASHSCOPE_API_KEY 时调用大模型做意图解析与报告生成；
未配置时自动使用离线规则版，全流程仍然完整可用（适合比赛现场断网演示）。
"""

from __future__ import annotations

import json
import re
import urllib.request

from .config import llm_settings


def call_llm(messages: list[dict], *, json_mode: bool = False, timeout: int = 60) -> str | None:
    """调用 OpenAI 兼容接口（DeepSeek / Qwen / 本地 vLLM 均可）。失败返回 None。"""

    cfg = llm_settings()
    if not cfg["api_key"]:
        return None
    base = (cfg["base_url"] or "https://api.openai.com/v1").rstrip("/")
    payload = {"model": cfg["model"], "messages": messages, "temperature": 0.2}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


INTENT_KEYWORDS = {
    "scan": ["扫描", "巡检", "有没有异常", "哪批", "异常批次", "最新情况"],
    "retrieve": ["手册", "依据", "sop", "规范", "标准", "怎么排查", "流程"],
    "explain": ["为什么", "原因", "解释", "哪些信号", "怎么判断"],
    "decision": ["停机", "要不要停", "继续生产", "划算", "成本", "交期"],
}


def parse_intent_rule(question: str, lot_ids: list[str], latest_lot: str) -> dict:
    q = question.strip()
    lot = None
    m = re.search(r"LOT[- ]?0*(\d{1,4})", q, flags=re.IGNORECASE)
    if m:
        candidate = f"LOT-{int(m.group(1)):04d}"
        lot = candidate if candidate in lot_ids else None

    intent = "diagnose"
    for name, words in INTENT_KEYWORDS.items():
        if any(w.lower() in q.lower() for w in words):
            intent = name
            break
    if intent != "scan" and not lot:
        lot = latest_lot
    return {"intent": intent, "lot_id": lot or "", "planner": "rule"}


def parse_intent_llm(question: str, lot_ids: list[str], latest_lot: str) -> dict | None:
    sys = (
        "你是一个制造业设备维保智能体的意图解析模块。"
        "请把用户问题解析为 JSON："
        '{"intent": "diagnose|scan|retrieve|explain|decision", "lot_id": "LOT-xxxx 或空字符串"}。'
        f"可选批次示例：{', '.join(lot_ids[:5])}；最新批次是 {latest_lot}。"
        "只输出 JSON。"
    )
    text = call_llm(
        [{"role": "system", "content": sys}, {"role": "user", "content": question}], json_mode=True
    )
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    lot = data.get("lot_id") or latest_lot
    if lot not in lot_ids:
        lot = latest_lot
    intent = data.get("intent")
    if intent not in INTENT_KEYWORDS and intent != "diagnose":
        intent = "diagnose"
    return {"intent": intent, "lot_id": lot, "planner": "llm"}


def parse_intent(question: str, lots_df) -> dict:
    lot_ids = lots_df["lot_id"].tolist()
    latest = str(lots_df.sort_values("timestamp").iloc[-1]["lot_id"])
    return parse_intent_llm(question, lot_ids, latest) or parse_intent_rule(question, lot_ids, latest)


def compose_report_rule(state: dict) -> str:
    risk = state.get("risk") or {}
    prop = state.get("proposal") or {}
    decision = state.get("decision") or {}
    expl = state.get("explanation") or {}
    scan = state.get("scan") or {}

    if scan and not risk:
        lines = [f"【批次巡检结论】最近 {scan.get('n_lots')} 个批次"]
        lines.append(f"其中高风险批次 {scan.get('n_high_risk')} 个。")
        for item in scan.get("high_risk_lots", []):
            lines.append(
                f"· {item['lot_id']}（{item['timestamp']}）风险 {item['risk']:.1%}"
            )
        if scan.get("suggested_lot"):
            lines.append(f"建议优先排查：{scan['suggested_lot']}")
        else:
            lines.append("当前窗口内没有超过触发阈值的批次，维持正常生产并继续监测。")
        return "\n".join(lines)

    lines = [f"【批次 {state.get('lot_id', '-')} 维保诊断结论】"]
    if risk:
        lines.append(
            f"风险概率：{risk.get('risk', 0):.1%}"
            f"（阈值 {risk.get('threshold', 0):.0%}，等级 {risk.get('level', '-')}）"
        )
    if expl:
        contribs = expl.get("contributions", [])
        positives = [c for c in contribs if c["shap_value"] > 0] or contribs
        top = "、".join(
            f"{c['feature']}({c['group']}) {c['shap_value']:+.2f}" for c in positives[:3]
        )
        lines.append(f"主要异常信号：{top}")
        lines.append(f"定位工序站：{expl.get('suspected_station', '-')}")
    if decision:
        best = next(
            (o for o in decision.get("options", []) if o["id"] == decision.get("recommended")), None
        )
        if best:
            lines.append(
                f"模拟决策：优选「{best['name']}」，期望代价 ¥{best['expected_cost']:,.0f}"
                f"（较最差方案节省 ¥{decision.get('saving_vs_worst', 0):,.0f}）"
            )
    if prop:
        lines.append(f"建议动作：{prop.get('action', '-')}")
        lines.append(
            f"优先级：{prop.get('priority', '-')}｜负责人：{prop.get('owner', '-')}"
            f"｜预计工时：{prop.get('repair_hours', '-')} 小时"
        )
        if prop.get("spare_parts"):
            lines.append(f"可能涉及备件：{prop['spare_parts']}")
        lines.append(f"决策依据：{prop.get('evidence_summary', '-')}")
    wo = state.get("work_order")
    if wo:
        lines.append(
            f"工单状态：{wo['code']} 已写入业务数据库，状态「{wo['status']}」，计划完成 {wo['due_at']}"
        )
    approval = state.get("approval")
    if approval and approval.get("approved") is False:
        lines.append("人工确认结果：工程师选择不自动派工，已记录并转人工处理。")
    lines.append("说明：风险概率为统计估计，决策结果为基于假设参数的模拟估算，不代表真实企业收益。")
    return "\n".join(lines)


def compose_report(state: dict) -> tuple[str, bool]:
    """返回 (报告文本, 是否使用了大模型)。"""

    context = {
        "lot_id": state.get("lot_id"),
        "risk": state.get("risk"),
        "explanation": state.get("explanation"),
        "evidence": state.get("evidence"),
        "decision": state.get("decision"),
        "proposal": state.get("proposal"),
        "work_order": (state.get("work_order") or {}).get("code") if state.get("work_order") else None,
        "approval": state.get("approval"),
    }
    sys = (
        "你是半导体制造企业的设备维保智能体。请基于给定的事实数据，"
        "用简洁、专业、可执行的中文生成一段维保结论，包含：风险判断、异常定位、"
        "推荐动作、决策依据、工单状态。不得编造未给出的数据，不要使用夸张表述。"
    )
    text = call_llm(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
    )
    if text:
        return text.strip(), True
    return compose_report_rule(state), False
