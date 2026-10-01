"""智能体工具箱（6 个真实可调用的 Python 工具）。

每个工具都是普通的 Python 函数，通过 LangGraph 节点被真实调用，
调用过程（参数、结果、耗时、重试次数）全部写入执行轨迹，
答辩现场可以直接看到"智能体真的在调工具"，而不是大模型把答案说出来。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import db
from .config import BusinessParams, DB_PATH, AgentParams
from .data import RAW_DIR
from .explain import Contribution, Explainer
from .model import RiskModel, load_lots
from .rag import KnowledgeBase, strip_markdown

TOOL_DOCS = {
    "get_sensor_data": "查询指定批次最近一次工序信号快照，并计算相对训练分布的标准分偏差",
    "predict_failure": "调用工业风险预测模型，输出该批次出现质量异常的概率",
    "explain_failure": "调用可解释性模块，说明是哪些信号把风险推高",
    "search_maintenance_manual": "检索维修知识库（设备维护手册 / 点检 SOP / 安全规范）",
    "simulate_maintenance_decision": "模拟“立即检修 / 延时检修 / 继续生产”三种方案的期望代价",
    "create_work_order": "在业务数据库中真实新增一条维修工单",
}


class DemoFault(Exception):
    """用于演示"工具失败—自动重试"链路的可控故障。"""


@dataclass
class TraceStep:
    index: int
    node: str
    tool: str
    tool_doc: str
    args: dict
    ok: bool
    output: dict
    summary: str
    elapsed_ms: int
    attempt: int
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Toolbox:
    """把模型、数据、知识库、数据库封装成智能体可调用的工具集合。"""

    model: RiskModel
    X_raw: pd.DataFrame
    X_proc: pd.DataFrame
    y: pd.Series
    ts: pd.Series
    lots: pd.DataFrame
    kb: KnowledgeBase
    explainer: Explainer
    business: BusinessParams = field(default_factory=BusinessParams)
    params: AgentParams = field(default_factory=AgentParams)
    db_path: Path = DB_PATH
    trace: list[dict] = field(default_factory=list)
    _attempts: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ 构建
    @classmethod
    def build(cls, raw_dir: Path | str = RAW_DIR) -> "Toolbox":
        from .data import load_processed

        model = RiskModel.load()
        X_raw, y, ts, pre = load_processed(raw_dir)
        lots = load_lots()
        kb = KnowledgeBase()
        explainer = Explainer(model, X_raw.loc[pre.train_index])
        return cls(
            model=model,
            X_raw=X_raw,
            X_proc=X_raw,
            y=y,
            ts=ts,
            lots=lots,
            kb=kb,
            explainer=explainer,
            params=AgentParams(risk_threshold=model.threshold),
        )

    # ------------------------------------------------------------- 轨迹记录
    def _record(self, node: str, tool: str, args: dict, ok: bool, output: dict,
                summary: str, elapsed_ms: int, attempt: int, error: str | None = None) -> dict:
        step = TraceStep(
            index=len(self.trace) + 1,
            node=node,
            tool=tool,
            tool_doc=TOOL_DOCS.get(tool, ""),
            args=args,
            ok=ok,
            output=output,
            summary=summary,
            elapsed_ms=elapsed_ms,
            attempt=attempt,
            error=error,
        ).as_dict()
        self.trace.append(step)
        return step

    def _call(self, node: str, tool: str, args: dict, fn: Callable[[], tuple[dict, str]]) -> dict:
        """统一执行入口：计时、审计、可控故障注入（演示重试）。"""

        self._attempts[tool] = self._attempts.get(tool, 0) + 1
        attempt = self._attempts[tool]
        t0 = time.perf_counter()

        fail_once = os.getenv("ZHIXI_DEMO_FAIL_FIRST_CALL", "").strip()
        if fail_once and fail_once in (tool, node) and attempt == 1:
            elapsed = int((time.perf_counter() - t0) * 1000)
            msg = f"{tool} 首次调用超时（演示用可控故障），已进入重试队列"
            self._record(node, tool, args, False, {}, msg, elapsed, attempt, msg)
            raise DemoFault(msg)

        try:
            output, summary = fn()
        except Exception as exc:  # noqa: BLE001 - 工具异常需要被图捕获以触发重试
            elapsed = int((time.perf_counter() - t0) * 1000)
            self._record(node, tool, args, False, {}, str(exc), elapsed, attempt, str(exc))
            raise
        elapsed = int((time.perf_counter() - t0) * 1000)
        self._record(node, tool, args, True, output, summary, elapsed, attempt)
        return output

    # ------------------------------------------------------------- 批次索引
    def _row_index(self, lot_id: str) -> int:
        hit = self.lots.loc[self.lots["lot_id"] == lot_id, "row_index"]
        if hit.empty:
            raise ValueError(f"未找到批次 {lot_id}")
        return int(hit.iloc[0])

    def lot_meta(self, lot_id: str) -> dict:
        row = self.lots.loc[self.lots["lot_id"] == lot_id].iloc[0]
        return {
            "lot_id": lot_id,
            "timestamp": str(row["timestamp"]),
            "y_true": int(row["y_true"]),
            "risk": float(row["risk"]),
            "split": row["split"],
        }

    def latest_lot_id(self) -> str:
        return str(self.lots.sort_values("timestamp").iloc[-1]["lot_id"])

    def top_risk_lot(self, window: int = 300) -> str:
        recent = self.lots.sort_values("timestamp").tail(window)
        return str(recent.sort_values("risk", ascending=False).iloc[0]["lot_id"])

    # --------------------------------------------------------------- 工具 1
    def get_sensor_data(self, lot_id: str, node: str = "fetch_sensor", top_k: int = 12) -> dict:
        row_index = self._row_index(lot_id)

        def run() -> tuple[dict, str]:
            row = self.X_raw.loc[row_index]
            z = (row - self.explainer.mean) / self.explainer.std
            z = z.reindex(self.model.feature_names).fillna(0.0)
            order = z.abs().sort_values(ascending=False).head(top_k)

            signals = [
                {
                    "feature": f,
                    "value": None if pd.isna(row[f]) else round(float(row[f]), 6),
                    "missing": bool(pd.isna(row[f])),
                    "z": round(float(z[f]), 3),
                    "group": self.model.preprocess.group_of(f),
                }
                for f in order.index
            ]
            group_max: dict[str, float] = {}
            for f in self.model.feature_names:
                g = self.model.preprocess.group_of(f)
                group_max[g] = max(group_max.get(g, 0.0), abs(float(z[f])))
            groups = sorted(
                ({"station": g, "max_abs_z": round(v, 3)} for g, v in group_max.items()),
                key=lambda d: -d["max_abs_z"],
            )[:6]

            output = {
                "lot_id": lot_id,
                "row_index": row_index,
                "timestamp": str(self.lots.loc[self.lots["lot_id"] == lot_id, "timestamp"].iloc[0]),
                "n_signals": int(self.X_raw.shape[1]),
                "missing_values": int(row.isna().sum()),
                "signals": signals,
                "stations": groups,
                "shifted_signals": sum(1 for s in signals if abs(s["z"]) >= 3),
            }
            summary = (
                f"读取批次 {lot_id} 的 {output['n_signals']} 路有效工艺信号（原始 591 路，已剔除常量变量）；"
                f"超过 3σ 的信号 {output['shifted_signals']} 路，"
                f"偏差最大的工序站为 {groups[0]['station']}（{groups[0]['max_abs_z']}σ）"
            )
            return output, summary

        return self._call(node, "get_sensor_data", {"lot_id": lot_id, "top_k": top_k}, run)

    # --------------------------------------------------------------- 工具 2
    def predict_failure(self, lot_id: str, node: str = "predict_failure") -> dict:
        row_index = self._row_index(lot_id)

        def run() -> tuple[dict, str]:
            risk_recomputed = float(self.model.predict_proba(self.X_raw.iloc[[row_index]])[0])
            thr = float(self.model.threshold)
            high = float(self.model.high_threshold)

            # 历史批次直接使用袋外评分：集成模型在训练集上仍有样本内乐观偏差，
            # 袋外评分才是"当时对未知批次会给出什么判断"的无偏估计。
            hist = self.lots.loc[self.lots["lot_id"] == lot_id, "risk"]
            risk = float(hist.iloc[0]) if not hist.empty else risk_recomputed
            level = "高" if risk >= high else ("中" if risk >= thr else "低")
            output = {
                "lot_id": lot_id,
                "risk": round(risk, 4),
                "risk_recomputed": round(risk_recomputed, 4),
                "risk_source": "out-of-fold（袋外评分）" if not hist.empty else "ensemble（实时集成推理）",
                "threshold": round(thr, 4),
                "high_threshold": round(high, 4),
                "level": level,
                "is_high": bool(risk >= thr),
                "model": f"{self.model.kind} × {len(self.model.estimators)} 集成",
                "dataset": "UCI SECOM",
            }
            summary = (
                f"模型判定该批次异常概率 {risk:.1%}"
                f"（触发阈值 {thr:.1%}，高优先级线 {high:.1%}）→ 风险等级：{level}"
            )
            return output, summary

        return self._call(node, "predict_failure", {"lot_id": lot_id}, run)

    # --------------------------------------------------------------- 工具 3
    def explain_failure(self, lot_id: str, node: str = "explain_failure") -> dict:
        row_index = self._row_index(lot_id)

        def run() -> tuple[dict, str]:
            contribs: list[Contribution] = self.explainer.contributions(
                self.X_raw, row_index, top_k=self.params.top_k_features
            )
            ranking = self.explainer.group_ranking(contribs)
            output = {
                "lot_id": lot_id,
                "backend": self.explainer.backend,
                "contributions": [asdict(c) for c in contribs],
                "station_ranking": [
                    {"station": g, "contribution": round(v, 4)} for g, v in ranking[:4]
                ],
                "suspected_station": ranking[0][0] if ranking else "工序站A",
            }
            positives = [c for c in contribs if c.shap_value > 0] or contribs
            top_txt = "；".join(
                f"{c.feature}({c.group}) {c.shap_value:+.2f}" for c in positives[:3]
            )
            summary = f"可解释性后端 {self.explainer.backend}，主要拉升风险的信号：{top_txt}"
            return output, summary

        return self._call(node, "explain_failure", {"lot_id": lot_id}, run)

    # --------------------------------------------------------------- 工具 4
    def search_maintenance_manual(
        self,
        query: "str | list[str]",
        node: str = "retrieve_manual",
        k: int | None = None,
    ) -> dict:
        """检索维修知识库。

        query 传字符串时做单条检索；传字符串列表时按多个「关注面」检索并合并，
        这样返回的依据能覆盖不同环节，而不是三条都来自同一份手册。
        """

        k = k or self.params.top_k_evidence

        def run() -> tuple[dict, str]:
            queries = [query] if isinstance(query, str) else list(query)
            hits = (
                self.kb.search(queries[0], k=k)
                if len(queries) == 1
                else self.kb.search_aspects(queries, k=k)
            )
            if not hits:
                raise LookupError(f"知识库未检索到与“{query}”相关条目")
            output = {
                "query": queries if len(queries) > 1 else queries[0],
                "aspects": len(queries),
                "backend": self.kb.backend,
                "evidence": [h.as_dict() for h in hits],
            }
            summary = "命中维修依据：" + "、".join(h.citation for h in hits)
            return output, summary

        return self._call(node, "search_maintenance_manual", {"query": query, "k": k}, run)

    # --------------------------------------------------------------- 工具 5
    def simulate_maintenance_decision(
        self, lot_id: str, risk: float | None = None, node: str = "simulate_decision",
        horizon_hours: float | None = None,
    ) -> dict:
        """期望代价模拟。

        模型：把风险 p 视为“该批次出现质量异常的概率”，按批次线性累加期望损失；
        检修后残存风险 p' = max(基线不良率, p × (1-检修有效性))。
        所有参数均可在界面上调整，结果标注为“模拟估算”。
        """

        def run() -> tuple[dict, str]:
            p = float(risk if risk is not None else self.model.predict_proba(
                self.X_raw.iloc[[self._row_index(lot_id)]])[0])
            b = self.business
            H = float(horizon_hours or b.shift_hours)
            n_total = H / b.hours_per_lot
            loss_per_bad_batch = b.batch_value * (1 - b.rework_recovery_ratio)
            p0 = float(self.y.mean())
            effectiveness = 0.7
            delay_penalty = 12000.0
            urgency = 0.5

            h_delay = min(2.0, H)
            # A 立即停机检修：属于非计划停机，停机成本上浮
            cost_downtime_A = b.downtime_cost_per_hour * b.repair_hours * b.emergency_stop_multiplier
            q_A = min(1.0, b.repair_hours / H)
            cost_A = cost_downtime_A + q_A * urgency * delay_penalty
            n_A2 = max(0.0, (H - b.repair_hours)) / b.hours_per_lot
            quality_A = n_A2 * p0 * loss_per_bad_batch
            # B 生产 h 小时后检修：可计划安排，按标准停机成本计
            cost_downtime_B = b.downtime_cost_per_hour * b.repair_hours
            n_B1 = h_delay / b.hours_per_lot
            p_res = max(p0, p * (1 - effectiveness))
            n_B2 = max(0.0, H - h_delay - b.repair_hours) / b.hours_per_lot
            q_B = min(1.0, (h_delay + b.repair_hours) / H)
            quality_B = n_B1 * p * loss_per_bad_batch + n_B2 * p_res * loss_per_bad_batch
            cost_B = cost_downtime_B + q_B * urgency * delay_penalty + quality_B
            # C 继续生产到班次结束
            quality_C = n_total * p * loss_per_bad_batch
            cost_C = quality_C

            options = [
                {
                    "id": "A",
                    "name": "立即停机检修",
                    "expected_cost": round(cost_A + quality_A, 1),
                    "quality_loss": round(quality_A, 1),
                    "downtime_cost": round(cost_downtime_A, 1),
                    "delay_risk_cost": round(q_A * urgency * delay_penalty, 1),
                    "downtime_hours": b.repair_hours,
                    "note": "风险立即被拦截；属非计划停机，停机成本按 1.3 倍计",
                },
                {
                    "id": "B",
                    "name": f"继续生产 {h_delay:g} 小时后检修",
                    "expected_cost": round(cost_B, 1),
                    "quality_loss": round(quality_B, 1),
                    "downtime_cost": round(cost_downtime_B, 1),
                    "delay_risk_cost": round(q_B * urgency * delay_penalty, 1),
                    "downtime_hours": b.repair_hours,
                    "note": f"可计划安排，停机成本不上浮；检修后残存风险降至 {p_res:.1%}",
                },
                {
                    "id": "C",
                    "name": "不停机，生产到班次结束",
                    "expected_cost": round(cost_C, 1),
                    "quality_loss": round(quality_C, 1),
                    "downtime_cost": 0.0,
                    "delay_risk_cost": 0.0,
                    "downtime_hours": 0.0,
                    "note": "不承担停机与交期风险，承担全部质量风险",
                },
            ]
            options.sort(key=lambda o: o["expected_cost"])
            best = options[0]
            output = {
                "lot_id": lot_id,
                "risk": round(p, 4),
                "horizon_hours": H,
                "options": options,
                "recommended": best["id"],
                "saving_vs_worst": round(options[-1]["expected_cost"] - best["expected_cost"], 1),
                "assumptions": {
                    **{k: v for k, v in b.__dict__.items()},
                    "baseline_defect_rate": round(p0, 4),
                    "repair_effectiveness": effectiveness,
                    "delay_penalty": delay_penalty,
                    "delay_urgency": urgency,
                    "disclaimer": "上述参数为演示用假设值，结果为模拟估算，不代表真实企业收益。",
                },
            }
            summary = (
                f"三方案模拟：优选 {best['name']}，期望代价 ¥{best['expected_cost']:,.0f}，"
                f"较最差方案节省 ¥{output['saving_vs_worst']:,.0f}"
            )
            return output, summary

        return self._call(
            node, "simulate_maintenance_decision",
            {"lot_id": lot_id, "risk": risk, "horizon_hours": horizon_hours}, run,
        )

    # --------------------------------------------------------------- 工具 6
    def create_work_order(
        self,
        lot_id: str,
        action: str,
        station: str,
        priority: str = "中",
        owner: str = "设备维护组",
        spare_parts: str = "",
        risk: float | None = None,
        evidence: str = "",
        repair_hours: float | None = None,
        node: str = "create_work_order",
    ) -> dict:
        def run() -> tuple[dict, str]:
            wo = db.create_work_order(
                lot_id=lot_id,
                station=station,
                priority=priority,
                risk=float(risk or 0.0),
                action=action,
                owner=owner,
                spare_parts=spare_parts,
                evidence=evidence,
                repair_hours=float(repair_hours or self.business.repair_hours),
                path=self.db_path,
            )
            output = {"work_order": wo}
            summary = (
                f"工单已写入业务数据库：{wo['code']}｜批次 {lot_id}｜工序站 {station}｜"
                f"优先级 {priority}｜负责人 {owner}"
            )
            return output, summary

        return self._call(
            node,
            "create_work_order",
            {"lot_id": lot_id, "station": station, "priority": priority, "action": action},
            run,
        )

    # -------------------------------------------------------- 便捷：整条链路
    def scan_recent(self, window: int = 300) -> dict:
        recent = self.lots.sort_values("timestamp").tail(window)
        high = recent[recent["risk"] >= self.model.threshold]
        return {
            "window": window,
            "n_lots": int(len(recent)),
            "n_high_risk": int(len(high)),
            "high_risk_lots": [
                {"lot_id": r["lot_id"], "risk": round(float(r["risk"]), 4), "timestamp": str(r["timestamp"])}
                for _, r in high.sort_values("risk", ascending=False).head(8).iterrows()
            ],
        }

    def trace_json(self) -> str:
        return json.dumps(self.trace, ensure_ascii=False, indent=2)

    def reset_trace(self) -> None:
        self.trace.clear()
        self._attempts.clear()


def evidence_text(evidence: list[dict], k: int = 2) -> str:
    return " ｜ ".join(
        f"{e['citation']}：{strip_markdown(e['text'], 120)}" for e in evidence[:k]
    )
