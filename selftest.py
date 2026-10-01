"""智维Agent 自检脚本 —— 一条命令跑完全部验收检查。

用法：
    python selftest.py            # 跑全部检查
    python selftest.py -v         # 显示每一步的详细内容

覆盖范围：
    数据完整性 → 预处理 → 模型产物 → 知识库 → 6 个工具 → 图的四条分支 → 工单库

自检使用独立的临时工单库，不会污染 artifacts/workorders.db。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

RESULTS: list[tuple[str, bool, str]] = []
VERBOSE = False


def check(name: str):
    """执行检查并把结果登记进结果表（装饰时立即运行）。"""

    def deco(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, str(detail)))
            print(f"  ✅ {name}")
            if VERBOSE and detail:
                print(f"       {detail}")
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, False, str(exc)))
            print(f"  ❌ {name}  →  {exc}")
            if VERBOSE:
                traceback.print_exc()
        return fn

    return deco


def section(title: str) -> None:
    print(f"\n{'─' * 66}\n{title}\n{'─' * 66}")


def main() -> int:
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    VERBOSE = args.verbose

    # 延迟导入，让 --help 也能快速返回
    from zhixi import db
    from zhixi.config import METRICS_PATH, MODEL_PATH, PREPROCESS_PATH
    from zhixi.data import load_secom
    from zhixi.graph import build_graph, invoke, new_thread, resume, snapshot
    from zhixi.model import load_lots, load_metrics
    from zhixi.tools import Toolbox

    tmp_db = Path(tempfile.mkdtemp(prefix="zhixi_selftest_")) / "workorders.db"

    # ------------------------------------------------------------ 数据
    section("① 数据集完整性")

    @check("SECOM 原始文件可读取，形状为 1567 × 591")
    def _():
        X, y, ts = load_secom()
        assert X.shape == (1567, 591), f"实际 {X.shape}"
        assert len(y) == 1567 and len(ts) == 1567
        return f"{X.shape[0]} 批次 × {X.shape[1]} 路信号"

    @check("异常批次数量为 104（占比约 6.6%）")
    def _():
        _, y, _ = load_secom()
        assert int(y.sum()) == 104, f"实际 {int(y.sum())}"
        return f"{int(y.sum())} 条异常，占比 {y.mean():.2%}"

    @check("时间戳完整且跨度合理")
    def _():
        _, _, ts = load_secom()
        assert ts.isna().sum() == 0
        return f"{ts.min():%Y-%m-%d} → {ts.max():%Y-%m-%d}"

    # ------------------------------------------------------------ 模型
    section("② 模型与预处理产物")

    @check("模型 / 预处理 / 指标文件齐备")
    def _():
        for p in (MODEL_PATH, PREPROCESS_PATH, METRICS_PATH):
            assert p.exists(), f"缺少 {p.name}"
        return "risk_model.joblib / preprocess.joblib / metrics.json"

    @check("主指标达到合格线（PR-AUC > 0.10 且高于基准不良率）")
    def _():
        m = load_metrics()
        assert m, "metrics.json 为空"
        pr, base = m["cv"]["pr_auc"], m["base_rate"]
        assert pr > 0.10, f"PR-AUC 仅 {pr}"
        assert pr > base, "PR-AUC 未超过基准不良率"
        return f"PR-AUC {pr} / 基准 {base} = {pr / base:.1f} 倍"

    @check("上线模型为集成模型（≥2 个基学习器）")
    def _():
        from zhixi.model import RiskModel

        model = RiskModel.load()
        assert len(model.estimators) >= 2, f"只有 {len(model.estimators)} 个"
        return f"{model.kind} × {len(model.estimators)}，触发阈值 {model.threshold:.1%}"

    @check("批次评分表完整、概率合法、编号唯一")
    def _():
        lots = load_lots()
        assert len(lots) == 1567
        assert lots["lot_id"].is_unique
        assert lots["risk"].between(0, 1).all()
        return f"{len(lots)} 条，风险区间 {lots['risk'].min():.4f} ~ {lots['risk'].max():.4f}"

    @check("风险分层有效：高档次异常率显著高于基准")
    def _():
        lots = load_lots()
        high = lots.loc[lots["level"] == "高", "y_true"].mean()
        base = lots["y_true"].mean()
        assert high > base * 1.5, f"高档次 {high:.3f} vs 基准 {base:.3f}"
        return f"高档次 {high:.1%} vs 基准 {base:.1%}"

    # ------------------------------------------------------------ 知识库
    section("③ 维修知识库（RAG）")

    toolbox = Toolbox.build()
    toolbox.db_path = tmp_db
    toolbox.reset_trace()

    @check("知识库加载成功且条目充足")
    def _():
        assert len(toolbox.kb.docs) >= 20, f"仅 {len(toolbox.kb.docs)} 条"
        return f"{len(toolbox.kb.docs)} 条知识片段，后端 {toolbox.kb.backend}"

    @check("检索能命中相关 SOP 并给出出处")
    def _():
        hits = toolbox.kb.search("过程参数漂移 排查 停机", k=3)
        assert hits, "没有命中任何条目"
        assert all(h.citation for h in hits)
        return "命中：" + "、".join(h.citation for h in hits[:2])

    # ------------------------------------------------------------ 工具
    section("④ 六个工具逐个调用")

    high_lot = str(toolbox.lots.nlargest(1, "risk")["lot_id"].iloc[0])
    low_lot = str(
        toolbox.lots[toolbox.lots["risk"] < toolbox.model.threshold * 0.5]
        .sort_values("timestamp").iloc[-1]["lot_id"]
    )

    @check(f"get_sensor_data 能读出信号与偏差（{high_lot}）")
    def _():
        d = toolbox.get_sensor_data(high_lot)
        assert d["n_signals"] > 0 and d["signals"]
        top = d["signals"][0]
        return f"{d['n_signals']} 路信号，最大偏差信号 {top['feature']}（{top['z']}σ）"

    @check(f"predict_failure 返回合法概率与等级（{high_lot}）")
    def _():
        d = toolbox.predict_failure(high_lot)
        assert 0 <= d["risk"] <= 1
        assert d["level"] in {"高", "中", "低"}
        return f"风险 {d['risk']:.1%}，等级 {d['level']}，来源 {d['risk_source']}"

    @check(f"explain_failure 给出带贡献值的归因（{high_lot}）")
    def _():
        d = toolbox.explain_failure(high_lot)
        assert d["contributions"], "没有归因结果"
        assert d["suspected_station"].startswith("工序站")
        return f"后端 {d['backend']}，定位 {d['suspected_station']}"

    @check("search_maintenance_manual 返回带出处的依据")
    def _():
        d = toolbox.search_maintenance_manual("工序站A 过程参数漂移 点检")
        assert d["evidence"] and d["evidence"][0]["citation"]
        return f"命中 {len(d['evidence'])} 条，首条 {d['evidence'][0]['citation']}"

    @check("simulate_maintenance_decision 给出 A/B/C 三方案并排序")
    def _():
        d = toolbox.simulate_maintenance_decision(high_lot)
        assert len(d["options"]) == 3
        costs = [o["expected_cost"] for o in d["options"]]
        assert costs == sorted(costs), "方案未按代价排序"
        assert d["recommended"] in {"A", "B", "C"}
        return f"推荐方案 {d['recommended']}，期望代价 ¥{costs[0]:,.0f}"

    @check("create_work_order 真实写入数据库并可查询")
    def _():
        d = toolbox.create_work_order(
            lot_id=high_lot, station="工序站A", priority="高",
            action="自检写入：立即停机检修", risk=0.9,
        )
        code = d["work_order"]["code"]
        df = db.list_work_orders(tmp_db)
        assert code in set(df["code"]), "工单未落库"
        return f"工单 {code} 已落库，共 {len(df)} 条"

    @check("close_work_order 能关闭工单并写入复盘结论")
    def _():
        code = db.list_work_orders(tmp_db)["code"].iloc[0]
        d = db.close_work_order(code, "自检：复测通过", path=tmp_db)
        assert d and d["status"] == "已完成"
        return f"{code} → {d['status']}"

    # ------------------------------------------------------------ 图
    section("⑤ LangGraph 四条执行分支")
    graph = build_graph(toolbox)

    @check("高风险链路：走到人工确认并暂停")
    def _():
        cfg = new_thread(toolbox)
        state = invoke(toolbox, graph, f"批次 {high_lot} 为什么风险高？要不要停机检修？", cfg)
        snap = snapshot(graph, cfg)
        assert "human_review" in snap["next"], f"next={snap['next']}"
        assert state.get("risk") and state.get("decision") and state.get("proposal")
        return f"{high_lot} 暂停于人工确认，轨迹 {len(state['trace'])} 步"

    @check("人工确认拒绝：不生成工单，转人工处理")
    def _():
        before = len(db.list_work_orders(tmp_db))
        cfg = new_thread(toolbox)
        invoke(toolbox, graph, f"批次 {high_lot} 要不要停机检修？", cfg)
        state = resume(graph, cfg, {"approved": False, "operator": "自检"})
        after = len(db.list_work_orders(tmp_db))
        assert after == before, "拒绝后仍然生成了工单"
        assert (state.get("approval") or {}).get("approved") is False, "审批结果未记录"
        assert state.get("stage") == "完成（人工接管）", f"阶段为 {state.get('stage')}"
        return f"工单数保持 {after} 条，阶段 {state.get('stage')}"

    @check("低风险链路：不触发决策与工单")
    def _():
        before = len(db.list_work_orders(tmp_db))
        cfg = new_thread(toolbox)
        state = invoke(toolbox, graph, f"批次 {low_lot} 需要检修吗？", cfg)
        after = len(db.list_work_orders(tmp_db))
        assert after == before, "低风险却生成了工单"
        assert state.get("stage") == "完成"
        assert state.get("decision") in (None, {}), "低风险不应进入决策模拟"
        return f"{low_lot} 风险 {(state.get('risk') or {}).get('risk', 0):.1%} → 放行"

    @check("工具失败后可自动重试并恢复")
    def _():
        os.environ["ZHIXI_DEMO_FAIL_FIRST_CALL"] = "search_maintenance_manual"
        try:
            cfg = new_thread(toolbox)
            state = invoke(toolbox, graph, f"批次 {high_lot} 为什么风险高？", cfg)
        finally:
            os.environ.pop("ZHIXI_DEMO_FAIL_FIRST_CALL", None)
        trace = state["trace"]
        failed = [i for i, s in enumerate(trace) if not s["ok"]]
        assert failed, "没有出现失败记录"
        assert any(s["ok"] for s in trace[failed[0]:]), "失败之后没有恢复"
        return f"第 {failed[0] + 1} 步失败，随后成功恢复，共 {len(trace)} 步"

    @check("巡检分支：扫描批次流并给出优先排查对象")
    def _():
        cfg = new_thread(toolbox)
        state = invoke(toolbox, graph, "帮我扫描最近的批次，看看有没有异常", cfg)
        scan = state.get("scan") or {}
        assert scan.get("n_lots"), "没有巡检结果"
        return f"扫描 {scan['n_lots']} 批，高风险 {scan['n_high_risk']} 个，建议排查 {scan.get('suggested_lot')}"

    # ------------------------------------------------------------ 汇总
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    section("自检汇总")
    print(f"  通过 {passed} / {len(RESULTS)} 项" + (f"，失败 {failed} 项" if failed else "，全部通过 🎉"))
    if failed:
        print("\n  未通过项：")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    ❌ {name}：{detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
