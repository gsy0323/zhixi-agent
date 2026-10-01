"""命令行端到端演示：一条命令跑完整个智能体闭环。

用法：
    python demo.py                         # 诊断最新批次（自动挑选近期最高风险批次）
    python demo.py --lot LOT-1370          # 指定批次
    python demo.py --scan                  # 巡检最近 40 个批次
    python demo.py --auto-approve          # 跳过人工确认（默认会在人工确认处暂停并打印）
    # 演示"工具失败 → 自动重试"（可用工具名或节点名）
    python demo.py --fail-first search_maintenance_manual
    python demo.py --fail-first retrieve_manual
"""

from __future__ import annotations

import argparse
import json
import os

from zhixi.graph import HAS_INTERRUPT, build_graph, invoke, new_thread, resume, snapshot
from zhixi.tools import Toolbox

BAR = "─" * 78


def show_trace(state: dict) -> None:
    print(f"\n{BAR}\nAgent 执行轨迹\n{BAR}")
    for step in state.get("trace", []):
        mark = "✓" if step["ok"] else "✗"
        print(
            f"{step['index']:>2}. {mark} [{step['node']}] {step['tool']} "
            f"({step['elapsed_ms']} ms, 第 {step['attempt']} 次调用)"
        )
        print(f"    → {step['summary']}")


def show_report(state: dict) -> None:
    print(f"\n{BAR}\n智能体结论\n{BAR}")
    print(state.get("report", "(无)"))
    scan = state.get("scan") or {}
    if scan:
        print(f"\n{BAR}\n批次巡检明细\n{BAR}")
        print(f"窗口：最近 {scan.get('n_lots')} 个批次；高风险 {scan.get('n_high_risk')} 个")
        for item in scan.get("high_risk_lots", []):
            print(f"  · {item['lot_id']}  风险 {item['risk']:.1%}  ({item['timestamp']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lot", default="")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument("--fail-first", default="", help="指定工具在首次调用时失败，用于演示重试")
    ap.add_argument("--json", action="store_true", help="额外打印完整状态 JSON")
    args = ap.parse_args()

    if args.fail_first:
        os.environ["ZHIXI_DEMO_FAIL_FIRST_CALL"] = args.fail_first

    print("正在加载模型、知识库与业务数据库…")
    toolbox = Toolbox.build()
    graph = build_graph(toolbox)

    if args.scan:
        question = "帮我扫描最近的批次，看看有没有异常"
    elif args.lot:
        question = f"批次 {args.lot} 为什么风险高？要不要停机检修？"
    else:
        lot = toolbox.top_risk_lot(window=300)
        question = f"批次 {lot} 为什么风险高？要不要停机检修？"
        print(f"（未指定批次，自动选择近期风险最高的批次：{lot}）")

    config = new_thread(toolbox)
    print(f"\n{BAR}\n用户提问：{question}\n{BAR}")

    state = invoke(toolbox, graph, question, config)
    show_trace(state)

    snap = snapshot(graph, config)
    if "human_review" in snap["next"]:
        prop = snap["values"].get("proposal", {})
        print(f"\n{BAR}\n⏸  已暂停，等待人工确认（LangGraph interrupt）\n{BAR}")
        print(f"建议动作：{prop.get('action')}")
        print(f"优先级：{prop.get('priority')}｜工序站：{prop.get('station')}｜负责人：{prop.get('owner')}")
        if not HAS_INTERRUPT:
            print("当前 langgraph 版本不支持 interrupt，跳过人工确认环节。")
            state = dict(snap["values"])
        elif args.auto_approve:
            print("（--auto-approve：自动确认通过）")
            state = resume(graph, config, {"approved": True, "operator": "值班工程师"})
        else:
            answer = input("是否生成维修工单？[y/N] ").strip().lower()
            state = resume(
                graph, config, {"approved": answer.startswith("y"), "operator": "值班工程师"}
            )
        show_trace(state)

    show_report(state)

    wo = state.get("work_order")
    if wo:
        print(f"\n{BAR}\n业务数据库新增工单\n{BAR}")
        for k in ["code", "lot_id", "station", "priority", "risk", "action", "owner",
                  "spare_parts", "due_at", "status"]:
            print(f"  {k:<12}: {wo.get(k)}")

    if args.json:
        print(f"\n{BAR}\n完整状态 JSON\n{BAR}")
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
