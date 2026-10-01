"""智维Agent 演示界面（Streamlit）。

启动：
    streamlit run app.py

界面分五个区域：
    ① 监控与诊断   ② Agent 执行轨迹   ③ 决策与工单   ④ 维修知识库   ⑤ 模型评估
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from zhixi import db
from zhixi.config import llm_enabled
from zhixi.graph import build_graph, invoke, new_thread, resume, snapshot
from zhixi.model import PR_CURVE_PATH, load_metrics
from zhixi.tools import Toolbox

st.set_page_config(page_title="智维Agent · 设备维保决策智能体", page_icon="🏭", layout="wide")

# Streamlit 新旧版本对宽度参数名不同，这里做一次兼容。
try:
    from packaging.version import Version

    _NEW_W = Version(st.__version__) >= Version("1.49")
except Exception:
    _NEW_W = False
_W = {"width": "stretch"} if _NEW_W else {"use_container_width": True}
_WC = {"width": "content"} if _NEW_W else {"use_container_width": False}


# ------------------------------------------------------------------ 访问口令（可选）
# 默认**不设口令**，任何人打开链接即可使用。
# 若比赛要求提交账号密码，只需配置其中之一即可自动启用口令门：
#   1. Streamlit / Hugging Face 的 secrets 里加 password = "你的口令"
#   2. 或设置环境变量 ZHIXI_PASSWORD
def _access_password() -> str:
    try:
        if "password" in st.secrets:
            return str(st.secrets["password"])
    except Exception:
        pass
    return os.getenv("ZHIXI_PASSWORD", "")


def require_login() -> None:
    password = _access_password()
    if not password or st.session_state.get("auth_ok"):
        return

    st.markdown("## 🏭 智维Agent")
    st.caption("面向工序设备的多模态故障诊断与自主维保决策智能体 · 评审专用入口")
    with st.form("login"):
        given = st.text_input("访问口令", type="password", placeholder="请输入评审访问口令")
        submitted = st.form_submit_button("进入系统", type="primary")
    if submitted:
        if given == password:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("口令不正确，请核对评审通知中的密码。")
    st.stop()


require_login()


# ---------------------------------------------------------------- 资源加载
@st.cache_resource(show_spinner="正在加载模型、知识库与业务数据库…")
def get_toolbox() -> Toolbox:
    return Toolbox.build()


@st.cache_resource(show_spinner="正在编译 LangGraph 智能体…")
def get_graph(_toolbox: Toolbox):
    return build_graph(_toolbox)


try:
    toolbox = get_toolbox()
    graph = get_graph(toolbox)
except FileNotFoundError as exc:
    st.error(f"启动失败：{exc}（请先运行 python setup_data.py 与 python train_model.py）")
    st.stop()

metrics = load_metrics()
lots = toolbox.lots.sort_values("timestamp").reset_index(drop=True)


def pretty_lot(row) -> str:
    return f"{row['lot_id']} ｜ {row['timestamp']:%m-%d %H:%M} ｜ 风险 {row['risk']:.1%} ｜ {row['level']}"


# ------------------------------------------------------------------ 侧边栏
with st.sidebar:
    st.title("🏭 智维Agent")
    st.caption("面向工序设备的多模态故障诊断与自主维保决策智能体")

    st.subheader("模型")
    if metrics:
        st.caption(
            f"{metrics['model_kind']} × {metrics['ensemble_size']} 集成 ｜ "
            f"SECOM {metrics['n_samples']} 批 × {metrics['n_features']} 路信号"
        )
        c1, c2 = st.columns(2)
        c1.metric("PR-AUC", f"{metrics['cv']['pr_auc']:.3f}")
        c2.metric("Recall", f"{metrics['cv']['recall']:.3f}")
        c1.metric("F1", f"{metrics['cv']['f1']:.3f}")
        c2.metric("ROC-AUC", f"{metrics['cv']['roc_auc']:.3f}")
        st.caption(
            f"基准不良率 {metrics['base_rate']:.1%} ｜ 触发阈值 "
            f"{metrics['thresholds']['operational_threshold']:.1%}"
        )
    else:
        st.warning("尚未训练模型，请先运行 `python train_model.py`")

    st.divider()
    st.subheader("业务参数（模拟）")
    b = toolbox.business
    b.batch_value = st.number_input("单批次产值（元）", 0.0, 1e6, b.batch_value, 500.0)
    b.downtime_cost_per_hour = st.number_input(
        "停机损失（元/小时）", 0.0, 1e6, b.downtime_cost_per_hour, 100.0
    )
    b.repair_hours = st.number_input("计划检修工时（小时）", 0.1, 48.0, b.repair_hours, 0.5)
    b.shift_hours = st.number_input("班次剩余时间（小时）", 0.5, 24.0, b.shift_hours, 0.5)
    b.rework_recovery_ratio = st.slider(
        "异常批次返工可挽回比例", 0.0, 0.95, b.rework_recovery_ratio, 0.05
    )
    st.caption("以上为演示假设值，决策结果为模拟估算。")

    st.divider()
    st.subheader("大模型")
    if llm_enabled():
        st.success("已配置 LLM：意图解析与结论生成走大模型")
    else:
        st.info("未配置 API Key，使用离线规则模式（链路完全可用）")

    st.divider()
    fail_target = st.selectbox(
        "演示：让工具首次调用失败（验证重试分支）",
        ["（不注入）", "search_maintenance_manual", "predict_failure", "simulate_maintenance_decision"],
    )
    if fail_target != "（不注入）":
        os.environ["ZHIXI_DEMO_FAIL_FIRST_CALL"] = fail_target
    else:
        os.environ.pop("ZHIXI_DEMO_FAIL_FIRST_CALL", None)

    if st.button("清空工单库", **_W):
        db.clear_all()
        st.success("工单库已清空")
        st.rerun()

    if st.session_state.get("auth_ok"):
        if st.button("退出登录", **_WC):
            st.session_state["auth_ok"] = False
            st.rerun()


# ------------------------------------------------------------ 顶部指标区
st.title("智维Agent —— 从“设备报警”到“诊断—决策—工单—复盘”的闭环")
st.caption(
    "数据来源：UCI SECOM 半导体前道工序数据集（1567 批次 × 591 路工艺信号，CC BY 4.0）。"
    "变量为匿名过程测量点，已按相关性聚为 12 个工序信号簇（工序站 A–L）。"
)

state = st.session_state.get("state", {})
config = st.session_state.get("config")
next_nodes = st.session_state.get("next", [])

risk = state.get("risk") or {}
orders = db.list_work_orders(limit=200)
cols = st.columns(5)
cols[0].metric("当前批次", state.get("lot_id", "—"))
cols[1].metric("异常概率", f"{risk.get('risk', 0):.1%}" if risk else "—")
cols[2].metric("风险等级", risk.get("level", "—") if risk else "—")
cols[3].metric(
    "触发阈值", f"{metrics['thresholds']['operational_threshold']:.1%}" if metrics else "—"
)
cols[4].metric("工单总数", len(orders))

if next_nodes:
    st.warning("⏸ 智能体已暂停，等待人工确认（LangGraph interrupt）—— 请到「③ 决策与工单」页确认。")

tab_monitor, tab_trace, tab_decision, tab_kb, tab_eval = st.tabs(
    ["① 监控与诊断", "② Agent 执行轨迹", "③ 决策与工单", "④ 维修知识库", "⑤ 模型评估"]
)


# ------------------------------------------------------------ ① 监控与诊断
with tab_monitor:
    default_lot = toolbox.top_risk_lot(window=300)
    default_idx = int(lots.index[lots["lot_id"] == default_lot][0])
    picked = st.selectbox(
        "选择批次（默认自动选中近期风险最高的批次）",
        lots.index.tolist(),
        index=default_idx,
        format_func=lambda i: pretty_lot(lots.loc[i]),
    )
    lot_id = lots.loc[picked, "lot_id"]

    question = st.text_input(
        "向智能体提问",
        value=f"批次 {lot_id} 为什么风险高？要不要停机检修？",
        key=f"q_{lot_id}",
    )
    c1, c2 = st.columns([1, 1])
    run = c1.button("▶ 运行诊断", type="primary", **_W)
    if c2.button("🧹 清空执行轨迹", **_WC):
        toolbox.reset_trace()
        st.session_state.pop("state", None)
        st.session_state.pop("next", None)
        st.rerun()

    if run:
        toolbox.reset_trace()
        cfg = new_thread(toolbox)
        with st.spinner("智能体正在执行…"):
            st.session_state["state"] = invoke(toolbox, graph, question, cfg)
        st.session_state["config"] = cfg
        st.session_state["next"] = snapshot(graph, cfg)["next"]
        st.rerun()

    st.divider()
    left, right = st.columns([1, 1])

    lot_row = lots.loc[picked]
    with left:
        st.subheader("批次风险")
        st.progress(min(max(float(lot_row["risk"]), 0.0), 1.0))
        st.caption(
            f"袋外评分 {lot_row['risk']:.2%}｜集成实时评分 {lot_row['risk_ensemble']:.2%}｜"
            f"等级 {lot_row['level']}｜真实标签 "
            f"{'异常' if lot_row['y_true'] == 1 else '正常'}"
            "（真实标签仅用于演示核对，智能体看不到）"
        )
        raw_row = toolbox.X_raw.loc[int(lot_row["row_index"])]
        z = (
            (raw_row - toolbox.explainer.mean)
            / toolbox.explainer.std
        ).reindex(toolbox.model.feature_names).fillna(0.0)
        top = z.abs().sort_values(ascending=False).head(15).index[::-1]
        fig = go.Figure(
            go.Bar(
                x=[float(z[f]) for f in top],
                y=list(top),
                orientation="h",
                marker_color=["#d62728" if abs(float(z[f])) >= 3 else "#4c78a8" for f in top],
            )
        )
        fig.update_layout(
            height=430,
            title="偏差最大的 15 路工艺信号（标准分 σ）",
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="σ",
        )
        st.plotly_chart(fig, **_W)

    with right:
        st.subheader("全流程风险分布（回放历史批次流）")
        fig2 = px.scatter(
            lots,
            x="timestamp",
            y="risk",
            color="level",
            color_discrete_map={"高": "#d62728", "中": "#f2a900", "低": "#8c8c8c"},
            hover_data=["lot_id", "y_true"],
            labels={"timestamp": "时间", "risk": "异常概率", "level": "等级"},
        )
        thr = metrics["thresholds"]["operational_threshold"] if metrics else 0.5
        fig2.add_hline(
            y=thr, line_dash="dash", line_color="#d62728", annotation_text=f"触发阈值 {thr:.1%}"
        )
        fig2.update_layout(
            height=430, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="等级"
        )
        st.plotly_chart(fig2, **_W)

    if state.get("report"):
        st.subheader("智能体结论")
        st.info(state["report"])
    if state.get("scan"):
        st.subheader("批次巡检结果")
        st.json(state["scan"])


# -------------------------------------------------------- ② 执行轨迹
with tab_trace:
    st.subheader("Agent 执行轨迹")
    st.caption(
        "每一步都是真实的 Python 工具调用：包含调用参数、返回结果、耗时与重试次数。"
        "这是“可验证智能体交互”的直接证据。"
    )
    trace = state.get("trace") or toolbox.trace
    if not trace:
        st.info("还没有执行记录。请在「① 监控与诊断」中点击“运行诊断”。")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": s["index"],
                        "节点": s["node"],
                        "工具": s["tool"],
                        "状态": "✓" if s["ok"] else "✗",
                        "耗时(ms)": s["elapsed_ms"],
                        "第几次调用": s["attempt"],
                        "结果摘要": s["summary"],
                    }
                    for s in trace
                ]
            ),
            hide_index=True,
            **_W,
        )
        for s in trace:
            icon = "✅" if s["ok"] else "❌"
            with st.expander(f"{icon} {s['index']}. [{s['node']}] {s['tool']} — {s['summary']}"):
                st.caption(s["tool_doc"])
                st.markdown("**调用参数**")
                st.json(s["args"])
                st.markdown("**返回结果**")
                st.json(s["output"])
                if s["error"]:
                    st.error(s["error"])
        st.download_button(
            "下载完整轨迹 JSON",
            data=json.dumps(trace, ensure_ascii=False, indent=2),
            file_name="agent_trace.json",
            mime="application/json",
        )


# -------------------------------------------------------- ③ 决策与工单
with tab_decision:
    decision = state.get("decision") or {}
    proposal = state.get("proposal") or {}

    if decision.get("options"):
        st.subheader("维保方案模拟对比")
        st.dataframe(
            pd.DataFrame(decision["options"])[
                ["id", "name", "expected_cost", "quality_loss", "downtime_cost",
                 "delay_risk_cost", "downtime_hours", "note"]
            ].rename(
                columns={
                    "id": "方案",
                    "name": "方案名称",
                    "expected_cost": "期望代价(元)",
                    "quality_loss": "质量损失(元)",
                    "downtime_cost": "停机损失(元)",
                    "delay_risk_cost": "交期风险(元)",
                    "downtime_hours": "停机时长(h)",
                    "note": "说明",
                }
            ),
            hide_index=True,
            **_W,
        )
        st.caption("模拟参数：" + json.dumps(decision.get("assumptions", {}), ensure_ascii=False))

    if proposal:
        st.subheader("智能体建议")
        st.markdown(
            f"**动作**：{proposal.get('action')}  \n"
            f"**工序站**：{proposal.get('station')}　**优先级**：{proposal.get('priority')}　"
            f"**负责人**：{proposal.get('owner')}　**预计工时**：{proposal.get('repair_hours')} 小时  \n"
            f"**备件**：{proposal.get('spare_parts') or '—'}  \n"
            f"**决策依据**：{proposal.get('evidence_summary')}"
        )

    if next_nodes and "human_review" in next_nodes:
        st.divider()
        st.markdown("### 人工确认")
        c1, c2, c3 = st.columns([1, 1, 2])
        operator = c3.text_input("确认人", value="值班工程师")
        if c1.button("✅ 确认生成工单", type="primary", **_W):
            st.session_state["state"] = resume(
                graph, config, {"approved": True, "operator": operator}
            )
            st.session_state["next"] = snapshot(graph, config)["next"]
            st.rerun()
        if c2.button("⛔ 拒绝，转人工", **_W):
            st.session_state["state"] = resume(
                graph,
                config,
                {"approved": False, "operator": operator, "note": "工程师选择人工处理"},
            )
            st.session_state["next"] = snapshot(graph, config)["next"]
            st.rerun()
    elif state:
        st.success("本轮流程已结束（无待确认事项）。")

    st.divider()
    st.subheader("工单台账（SQLite 业务数据库）")
    if orders.empty:
        st.info("暂无工单。")
    else:
        st.dataframe(orders, hide_index=True, **_W)
        done = st.selectbox("选择要关闭的工单", ["（不操作）"] + orders["code"].tolist())
        if done != "（不操作）":
            note = st.text_input("复测结论", value="复机后信号回落至基线，风险概率已降至阈值以下")
            if st.button("标记为已完成并写入复盘记录"):
                db.close_work_order(done, note)
                st.success(f"{done} 已关闭")
                st.rerun()


# -------------------------------------------------------- ④ 维修知识库
with tab_kb:
    st.subheader("维修知识库检索（RAG）")
    st.caption(
        f"检索后端：{toolbox.kb.backend}（字符 n-gram TF-IDF；安装 faiss-cpu 后自动切换为 FAISS 向量检索）"
        f"｜知识条目：{len(toolbox.kb.docs)} 条"
    )
    query = st.text_input("输入检索问题", value="过程参数漂移 怎么排查 停机")
    if query:
        for h in toolbox.kb.search(query, k=5):
            with st.expander(f"[{h.score:.3f}] {h.citation}"):
                st.markdown(h.text)

    if state.get("evidence"):
        st.divider()
        st.subheader("本轮智能体实际检索到的依据")
        for e in state["evidence"]["evidence"]:
            st.markdown(f"- **{e['citation']}**（相似度 {e['score']}）")


# -------------------------------------------------------- ⑤ 模型评估
with tab_eval:
    st.subheader("模型评估")
    if not metrics:
        st.warning("尚未训练模型。")
    else:
        st.caption(metrics["dataset"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**主协议：分层 5 折交叉验证（袋外预测）**")
            st.dataframe(
                pd.DataFrame([metrics["cv"]])
                .T.drop(index=["confusion"], errors="ignore")
                .rename(columns={0: "数值"}),
                **_W,
            )
            st.json(metrics["cv"]["confusion"])
        with c2:
            st.markdown("**漂移检查：按时间切分留出集**")
            st.dataframe(
                pd.DataFrame([metrics["holdout"]])
                .T.drop(index=["confusion"], errors="ignore")
                .rename(columns={0: "数值"}),
                **_W,
            )
            st.json(metrics["holdout"]["confusion"])
            st.caption(
                "时间留出集明显劣化，说明 SECOM 存在真实的时间漂移（工艺与设备状态随时间变化）。"
                "这正是上线时必须做滚动重训与阈值在线校准的原因。"
            )

        st.markdown("**PR 曲线（时间留出集）**")
        st.line_chart(pd.read_csv(PR_CURVE_PATH), x="recall", y="precision")

        st.markdown("**风险等级分层效果（历史批次，袋外评分）**")
        st.dataframe(
            lots.groupby("level")["y_true"]
            .agg(批次数="count", 真实异常数="sum", 异常率="mean")
            .assign(异常率=lambda d: (d["异常率"] * 100).round(2))
            .rename(columns={"异常率": "异常率(%)"}),
            **_W,
        )
        st.caption(
            f"“高”档的异常率约为基准不良率（{metrics['base_rate']:.1%}）的 "
            f"{lots[lots['level'] == '高']['y_true'].mean() / metrics['base_rate']:.1f} 倍，"
            "说明风险排序有效；绝对准确率受 SECOM 本身的可分性限制。"
        )
