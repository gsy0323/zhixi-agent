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


THEME_CSS = """
<style>
  :root{
    --zx-bg:#eef2f8; --zx-surface:#fff; --zx-line:#e2e8f2; --zx-line2:#cfd8e6;
    --zx-ink:#0f1720; --zx-ink2:#334155; --zx-muted:#64748b; --zx-faint:#94a3b8;
    --zx-brand:#1f5fbf; --zx-brand2:#2a7fd4; --zx-tint:#eaf1fd;
    --zx-high:#dc4c4c; --zx-mid:#e2922f; --zx-ok:#2f9e6f;
    --zx-sh:0 1px 2px rgba(15,32,64,.05), 0 1px 3px rgba(15,32,64,.04);
    --zx-sh2:0 2px 6px rgba(15,32,64,.06), 0 10px 26px rgba(15,32,64,.06);
  }
  .stApp{ background:var(--zx-bg); }
  header[data-testid="stHeader"]{ background:transparent; }
  .block-container{ padding-top:1.2rem; padding-bottom:3rem; max-width:1400px; }

  /* 顶部横幅 */
  .zx-banner{
    position:relative; overflow:hidden; border-radius:16px; padding:26px 28px 22px; color:#fff;
    background:linear-gradient(118deg,#0f2a52 0%,#1f5fbf 52%,#2f88d8 100%);
    box-shadow:var(--zx-sh2); margin-bottom:20px;
  }
  .zx-banner::after{
    content:""; position:absolute; width:520px; height:520px; right:-150px; top:-300px;
    background:radial-gradient(circle,rgba(255,255,255,.20),transparent 66%);
  }
  .zx-brandrow{ position:relative; z-index:1; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .zx-mark{ width:42px;height:42px;border-radius:12px;display:grid;place-items:center;font-size:21px;
            background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3); }
  .zx-title{ font-size:23px; font-weight:700; letter-spacing:.4px; }
  .zx-live{ display:inline-flex; align-items:center; gap:7px; font-size:12px;
            background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.26);
            border-radius:999px; padding:4px 12px 4px 10px; }
  .zx-dot{ width:7px;height:7px;border-radius:50%;background:#5ee6a8;box-shadow:0 0 0 0 rgba(94,230,168,.7);
           animation:zxpulse 2.2s infinite; }
  @keyframes zxpulse{70%{box-shadow:0 0 0 8px rgba(94,230,168,0)}100%{box-shadow:0 0 0 0 rgba(94,230,168,0)}}
  .zx-sub{ position:relative; z-index:1; margin:10px 0 0; font-size:13px; line-height:1.75; opacity:.92; max-width:960px; }
  .zx-chips{ position:relative; z-index:1; display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
  .zx-chip{ background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22);
            border-radius:999px; padding:5px 13px; font-size:12px; }

  /* KPI 卡片 */
  .zx-kpirow{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:8px; }
  @media(max-width:1000px){ .zx-kpirow{ grid-template-columns:repeat(2,1fr); } }
  .zx-kpi{ position:relative; overflow:hidden; background:var(--zx-surface); border:1px solid var(--zx-line);
           border-radius:14px; padding:14px 16px; box-shadow:var(--zx-sh); }
  .zx-kpi::after{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--c,#1f5fbf); }
  .zx-kpi .k{ font-size:12px; color:var(--zx-muted); }
  .zx-kpi .v{ font-size:23px; font-weight:700; margin-top:5px; line-height:1.25; letter-spacing:.2px; }

  /* 卡片容器 */
  div[data-testid="stVerticalBlockBorderWrapper"]{
    background:var(--zx-surface); border:1px solid var(--zx-line); border-radius:14px;
    box-shadow:var(--zx-sh2); padding:8px 14px;
  }
  /* 标签页 */
  .stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--zx-line2); }
  .stTabs [data-baseweb="tab"]{ font-weight:600; color:var(--zx-muted); padding:10px 16px; }
  .stTabs [aria-selected="true"]{ color:var(--zx-brand) !important; }
  /* 按钮 */
  .stButton > button{ border-radius:9px; font-weight:600; }
  .stButton > button[kind="primary"]{
    background:linear-gradient(180deg,var(--zx-brand2),var(--zx-brand)); border:none;
    box-shadow:0 2px 8px rgba(31,95,191,.28);
  }
  .stButton > button[kind="secondary"]{ border:1px solid var(--zx-brand); color:var(--zx-brand); background:#fff; }
  /* 指标（侧边栏） */
  [data-testid="stMetricValue"]{ font-size:19px; font-weight:700; }
  [data-testid="stMetricLabel"]{ font-size:12px; color:var(--zx-muted); }
  /* 输入控件 */
  .stTextInput input, .stSelectbox div[data-baseweb="select"] > div{ border-radius:9px; }
  /* 展开器（执行轨迹） */
  div[data-testid="stExpander"]{
    border:1px solid var(--zx-line); border-radius:12px; background:#fff; box-shadow:var(--zx-sh); margin-bottom:8px;
  }
  div[data-testid="stExpander"] summary{ font-weight:600; }
  /* 表格 */
  [data-testid="stDataFrame"]{ border-radius:10px; border:1px solid var(--zx-line); }
  /* 提示框 */
  div[data-testid="stAlert"]{ border-radius:10px; }
  /* 进度条 */
  .stProgress > div > div > div > div{ background:linear-gradient(90deg,#2f9e6f,#e2922f,#dc4c4c); }
  /* 风险仪表 */
  .zx-gauge{ display:flex; align-items:center; gap:20px; flex-wrap:wrap; }
  .zx-gauge svg{ flex:0 0 200px; }
  .zx-risknum{ font-size:38px; font-weight:800; line-height:1.05; letter-spacing:-.5px; }
  .zx-badge{ display:inline-block; padding:3px 11px; border-radius:999px; font-size:12px; font-weight:700; }
  .zx-b-high{ background:#fdecec; color:#dc4c4c; }
  .zx-b-mid{ background:#fdf2e2; color:#a5701a; }
  .zx-b-low{ background:#eef1f6; color:#5b6675; }
  /* 方案卡 */
  .zx-opt{ position:relative; border:1.5px solid var(--zx-line); border-radius:14px; padding:16px; background:#fff; height:100%; }
  .zx-opt.best{ border-color:var(--zx-ok); box-shadow:0 6px 22px rgba(47,158,111,.16); }
  .zx-opt .rb{ position:absolute; top:-11px; right:14px; background:var(--zx-ok); color:#fff;
               font-size:11px; font-weight:700; padding:3px 11px; border-radius:999px; }
  .zx-opt .tag{ font-size:12px; font-weight:700; color:var(--zx-muted); }
  .zx-opt .nm{ font-size:15px; font-weight:700; margin:6px 0 10px; }
  .zx-opt .cost{ font-size:24px; font-weight:800; }
  .zx-opt .cost small{ font-size:12px; color:var(--zx-muted); font-weight:600; margin-left:4px; }
  .zx-opt .dt{ font-size:12px; color:var(--zx-muted); line-height:1.9; margin-top:10px;
               border-top:1px dashed var(--zx-line); padding-top:9px; }
  .zx-opt .dt span{ float:right; color:var(--zx-ink2); }
  /* 步骤标签 */
  .zx-b-pre{ background:#eef1f6; color:#5b6675; }
  .zx-b-live{ background:#e6f7ef; color:#2f9e6f; }
</style>
"""


def gauge_svg(value: float, threshold: float, level: str) -> str:
    """半圆风险仪表盘（与单文件 HTML 版同一套视觉）。"""

    import math

    cx, cy, r = 105, 100, 78
    v = min(max(float(value), 0.0), 1.0)
    ang = v * 180
    color = {"高": "#dc4c4c", "中": "#e2922f", "低": "#2f9e6f"}.get(level, "#2f9e6f")
    th = min(max(float(threshold), 0.0), 1.0) * 180

    def arc(a0: float, a1: float) -> str:
        ra0, ra1 = math.radians(a0 - 180), math.radians(a1 - 180)
        x0, y0 = cx + r * math.cos(ra0), cy + r * math.sin(ra0)
        x1, y1 = cx + r * math.cos(ra1), cy + r * math.sin(ra1)
        large = 1 if (a1 - a0) > 180 else 0
        return f"M {x0:.2f},{y0:.2f} A {r},{r} 0 {large} 1 {x1:.2f},{y1:.2f}"

    tx = cx + (r + 13) * math.cos(math.radians(th - 180))
    ty = cy + (r + 13) * math.sin(math.radians(th - 180))
    return f"""
    <svg viewBox="0 0 210 132" width="200" height="126">
      <path d="{arc(0,180)}" fill="none" stroke="#e6ebf3" stroke-width="15" stroke-linecap="round"/>
      <path d="{arc(0, max(ang, 0.6))}" fill="none" stroke="{color}" stroke-width="15" stroke-linecap="round"/>
      <line x1="{cx}" y1="{cy}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="#334155" stroke-width="2" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="5" fill="#334155"/>
      <text x="{cx-r}" y="{cy+22}" font-size="10" fill="#94a3b8" text-anchor="middle">0%</text>
      <text x="{cx+r}" y="{cy+22}" font-size="10" fill="#94a3b8" text-anchor="middle">100%</text>
    </svg>"""


def kpi_cards(items: list[tuple[str, str, str]]) -> None:
    """items: [(标题, 数值, 颜色)]"""

    cards = "".join(
        f'<div class="zx-kpi" style="--c:{color}"><div class="k">{title}</div>'
        f'<div class="v">{value}</div></div>'
        for title, value, color in items
    )
    st.markdown(f'<div class="zx-kpirow">{cards}</div>', unsafe_allow_html=True)


def confusion_table(c: dict, note: str = "") -> None:
    """把混淆矩阵渲染成表格，而不是原始的 JSON 代码块。"""

    tn, fp = int(c.get("tn", 0)), int(c.get("fp", 0))
    fn, tp = int(c.get("fn", 0)), int(c.get("tp", 0))
    cell = 'style="border-radius:9px;text-align:center;padding:9px 6px;{bg}"'

    def td(value: int, label: str, good: bool) -> str:
        bg = "#f1fbf6" if good else "#fdf2e2"
        color = "#2f9e6f" if good else "#a5701a"
        return (
            f'<td style="border-radius:9px;text-align:center;padding:9px 6px;background:{bg}">'
            f'<b style="font-size:17px;color:{color}">{value}</b>'
            f'<div style="font-size:11px;color:#64748b;margin-top:2px">{label}</div></td>'
        )

    st.markdown(
        f"""
        <div style="margin-top:12px">
          <div style="font-size:12px;color:#64748b;margin-bottom:6px">混淆矩阵{(' · ' + note) if note else ''}</div>
          <table style="width:100%;border-collapse:separate;border-spacing:4px;font-size:12.5px">
            <tr>
              <td style="color:#94a3b8;font-size:11px"></td>
              <td style="text-align:center;color:#64748b;font-size:11.5px">预测正常</td>
              <td style="text-align:center;color:#64748b;font-size:11.5px">预测异常</td>
            </tr>
            <tr>
              <td style="color:#64748b;white-space:nowrap">实际正常</td>
              {td(tn, "TN 正确放行", True)}
              {td(fp, "FP 误报", False)}
            </tr>
            <tr>
              <td style="color:#64748b;white-space:nowrap">实际异常</td>
              {td(fn, "FN 漏报", False)}
              {td(tp, "TP 正确拦截", True)}
            </tr>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


_METRIC_ROWS = [
    ("pr_auc", "PR-AUC", "{:.4f}"),
    ("roc_auc", "ROC-AUC", "{:.4f}"),
    ("recall", "召回率 Recall", "{:.4f}"),
    ("precision", "精确率 Precision", "{:.4f}"),
    ("f1", "F1", "{:.4f}"),
    ("balanced_accuracy", "平衡准确率", "{:.4f}"),
    ("threshold", "风险阈值", "{:.1%}"),
    ("alarm_rate", "报警比例", "{:.1%}"),
]


def metric_table(m: dict) -> None:
    """用中文字段名展示指标，避免出现 pr_auc 这类代码风格列名。"""

    rows = [
        {"指标": label, "数值": fmt.format(float(m[key]))}
        for key, label, fmt in _METRIC_ROWS
        if key in m and m[key] is not None
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, **_W)


def scan_result_block(scan: dict) -> None:
    """把批次巡检结果渲染成卡片式列表，而不是原始 JSON。"""

    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:7px 12px;'
        f'border-bottom:1px solid #eef2f8;font-size:13px">'
        f'<span><b>{i["lot_id"]}</b>'
        f'<span style="color:#94a3b8;margin-left:10px">{i["timestamp"]}</span></span>'
        f'<b style="color:#dc4c4c">{i["risk"]:.1%}</b></div>'
        for i in scan.get("high_risk_lots", [])
    )
    suggested = scan.get("suggested_lot")
    st.markdown(
        f"""
        <div style="border:1px solid #e2e8f2;border-radius:12px;overflow:hidden;background:#fff">
          <div style="background:#f7f9fc;padding:11px 14px;font-size:13px;font-weight:700">
            扫描最近 {scan.get('n_lots', 0)} 个批次，其中高风险
            <span style="color:#dc4c4c">{scan.get('n_high_risk', 0)}</span> 个
          </div>
          {rows}
        </div>
        {f'<div style="font-size:13px;color:#64748b;margin-top:10px">建议优先排查：<b style="color:#1f5fbf">{suggested}</b></div>' if suggested else ''}
        """,
        unsafe_allow_html=True,
    )


def render_banner(chips: list[str]) -> None:
    chip_html = "".join(f'<span class="zx-chip">{c}</span>' for c in chips)
    st.markdown(
        f"""
        <div class="zx-banner">
          <div class="zx-brandrow">
            <div class="zx-mark">🏭</div>
            <div class="zx-title">智维Agent</div>
            <div class="zx-live"><span class="zx-dot"></span>模型已加载 · 可交互</div>
          </div>
          <div class="zx-sub">从“设备报警”到“诊断—决策—工单—复盘”的完整闭环。
            数据来源：UCI SECOM 半导体前道工序数据集（1567 批次 × 591 路工艺信号，CC BY 4.0）。
            变量为匿名过程测量点，已按相关性聚为 12 个工序信号簇（工序站 A–L）。</div>
          <div class="zx-chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(THEME_CSS, unsafe_allow_html=True)


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


# ------------------------------------------------------------ 顶部横幅与指标
state = st.session_state.get("state", {})
config = st.session_state.get("config")
next_nodes = st.session_state.get("next", [])
risk = state.get("risk") or {}
orders = db.list_work_orders(limit=200)

_thr = metrics["thresholds"]["operational_threshold"] if metrics else 0.5
render_banner([
    f"模型：{metrics.get('model_kind', 'xgboost')} × {metrics.get('ensemble_size', 5)} 集成" if metrics else "模型：未加载",
    f"样本：{metrics.get('n_samples', 1567)} 批次 × {metrics.get('n_features', 468)} 路信号",
    f"异常批次：{metrics.get('n_positives', 104)}（{metrics.get('base_rate', 0.0664):.1%}）",
    f"PR-AUC：{metrics.get('cv', {}).get('pr_auc', 0):.3f}",
    f"Recall：{metrics.get('cv', {}).get('recall', 0):.3f}",
    f"F1：{metrics.get('cv', {}).get('f1', 0):.3f}",
    f"触发阈值：{_thr:.1%}",
])

# 还没跑诊断时，先用默认选中批次的信息填充 KPI，避免整排显示"—"
_fallback_id = state.get("lot_id") or toolbox.top_risk_lot(window=300)
_fb = lots.loc[lots["lot_id"] == _fallback_id]
_fb_risk = float(_fb["risk"].iloc[0]) if len(_fb) else 0.0
_fb_level = str(_fb["level"].iloc[0]) if len(_fb) else "—"
_cur_id = state.get("lot_id") or _fallback_id
_cur_risk = float(risk.get("risk", _fb_risk)) if risk else _fb_risk
_cur_level = str(risk.get("level", _fb_level)) if risk else _fb_level

_level = _cur_level
kpi_cards([
    ("当前批次", _cur_id, "#1f5fbf"),
    ("异常概率", f"{_cur_risk:.1%}",
     {"高": "#dc4c4c", "中": "#e2922f", "低": "#94a3b8"}.get(_level, "#1f5fbf")),
    ("风险等级", _level, {"高": "#dc4c4c", "中": "#e2922f", "低": "#94a3b8"}.get(_level, "#1f5fbf")),
    ("触发阈值", f"{_thr:.1%}", "#e2922f"),
    ("工单总数", f"{len(orders)} 条", "#2f9e6f"),
])

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
        _lv = str(lot_row["level"])
        _lvcolor = {"高": "#dc4c4c", "中": "#e2922f", "低": "#94a3b8"}.get(_lv, "#94a3b8")
        _badge = {"高": "zx-b-high", "中": "zx-b-mid", "低": "zx-b-low"}.get(_lv, "zx-b-low")
        _true = "异常" if lot_row["y_true"] == 1 else "正常"
        _truecolor = "#dc4c4c" if lot_row["y_true"] == 1 else "#2f9e6f"
        _raw = toolbox.X_raw.loc[int(lot_row["row_index"])]
        _z = ((_raw - toolbox.explainer.mean) / toolbox.explainer.std).reindex(
            toolbox.model.feature_names
        ).fillna(0.0)
        _top_feat = str(_z.abs().idxmax())
        _suspect = toolbox.model.preprocess.group_of(_top_feat)
        st.markdown(
            f"""
            <div class="zx-gauge">
              {gauge_svg(float(lot_row['risk']), float(_thr), _lv)}
              <div style="flex:1;min-width:190px">
                <div class="zx-risknum" style="color:{_lvcolor}">{lot_row['risk']:.1%}</div>
                <div style="margin-top:8px">
                  <span class="zx-badge {_badge}">风险等级 {_lv}</span>
                  <span class="zx-badge zx-b-low" style="margin-left:6px">触发阈值 {_thr:.1%}</span>
                </div>
                <div style="font-size:12.5px;color:#64748b;margin-top:8px">
                  袋外评分 {lot_row['risk']:.2%}　·　集成实时评分 {lot_row['risk_ensemble']:.2%}</div>
              </div>
            </div>
            <div style="font-size:12.5px;color:#64748b;line-height:1.9;margin-top:14px;
                        padding-top:12px;border-top:1px solid #e2e8f2">
              批次编号 <b>{lot_row['lot_id']}</b>　·　时间 {lot_row['timestamp']}<br>
                  疑似工序站 <b style="color:#1f5fbf">{_suspect}</b>
              　·　真实标签 <b style="color:{_truecolor}">{_true}</b>
              <span style="color:#94a3b8">（真实标签仅用于演示核对，智能体看不到）</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        raw_row = toolbox.X_raw.loc[int(lot_row["row_index"])]
        z = ((raw_row - toolbox.explainer.mean) / toolbox.explainer.std).reindex(
            toolbox.model.feature_names
        ).fillna(0.0)
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
        scan_result_block(state["scan"])


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
        _opts = decision["options"]
        _best_id = decision.get("recommended")
        _ocols = st.columns(len(_opts))
        for _col, _o in zip(_ocols, _opts):
            _is_best = _o["id"] == _best_id
            _col.markdown(
                f"""
                <div class="zx-opt {'best' if _is_best else ''}">
                  {'<span class="rb">推荐</span>' if _is_best else ''}
                  <div class="tag">方案 {_o['id']}</div>
                  <div class="nm">{_o['name']}</div>
                  <div class="cost">¥{_o['expected_cost']:,.0f}<small>期望代价</small></div>
                  <div class="dt">
                    质量损失<span>¥{_o['quality_loss']:,.0f}</span><br>
                    停机损失<span>¥{_o['downtime_cost']:,.0f}</span><br>
                    交期风险<span>¥{_o['delay_risk_cost']:,.0f}</span><br>
                    停机时长<span>{_o['downtime_hours']} h</span>
                  </div>
                  <div style="font-size:12px;color:#64748b;margin-top:9px">{_o['note']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        _worst = max(_opts, key=lambda o: o["expected_cost"])
        _best = next(o for o in _opts if o["id"] == _best_id)
        _a = decision.get("assumptions", {})
        _param_txt = "　·　".join([
            f"单批次产值 ¥{_a.get('batch_value', 0):,.0f}",
            f"停机损失 ¥{_a.get('downtime_cost_per_hour', 0):,.0f}/小时",
            f"检修工时 {_a.get('repair_hours', 0)} 小时",
            f"班次剩余 {_a.get('shift_hours', 0)} 小时",
            f"返工可挽回 {_a.get('rework_recovery_ratio', 0):.0%}",
            f"基准不良率 {_a.get('baseline_defect_rate', 0):.1%}",
            f"检修有效性 {_a.get('repair_effectiveness', 0):.0%}",
            f"延期罚金 ¥{_a.get('delay_penalty', 0):,.0f}",
            f"非计划停机成本上浮 {(_a.get('emergency_stop_multiplier', 1) - 1):.0%}",
        ])
        st.markdown(
            f"<div style='font-size:12.5px;color:#64748b;line-height:1.9;margin-top:14px'>"
            f"优选方案 <b style='color:#2f9e6f'>{_best['name']}</b>，比最差方案节省 "
            f"<b>¥{_worst['expected_cost'] - _best['expected_cost']:,.0f}</b>。"
            f"<br>模拟参数：{_param_txt}。<b>以上均为演示假设值，结果为模拟估算。</b></div>",
            unsafe_allow_html=True,
        )

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
            metric_table(metrics["cv"])
            confusion_table(metrics["cv"]["confusion"], "袋外预测")
        with c2:
            st.markdown("**漂移检查：按时间切分留出集**")
            metric_table(metrics["holdout"])
            confusion_table(metrics["holdout"]["confusion"], "时间留出集")
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
