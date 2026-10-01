"""生成参赛 PPT 用的两张图（SVG，可再渲染成高清 PNG）。

    ① 分析路径思维导图：从业务问题推导到技术方案
    ② 数据与知识库预处理流程图：两条链路如何汇入智能体工具箱

用法：
    python make_diagrams.py

输出目录：../智维Agent-图表/
配色与界面一致：主色 #1f5fbf，风险色阶 灰 → 橙 → 红。
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT.parent / "智维Agent-图表"

W, H = 1600, 900
FONT = "'Microsoft YaHei','PingFang SC','Hiragino Sans GB',system-ui,sans-serif"

BLUE = "#1f5fbf"
BLUE_DARK = "#143f83"
BLUE_LIGHT = "#eaf1fd"
INK = "#1b1f24"
MUTED = "#6b7684"
LINE = "#d7dde6"
RED = "#d64545"
ORANGE = "#e8a33d"
GREEN = "#2f9e6f"
GRAY = "#5b6675"
CYAN = "#2a7fd4"
SLATE = "#4c78a8"


def head(title: str, subtitle: str) -> str:
    return (
        f'<text x="60" y="56" font-family="{FONT}" font-size="30" font-weight="700" '
        f'fill="{INK}">{title}</text>'
        f'<text x="60" y="88" font-family="{FONT}" font-size="15" fill="{MUTED}">{subtitle}</text>'
        f'<line x1="60" y1="106" x2="{W - 60}" y2="106" stroke="{LINE}" stroke-width="1"/>'
    )


def rect(x, y, w, h, fill, stroke="none", rx=10, sw=1.5, op=1.0) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>'
    )


def text(x, y, content, size=14, fill=INK, anchor="start", weight="400", lh=None) -> str:
    lines = content.split("\n")
    lh = lh or size * 1.38
    total = (len(lines) - 1) * lh
    out = []
    for i, ln in enumerate(lines):
        yy = y - total / 2 + i * lh + size * 0.35
        out.append(
            f'<text x="{x}" y="{yy:.1f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{ln}</text>'
        )
    return "".join(out)


def bezier(x1, y1, x2, y2, color=LINE, sw=1.6, dash="") -> str:
    mx = (x1 + x2) / 2
    d = f"M {x1},{y1} C {mx},{y1} {mx},{y2} {x2},{y2}"
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{dash_attr}/>'


def arrow_down(x, y1, y2, color=LINE, sw=1.8) -> str:
    return (
        f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 7}" stroke="{color}" stroke-width="{sw}"/>'
        f'<path d="M {x - 5},{y2 - 8} L {x},{y2} L {x + 5},{y2 - 8} Z" fill="{color}"/>'
    )


def arrow_right(x1, x2, y, color=LINE, sw=1.8) -> str:
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 7}" y2="{y}" stroke="{color}" stroke-width="{sw}"/>'
        f'<path d="M {x2 - 8},{y - 5} L {x2},{y} L {x2 - 8},{y + 5} Z" fill="{color}"/>'
    )


_TOKEN = re.compile(r"[A-Za-z0-9_.\-+/×≥→()]+|.")


def wrap_text(text: str, max_units: float, max_lines: int = 3) -> list[str]:
    """按显示宽度折行：中文按 1 个单位、ASCII 按 0.55 个单位，且不在英文单词中间断开。"""

    def width(tok: str) -> float:
        return sum(0.55 if ord(c) < 128 else 1.0 for c in tok)

    lines: list[str] = []
    cur, cur_w = "", 0.0
    for tok in _TOKEN.findall(text):
        w = width(tok)
        if cur and cur_w + w > max_units:
            lines.append(cur)
            cur, cur_w = tok, w
            if len(lines) == max_lines:
                break
        else:
            cur += tok
            cur_w += w
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and sum(len(x) for x in lines) < len(text):
        lines[-1] = lines[-1][:-1] + "…"
    return lines


# ---------------------------------------------------------------- 图一：思维导图
BRANCHES = [
    ("业务问题", GRAY, ["维保依赖老师傅经验", "预警与维修执行脱节", "停机决策缺少量化依据"]),
    ("数据基础", CYAN, ["SECOM 1567 × 591 路信号", "异常批次 104（6.64%）", "缺失值 43518 个", "变量匿名，无物理含义"]),
    ("特征加工", BLUE, ["剔常量列 591 → 468", "中位数填补缺失值", "相关性聚类 → 12 工序站", "按时间 7:3 切分"]),
    ("风险模型", RED, ["XGBoost × 5 自助集成", "袋外 F1 阈值 15%", "PR-AUC 0.175", "Recall 0.337 / F1 0.250"]),
    ("可解释归因", ORANGE, ["SHAP 归因（5 模型平均）", "单批次贡献值排序", "归因到工序站 A–L"]),
    ("知识检索", GREEN, ["6 篇维修 SOP", "分节切片 → 30 条", "字符 n-gram 检索", "回答必须标注出处"]),
    ("Agent 编排", BLUE_DARK, ["6 个真实 Python 工具", "风险高低条件分支", "人工确认 interrupt", "失败重试 + 检查点"]),
    ("决策闭环", SLATE, ["A/B/C 期望代价模拟", "参数可调、结论实时变", "工单落库 SQLite", "复测 → 复盘记录"]),
]


def mindmap() -> str:
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    s.append(rect(0, 0, W, H, "#ffffff", rx=0))
    s.append(head("分析路径思维导图 · 智维Agent",
                  "从业务问题出发，每一步推导都有数据或指标支撑；沿箭头自左向右阅读"))

    top, bottom = 152, 862
    n = len(BRANCHES)
    row_h = (bottom - top) / n
    box_h = 54
    root_x, root_w = 60, 168
    br_x, br_w = 296, 214
    chip_x, chip_w, chip_gap = 548, 232, 20
    root_cy = (top + bottom) / 2

    # 根节点
    s.append(rect(root_x, root_cy - 56, root_w, 112, BLUE, rx=16))
    s.append(text(root_x + root_w / 2, root_cy - 22, "设备维保", 19, "#ffffff", "middle", "700"))
    s.append(text(root_x + root_w / 2, root_cy + 6, "决策问题", 19, "#ffffff", "middle", "700"))
    s.append(text(root_x + root_w / 2, root_cy + 38, "智维Agent", 12, "#cfe0ff", "middle", "500"))

    for i, (name, color, leaves) in enumerate(BRANCHES):
        cy = top + row_h * i + row_h / 2
        by = cy - box_h / 2
        s.append(bezier(root_x + root_w, root_cy, br_x, cy, color, 1.5))
        s.append(rect(br_x, by, br_w, box_h, "#ffffff", color, rx=10))
        s.append(rect(br_x, by, 5, box_h, color, rx=3))
        s.append(text(br_x + 20, cy, name, 15.5, color, "start", "700"))

        for j, leaf in enumerate(leaves):
            lx = chip_x + j * (chip_w + chip_gap)
            s.append(bezier(br_x + br_w, cy, lx, cy, LINE, 1.2))
            s.append(rect(lx, cy - 18, chip_w, 36, "#fbfcfe", LINE, rx=8, sw=1))
            s.append(text(lx + 14, cy, leaf, 12.5, "#37414f"))

    s.append(text(60, 892, "数据来源：UCI SECOM（CC BY 4.0）｜模型与指标均为可复现的真实运行结果",
                  12, "#9aa4b1"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- 图二：流程图
DATA_STEPS = [
    "SECOM 原始数据\n1567 批次 × 591 路信号，含缺失值",
    "按时间排序，剔除常量列\n有效信号 591 → 468 路",
    "用训练集中位数填补缺失\n共处理 43518 个空值",
    "相关性层次聚类\n归并为 12 个工序信号簇（工序站 A–L）",
    "按时间 7:3 切分\n训练 XGBoost × 5 自助采样集成",
    "袋外 F1 最优阈值 15%\n输出 1567 个批次的风险评分",
]

KB_STEPS = [
    "维修知识库 · 6 篇 SOP\n维护总则 / 点检 SOP / 漂移排查 / 停机安全 / 批次追溯 / 备件工单",
    "按二级标题切片\n得到 30 条可检索的知识片段",
    "字符 2–4 gram 向量化\nTF-IDF 索引；装 faiss 后自动升级为向量检索",
    "多关注面检索\n拆成「怎么查 · 怎么修 · 要不要停」三条子查询，各取最佳命中后合并去重",
]

PIPE_STEPS = [
    ("智能体工具箱", "6 个真实 Python 函数\n查信号 / 预测风险 / 归因 / 检索 / 决策模拟 / 工单", BLUE),
    ("LangGraph 编排", "条件分支 · 人工确认(interrupt)\n失败重试 · 检查点持久化", BLUE_DARK),
    ("维保决策", "A/B/C 期望代价模拟\n业务参数可调，推荐结论实时变化", ORANGE),
    ("执行与复盘", "工单落库 SQLite\n复测 → 自动生成复盘记录", GREEN),
]


def flowchart() -> str:
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    s.append(rect(0, 0, W, H, "#ffffff", rx=0))
    s.append(head("数据与知识库预处理流程图 · 智维Agent",
                  "两条链路各自加工，在智能体工具箱汇合；模型负责“检测”，知识库负责“依据”，智能体负责“编排与执行”"))

    col_h, gap = 62, 26

    # 左列：数据链路
    lx, lw = 60, 596
    s.append(rect(lx, 128, lw, 34, BLUE_LIGHT, BLUE, rx=8, sw=1.2))
    s.append(text(lx + 16, 145, "① 数据链路 · 让模型能算", 14, BLUE_DARK, "start", "700"))
    y = 178
    for i, step in enumerate(DATA_STEPS):
        lines = step.split("\n")
        s.append(rect(lx, y, lw, col_h, "#ffffff", LINE, rx=10))
        s.append(rect(lx, y, 5, col_h, BLUE, rx=3))
        s.append(f'<circle cx="{lx + 34}" cy="{y + col_h / 2}" r="13" fill="{BLUE}"/>')
        s.append(text(lx + 34, y + col_h / 2, str(i + 1), 12.5, "#ffffff", "middle", "700"))
        s.append(text(lx + 60, y + col_h / 2 - 10, lines[0], 14.5, INK, "start", "700"))
        s.append(text(lx + 60, y + col_h / 2 + 13, lines[1], 12, MUTED))
        if i < len(DATA_STEPS) - 1:
            s.append(arrow_down(lx + lw / 2, y + col_h, y + col_h + gap, BLUE, 1.6))
        y += col_h + gap
    data_bottom = y - gap

    # 右列：知识链路
    rx_, rw = 860, 680
    s.append(rect(rx_, 128, rw, 34, "#e8f7f0", GREEN, rx=8, sw=1.2))
    s.append(text(rx_ + 16, 145, "② 知识链路 · 让答案有据可查", 14, "#1c6b4b", "start", "700"))
    y2 = 178
    for i, step in enumerate(KB_STEPS):
        lines = step.split("\n")
        s.append(rect(rx_, y2, rw, col_h, "#ffffff", LINE, rx=10))
        s.append(rect(rx_, y2, 5, col_h, GREEN, rx=3))
        s.append(f'<circle cx="{rx_ + 34}" cy="{y2 + col_h / 2}" r="13" fill="{GREEN}"/>')
        s.append(text(rx_ + 34, y2 + col_h / 2, str(i + 1), 12.5, "#ffffff", "middle", "700"))
        s.append(text(rx_ + 60, y2 + col_h / 2 - 10, lines[0], 14.5, INK, "start", "700"))
        s.append(text(rx_ + 60, y2 + col_h / 2 + 13, lines[1], 12, MUTED))
        if i < len(KB_STEPS) - 1:
            s.append(arrow_down(rx_ + rw / 2, y2 + col_h, y2 + col_h + gap, GREEN, 1.6))
        y2 += col_h + gap
    kb_bottom = y2 - gap

    # 汇合
    merge_y = max(data_bottom, kb_bottom) + 46
    s.append(bezier(lx + lw / 2, data_bottom, W / 2 - 90, merge_y, BLUE, 1.6))
    s.append(bezier(rx_ + rw / 2, kb_bottom, W / 2 + 90, merge_y, GREEN, 1.6))
    s.append(rect(W / 2 - 118, merge_y - 20, 236, 40, "#f7f9fc", BLUE, rx=20, sw=1.4))
    s.append(text(W / 2, merge_y, "汇合 → 交给同一个智能体", 13.5, BLUE_DARK, "middle", "700"))

    # 底部：智能体闭环
    by = merge_y + 40
    bw, bgap = 355, 20
    for i, (title, desc, color) in enumerate(PIPE_STEPS):
        bx = 60 + i * (bw + bgap)
        s.append(rect(bx, by, bw, 84, "#ffffff", color, rx=12, sw=1.6))
        s.append(rect(bx, by, bw, 26, color, rx=12, op=0.10))
        s.append(text(bx + 16, by + 16, f"{i + 1}. {title}", 14, color, "start", "700"))
        for k, ln in enumerate(desc.split("\n")):
            s.append(text(bx + 16, by + 46 + k * 17, ln, 11.5, MUTED))
        if i < len(PIPE_STEPS) - 1:
            s.append(arrow_right(bx + bw, bx + bw + bgap, by + 42, color, 2))

    s.append(text(60, 884,
                  "全流程 22 项自动化自检；发现时间漂移后采用滚动重训 + 阈值在线校准 + 人工确认兜底",
                  12, "#9aa4b1"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- 图三：功能架构
LAYERS = [
    ("交互层", "用户看到与操作的部分", CYAN, [
        ("Streamlit 界面", "监控诊断 / 执行轨迹 / 决策工单 / 知识库 / 模型评估"),
        ("命令行演示", "demo.py：一条命令跑完整闭环，录视频用"),
        ("单文件 HTML 版", "离线可用，双击即开，评委快速看效果"),
        ("访问口令门", "可选开启，满足提交账号密码的要求"),
    ]),
    ("编排层", "LangGraph 主图，16 节点 / 30 条条件边", BLUE_DARK, [
        ("意图解析", "识别诊断 / 巡检 / 检索，定位目标批次"),
        ("条件分支", "风险高低分流、审批通过与拒绝分流"),
        ("人工确认", "interrupt 暂停，Command(resume) 恢复"),
        ("失败重试", "工具异常 → retry → 回原节点，超限转人工"),
        ("检查点与轨迹", "MemorySaver 保存状态；每步留痕可导出"),
    ]),
    ("能力层", "6 个真实可调用的 Python 工具", BLUE, [
        ("查信号", "get_sensor_data"),
        ("预测风险", "predict_failure"),
        ("归因解释", "explain_failure"),
        ("检索依据", "search_maintenance_manual"),
        ("决策模拟", "simulate_maintenance_decision"),
        ("工单落库", "create_work_order"),
    ]),
    ("模型层", "检测与解释，不做业务判断", RED, [
        ("风险模型", "XGBoost × 5 自助采样集成，输出异常概率"),
        ("阈值校准", "袋外 F1 最优阈值 15%，分位数对齐"),
        ("可解释性", "SHAP 归因，给出单批次信号贡献值"),
        ("特征加工", "常量列剔除、中位数填补、12 工序站聚类"),
    ]),
    ("数据与知识层", "一切结论的事实来源", GREEN, [
        ("SECOM 数据集", "1567 批次 × 591 路信号，CC BY 4.0"),
        ("批次风险表", "1567 条评分，供巡检与回放"),
        ("维修知识库", "6 篇 SOP，30 条可检索片段"),
        ("工单数据库", "SQLite，含工单全生命周期字段"),
    ]),
]


def layered_architecture() -> str:
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    s.append(rect(0, 0, W, H, "#ffffff", rx=0))
    s.append(head("智维Agent 功能架构 · 五层分层设计",
                  "上层调用下层，下层只提供事实；模型负责“检测”，知识库负责“依据”，编排层负责“决策与执行”"))

    y = 132
    band_h, gap = 112, 32
    label_x, label_w = 60, 158
    mod_x = label_x + label_w + 22
    mod_w_total = W - mod_x - 60

    for li, (name, desc, color, modules) in enumerate(LAYERS):
        s.append(rect(label_x, y, label_w, band_h, color, rx=12))
        s.append(text(label_x + label_w / 2, y + band_h / 2 - 14, name, 19, "#ffffff", "middle", "700"))
        s.append(text(label_x + label_w / 2, y + band_h / 2 + 16, desc.split("，")[0], 10.5, "#e6eeff", "middle", "400"))

        n = len(modules)
        g = 16
        bw = (mod_w_total - (n - 1) * g) / n
        for mi, (title, body) in enumerate(modules):
            bx = mod_x + mi * (bw + g)
            s.append(rect(bx, y, bw, band_h, "#ffffff", LINE, rx=10, sw=1.2))
            s.append(rect(bx, y, bw, 4, color, rx=2))
            s.append(text(bx + 14, y + 34, title, 14.5, color, "start", "700"))
            # 正文按长度自动折行
            lines = wrap_text(body, max_units=(bw - 26) / 11.4, max_lines=3)
            for k, ln in enumerate(lines):
                s.append(text(bx + 14, y + 58 + k * 16, ln, 10.5, MUTED))
        if li < len(LAYERS) - 1:
            s.append(f'<path d="M 800,{y + band_h + 3} L 810,{y + band_h + 11} L 800,{y + band_h + 19} '
                     f'L 790,{y + band_h + 11} Z" fill="{LINE}"/>')
        y += band_h + gap

    s.append(text(60, 886, "代码位置：zhixi/graph.py（编排层）· zhixi/tools.py（能力层）· zhixi/model.py + explain.py（模型层）"
                           "· zhixi/data.py + rag.py + db.py（数据与知识层）", 11.5, "#9aa4b1"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- 图四：LangGraph
MAIN_ROW1 = [("START", "入口", GRAY), ("parse_intent", "解析意图", CYAN),
             ("fetch_sensor", "读工艺信号", BLUE), ("predict_failure", "风险预测", BLUE),
             ("route_risk", "风险分支", RED), ("explain_failure", "SHAP 归因", ORANGE),
             ("retrieve_manual", "检索依据", GREEN)]
MAIN_ROW2 = [("simulate_decision", "三方案模拟", ORANGE), ("propose", "形成建议", BLUE),
             ("human_review", "人工确认 interrupt", RED), ("create_work_order", "工单落库", GREEN),
             ("finalize", "生成结论", BLUE_DARK), ("END", "结束", GRAY)]
SIDE_NODES = [("scan_stream", "批次巡检分支：扫描最近 300 批，给出优先排查对象"),
              ("low_risk_report", "低风险分支：不触发决策，直接放行并纳入趋势观察"),
              ("handover_human", "拒绝分支：工程师不确认时不派工，转人工处理"),
              ("retry", "失败重试：任意工具异常回到原节点，最多 2 次"),
              ("handle_error", "异常终止：重试超限后停止自动执行并转人工")]


def langgraph_map() -> str:
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    s.append(rect(0, 0, W, H, "#ffffff", rx=0))
    s.append(head("LangGraph 编排图 · 16 个节点 / 30 条条件边",
                  "实线为主干链路，虚线为分支与容错路径；每个工具节点后都挂着一个重试守卫"))

    bx0, bw, g = 50, 198, 18
    row_h = 84

    for row, (items, y) in enumerate([(MAIN_ROW1, 148), (MAIN_ROW2, 296)]):
        for i, (name, label, color) in enumerate(items):
            bx = bx0 + i * (bw + g)
            s.append(rect(bx, y, bw, row_h, "#ffffff", color, rx=12, sw=1.6))
            s.append(text(bx + bw / 2, y + 28, name, 13, color, "middle", "700"))
            s.append(text(bx + bw / 2, y + 52, label, 11.5, MUTED, "middle"))
            if i < len(items) - 1:
                s.append(arrow_right(bx + bw, bx + bw + g, y + row_h / 2, color, 1.8))
        s.append(text(bx0, y - 12, f"主干第 {row + 1} 段", 11.5, "#9aa4b1"))

    # 第一段末尾回到第二段开头
    end_x = bx0 + len(MAIN_ROW1) * (bw + g) - g
    s.append(
        f'<path d="M {end_x},{148 + row_h} L {end_x},{252} L {bx0 + bw / 2},{252} '
        f'L {bx0 + bw / 2},{296}" fill="none" stroke="{BLUE}" '
        f'stroke-width="1.8" stroke-dasharray="6 4"/>'
    )
    s.append(f'<path d="M {bx0 + bw / 2 - 5},{288} L {bx0 + bw / 2},{296} L {bx0 + bw / 2 + 5},{288} Z" fill="{BLUE}"/>')

    # 旁路节点
    sy = 430
    pad = 22
    width = (W - 120 - (len(SIDE_NODES) - 1) * g) / len(SIDE_NODES)
    for i, (name, desc) in enumerate(SIDE_NODES):
        sxx = 60 + i * (width + g)
        s.append(rect(sxx, sy, width, 68, "#fbfcfe", LINE, rx=10, sw=1.2))
        s.append(rect(sxx, sy, 4, 68, GRAY, rx=2))
        s.append(text(sxx + 14, sy + 24, name, 12.5, INK, "start", "700"))
        for k, ln in enumerate(wrap_text(desc, max_units=(width - 26) / 10.4, max_lines=3)):
            s.append(text(sxx + 14, sy + 44 + k * 14, ln, 10, MUTED))
    s.append(text(60, sy - 12, "旁路与容错节点", 11.5, "#9aa4b1"))

    # 说明面板
    py = 540
    panels = [
        ("3 类条件分支", BLUE, [
            "① 风险分流：预测概率 ≥ 15% 走完整决策链路；",
            "    低于阈值直接放行，不生成工单。",
            "② 审批分流：工程师确认 → 生成工单；",
            "    拒绝 → 转人工接管，同样留痕。",
            "③ 意图分流：单批次诊断 / 批量巡检。",
        ]),
        ("失败重试机制", ORANGE, [
            "• 每个工具节点后接一个 guard 守卫；",
            "• 工具抛异常 → 进入 retry 节点计数 +1 →",
            "  回到原节点重新执行；",
            "• 超过 2 次仍未成功 → handle_error，",
            "  停止自动执行并转人工，不静默失败。",
        ]),
        ("状态与可验证性", GREEN, [
            "• AgentState 共 22 个字段，贯穿全流程；",
            "• MemorySaver 检查点，按 thread_id 恢复；",
            "• 每步记录工具名、参数、返回值、耗时、",
            "  第几次调用，可导出 JSON；",
            "• 22 项自动化自检覆盖全部分支。",
        ]),
    ]
    pw = (W - 120 - 2 * 20) / 3
    for i, (title, color, lines) in enumerate(panels):
        px = 60 + i * (pw + 20)
        s.append(rect(px, py, pw, 212, "#ffffff", LINE, rx=12, sw=1.2))
        s.append(rect(px, py, pw, 5, color, rx=2))
        s.append(text(px + 18, py + 32, title, 14.5, color, "start", "700"))
        for k, ln in enumerate(lines):
            s.append(text(px + 18, py + 62 + k * 22, ln, 11.5, "#37414f"))

    s.append(text(60, 886, "上图由 zhixi/graph.py 的实际编译结果导出：16 个节点、37 条边（其中 30 条为条件边）",
                  11.5, "#9aa4b1"))
    s.append("</svg>")
    return "".join(s)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, svg in [
        ("01-分析路径思维导图.svg", mindmap()),
        ("02-数据与知识库预处理流程图.svg", flowchart()),
        ("03-功能架构分层图.svg", layered_architecture()),
        ("04-LangGraph编排图.svg", langgraph_map()),
    ]:
        p = OUT_DIR / name
        p.write_text(svg, encoding="utf-8")
        print(f"已生成 {p}  ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
