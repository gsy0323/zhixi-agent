"""全局配置：路径、业务参数、智能体参数。"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
TRACE_DIR = ARTIFACTS_DIR / "traces"

for _d in (DATA_DIR, RAW_DIR, ARTIFACTS_DIR, TRACE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "risk_model.joblib"
PREPROCESS_PATH = ARTIFACTS_DIR / "preprocess.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
# 部署到云平台时可以用环境变量 ZHIXI_DB_PATH 指向可写的持久化目录
DB_PATH = Path(os.getenv("ZHIXI_DB_PATH", str(ARTIFACTS_DIR / "workorders.db")))

# SECOM 官方下载地址（CC BY 4.0）
SECOM_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"


@dataclass
class BusinessParams:
    """业务参数。全部为可在界面上调整的假设值，用于“模拟决策”，不代表真实企业数据。"""

    batch_value: float = 8000.0          # 单批次产值(元)
    downtime_cost_per_hour: float = 3000.0  # 停机损失(元/小时)
    repair_hours: float = 1.5            # 计划检修工时(小时)
    rework_recovery_ratio: float = 0.6   # 异常批次返工可挽回比例
    hours_per_lot: float = 0.5           # 一个批次的生产节拍(小时)
    shift_hours: float = 8.0             # 当前班次剩余时间(小时)
    # 立即停机属于「非计划停机」，需紧急调配备件与维修人力，实际成本高于可计划的停机。
    # 工业实践里非计划停机成本通常比计划停机高 20%~40%，这里取 1.3。
    emergency_stop_multiplier: float = 1.3


@dataclass
class AgentParams:
    """智能体与检索参数。"""

    risk_threshold: float = 0.5   # 触发维修决策的风险阈值（训练后由模型指标覆盖）
    max_retries: int = 2          # 工具失败重试次数
    top_k_features: int = 8       # 解释时展示的关键信号数
    top_k_evidence: int = 3       # RAG 返回的维修依据条数


def llm_settings() -> dict:
    """读取 LLM 配置。未配置时智能体自动退化为离线规则模式，全流程仍可运行。"""

    return {
        "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "",
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "model": os.getenv("ZHIXI_LLM_MODEL", "qwen-plus"),
    }


def llm_enabled() -> bool:
    return bool(llm_settings()["api_key"])


def business_as_dict(p: BusinessParams | None = None) -> dict:
    return asdict(p or BusinessParams())
