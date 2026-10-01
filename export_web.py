"""把智能体导出成一个**单文件 HTML**（离线可用，双击即开，无需 Python）。

做法：把模型对全部 1567 个批次的推理结果（风险、SHAP 归因、信号偏差、
检索依据）在本地预计算好，连同知识库一起内嵌进一个 HTML 文件；
决策模拟、知识库检索、工单管理、执行轨迹这些逻辑用 JavaScript 在浏览器里跑。

用法：
    python export_web.py

输出：
    ../智维Agent-单文件版.html
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATE = PROJECT_ROOT / "web" / "template.html"
OUT_PATH = PROJECT_ROOT.parent / "智维Agent-单文件版.html"
TOP_SIGNALS = 12
TOP_SHAP = 10


def _load():
    from zhixi.data import load_processed
    from zhixi.model import RiskModel, load_lots, load_metrics
    from zhixi.rag import KnowledgeBase

    model = RiskModel.load()
    X_raw, y, ts, pre = load_processed()
    lots = load_lots()
    kb = KnowledgeBase()
    metrics = load_metrics()
    return model, X_raw, pre, lots, kb, metrics


def _shap_matrix(model, X: "object", chunk: int = 400) -> np.ndarray:
    """对全部批次批量计算 SHAP，返回 (n_samples, n_features)。多个基学习器取平均。"""

    import shap

    acc = np.zeros((len(X), X.shape[1]), dtype=float)
    for estimator in model.estimators:
        explainer = shap.TreeExplainer(estimator)
        for start in range(0, len(X), chunk):
            block = X.iloc[start : start + chunk]
            values = explainer.shap_values(block)
            if isinstance(values, list):
                values = values[1]
            acc[start : start + len(block)] += np.asarray(values).reshape(len(block), -1)
    return acc / max(len(model.estimators), 1)


def build_payload(verbose: bool = True) -> dict:
    model, X_raw, pre, lots, kb, metrics = _load()
    feature_names = list(model.feature_names)
    feature_index = {name: i for i, name in enumerate(feature_names)}
    groups = [pre.group_of(f) for f in feature_names]
    group_names = sorted(set(groups), key=lambda g: groups.index(g))
    group_index = {g: i for i, g in enumerate(group_names)}

    X = X_raw.loc[:, feature_names].fillna(pre.medians)
    mean = X_raw.loc[pre.train_index, feature_names].mean()
    std = X_raw.loc[pre.train_index, feature_names].std().replace(0.0, 1.0)
    Z = (X - mean) / std

    if verbose:
        print("  计算 SHAP（全部批次）…")
    shap_values = _shap_matrix(model, X)

    if verbose:
        print("  计算每批次的信号偏差、归因与检索依据…")

    docs: list[dict] = [
        {"doc": d["doc"], "section": d["section"], "text": d["text"]} for d in kb.docs
    ]
    doc_index: dict[tuple[str, str], int] = {
        (d["doc"], d["section"]): i for i, d in enumerate(docs)
    }

    def doc_id(doc: str, section: str) -> int:
        key = (doc, section)
        if key not in doc_index:
            doc_index[key] = len(docs)
            docs.append({"doc": doc, "section": section, "text": ""})
        return doc_index[key]

    lots_sorted = lots.sort_values("timestamp").reset_index(drop=True)
    out_lots = []
    for order, row in enumerate(lots_sorted.itertuples(index=False)):
        raw_i = int(row.row_index)
        z = Z.iloc[raw_i]
        s = shap_values[raw_i]

        top_sig = np.argsort(-np.abs(z.to_numpy()))[:TOP_SIGNALS]
        top_shap = np.argsort(-np.abs(s))[:TOP_SHAP]

        station_score: dict[int, float] = {}
        for fi in top_shap:
            if s[fi] > 0:
                gi = group_index[groups[fi]]
                station_score[gi] = station_score.get(gi, 0.0) + float(s[fi])
        station_rank = sorted(station_score.items(), key=lambda kv: -kv[1])
        suspected = station_rank[0][0] if station_rank else group_index[groups[top_shap[0]]]

        # 检索依据：与 Python 版一致，按多个「关注面」检索并合并
        second = group_names[station_rank[1][0]] if len(station_rank) > 1 else group_names[suspected]
        aspects = kb.aspect_queries(
            [group_names[suspected], second], high_risk=bool(row.risk >= float(metrics.get("thresholds", {}).get("operational_threshold", 0.5)))
        )
        hits = kb.search_aspects(aspects, k=3)
        evidence = [[doc_id(h.doc, h.section), round(float(h.score), 4)] for h in hits]

        out_lots.append(
            {
                "id": row.lot_id,
                "ts": str(row.timestamp),
                "risk": round(float(row.risk), 6),
                "riskEns": round(float(row.risk_ensemble), 6),
                "level": row.level,
                "y": int(row.y_true),
                "sig": [
                    [int(fi), round(float(z.iloc[fi]), 3), round(float(X.iloc[raw_i, fi]), 5)]
                    for fi in top_sig
                ],
                "shap": [[int(fi), round(float(s[fi]), 4)] for fi in top_shap],
                "suspected": int(suspected),
                "stations": [[int(gi), round(float(v), 4)] for gi, v in station_rank[:4]],
                "ev": evidence,
            }
        )

    payload = {
        "meta": {
            "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dataset": metrics.get("dataset", "UCI SECOM"),
            "model": f"{metrics.get('model_kind', 'xgboost')} × {metrics.get('ensemble_size', 5)}",
            "nSamples": metrics.get("n_samples", len(lots)),
            "nFeatures": metrics.get("n_features", len(feature_names)),
            "nPositives": metrics.get("n_positives", int(lots["y_true"].sum())),
            "baseRate": metrics.get("base_rate", float(lots["y_true"].mean())),
            "threshold": metrics.get("thresholds", {}).get("operational_threshold", 0.15),
            "highThreshold": metrics.get("thresholds", {}).get("high_threshold", 0.6),
            "cv": metrics.get("cv", {}),
            "holdout": metrics.get("holdout", {}),
            "business": {
                "batchValue": 8000.0,
                "downtimeCostPerHour": 3000.0,
                "repairHours": 1.5,
                "reworkRecovery": 0.6,
                "hoursPerLot": 0.5,
                "shiftHours": 8.0,
                "repairEffectiveness": 0.7,
                "delayPenalty": 12000.0,
                "urgency": 0.5,
                "emergencyStopMultiplier": 1.3,
            },
        },
        "features": feature_names,
        "groups": groups,
        "groupNames": group_names,
        "docs": docs,
        "lots": out_lots,
        "curves": {
            "pr": _curve_csv("pr_curve.csv", ["recall", "precision"], 120),
            "roc": _curve_csv("roc_curve.csv", ["fpr", "tpr"], 120),
        },
    }
    return payload


def _curve_csv(name: str, cols: list[str], points: int) -> list[list[float]]:
    import pandas as pd

    path = PROJECT_ROOT / "artifacts" / name
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if len(df) > points:
        df = df.iloc[np.linspace(0, len(df) - 1, points).astype(int)]
    return [[round(float(v), 5) for v in row] for row in df[cols].to_numpy()]


def main() -> None:
    print("① 预计算模型输出")
    payload = build_payload()

    print("② 生成单文件 HTML")
    html = TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = html.replace("/*__PAYLOAD__*/", data)
    OUT_PATH.write_text(html, encoding="utf-8")

    size = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n已生成：{OUT_PATH}")
    print(f"大小：{size:.2f} MB（含全部 {len(payload['lots'])} 个批次的推理结果与知识库）")
    print("双击即可在浏览器打开，无需 Python、无需联网。")


if __name__ == "__main__":
    main()
