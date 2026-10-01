"""风险模型：训练、评估、加载、预测。

SECOM 的异常批次只有 104 条（约 6.6%），属于典型不平衡工业数据，
因此本项目不把 Accuracy 当作主指标，重点报告 Recall / F1 / PR-AUC。

两个关键设计：
1. **上线模型用自助采样集成（bagging）**：单个 XGBoost 在全量数据上重训会严重过拟合，
   历史批次评分几乎退化成 0/1 两档。改用 5 个自助采样模型取平均后，
   评分分布平滑，也更接近"新批次"的真实表现。
2. **阈值来自袋外预测**：用分层 5 折的袋外概率选 F1 最优阈值，
   再对上线集成模型的评分做一次分位数对齐，保证"报警比例"与离线评估一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, METRICS_PATH, MODEL_PATH, PREPROCESS_PATH, RAW_DIR
from .data import Preprocess, build_preprocess, load_secom

LOTS_PATH = ARTIFACTS_DIR / "lots.csv"
PR_CURVE_PATH = ARTIFACTS_DIR / "pr_curve.csv"
ROC_CURVE_PATH = ARTIFACTS_DIR / "roc_curve.csv"

# 正则强度由 experiments/tune_model.py 的对比实验选出
BASE_KWARGS = dict(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.3,
    min_child_weight=5,
    reg_lambda=2.0,
)
N_ENSEMBLE = 5


def make_estimator(scale_pos_weight: float, seed: int = 42, **overrides):
    """优先使用 XGBoost，未安装时退化为 sklearn 的直方图梯度提升。"""

    kwargs = {**BASE_KWARGS, **overrides}
    try:
        from xgboost import XGBClassifier

        return (
            XGBClassifier(
                eval_metric="aucpr",
                tree_method="hist",
                random_state=seed,
                n_jobs=4,
                scale_pos_weight=scale_pos_weight,
                **kwargs,
            ),
            "xgboost",
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return (
            HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=seed,
                learning_rate=kwargs["learning_rate"],
                max_depth=kwargs["max_depth"],
                max_iter=kwargs["n_estimators"],
            ),
            "hist-gbdt",
        )


# --------------------------------------------------------------------- 集成
def fit_ensemble(X: pd.DataFrame, y: pd.Series, spw: float, seed: int = 42,
                 n_models: int = N_ENSEMBLE, subsample: float = 0.8) -> list:
    """训练 n 个自助采样模型，返回模型列表。"""

    rng = np.random.default_rng(seed)
    n = len(X)
    models = []
    for i in range(n_models):
        idx = rng.choice(n, size=int(n * subsample), replace=False)
        est, _ = make_estimator(spw, seed + i)
        est.fit(X.iloc[idx], y.iloc[idx])
        models.append(est)
    return models


def ensemble_proba(models: list, X: pd.DataFrame) -> np.ndarray:
    if not models:
        raise ValueError("集成模型为空")
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


# --------------------------------------------------------------------- 指标
def _metrics(y_true, prob, threshold: float) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": round(float(average_precision_score(y_true, prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": round(float(threshold), 4),
        "alarm_rate": round(float(pred.mean()), 4),
    }


def _tune_threshold(y_true, prob) -> tuple[float, dict]:
    """在袋外预测上选 F1 最优阈值。"""

    from sklearn.metrics import f1_score

    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.02, 0.91, 0.01):
        f1 = f1_score(y_true, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, _metrics(y_true, prob, best_t)


# --------------------------------------------------------------------- 训练
def train(
    raw_dir: Path | str = RAW_DIR,
    *,
    seed: int = 42,
    n_splits: int = 5,
    test_ratio: float = 0.3,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """训练并评估风险模型，返回指标字典。"""

    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.model_selection import StratifiedKFold

    X_raw, y, ts = load_secom(raw_dir)
    pre = build_preprocess(X_raw, y, ts, test_ratio=test_ratio, seed=seed)
    X = pre.transform(X_raw)

    pos = int(y.sum())
    neg = int(len(y) - pos)
    spw = neg / max(pos, 1)

    # (1) 全量分层 5 折：主指标与阈值都来自袋外预测
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X, y):
        est, kind = make_estimator(spw, seed)
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = est.predict_proba(X.iloc[va_idx])[:, 1]
    cv_threshold, cv_metrics = _tune_threshold(y.to_numpy(), oof)

    # (2) 时间留出集：用历史批次训练的集成预测未来批次，检查时间漂移
    X_tr, y_tr = X.loc[pre.train_index], y.loc[pre.train_index]
    X_te, y_te = X.loc[pre.test_index], y.loc[pre.test_index]
    ens_tr = fit_ensemble(X_tr, y_tr, spw, seed)
    prob_te = ensemble_proba(ens_tr, X_te)
    holdout = _metrics(y_te.to_numpy(), prob_te, cv_threshold)

    # (3) 上线集成模型：全量数据上的自助采样集成
    ens_full = fit_ensemble(X, y, spw, seed)
    risk_ens = ensemble_proba(ens_full, X)

    # (4) 两套阈值
    #   threshold           → 历史批次的袋外评分（与离线评估同尺度）
    #   ensemble_threshold  → 新批次的集成评分（按报警比例做分位数对齐）
    alarm_rate = float(cv_metrics["alarm_rate"])
    op_threshold = float(cv_threshold)
    ensemble_threshold = float(np.quantile(risk_ens, 1 - alarm_rate))
    if float((risk_ens >= op_threshold).mean()) >= 0.5 * alarm_rate:
        ensemble_threshold = op_threshold
    high_threshold = float(np.quantile(oof, 0.95))

    precision, recall, pr_t = precision_recall_curve(y_te, prob_te)
    fpr, tpr, roc_t = roc_curve(y_te, prob_te)

    lots = (
        pd.DataFrame(
            {
                "row_index": list(range(len(X))),
                "timestamp": ts.to_numpy(),
                "y_true": y.to_numpy(),
                # 历史批次一律用袋外评分，避免样本内乐观偏差
                "risk": oof,
                "risk_ensemble": risk_ens,
                "split": [
                    "train" if i in set(pre.train_index) else "holdout" for i in range(len(X))
                ],
            }
        )
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    lots["lot_id"] = [f"LOT-{i + 1:04d}" for i in range(len(lots))]
    lots["level"] = np.where(
        lots["risk"] >= high_threshold, "高", np.where(lots["risk"] >= op_threshold, "中", "低")
    )

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "UCI SECOM (id=179) · 1567 样本 × 591 变量 · CC BY 4.0",
        "model_kind": kind,
        "ensemble_size": N_ENSEMBLE,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_positives": int(y.sum()),
        "base_rate": round(float(y.mean()), 4),
        "cv": cv_metrics,
        "holdout": holdout,
        "thresholds": {
            "cv_threshold": round(cv_threshold, 4),
            "operational_threshold": round(op_threshold, 4),
            "ensemble_threshold": round(ensemble_threshold, 4),
            "high_threshold": round(high_threshold, 4),
            "deployed_alarm_rate": round(float((oof >= op_threshold).mean()), 4),
        },
        "protocol": {
            "primary": "stratified 5-fold cross validation（袋外预测）",
            "drift_check": "time-ordered 70/30 holdout",
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "n_pos_train": int(y_tr.sum()),
            "n_pos_test": int(y_te.sum()),
        },
    }

    if save:
        import joblib

        payload = {
            "estimators": ens_full,
            "feature_names": list(X.columns),
            "threshold": op_threshold,
            "ensemble_threshold": ensemble_threshold,
            "high_threshold": high_threshold,
            "kind": kind,
        }
        joblib.dump(payload, MODEL_PATH)
        pre.save(PREPROCESS_PATH)
        METRICS_PATH.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        lots.to_csv(LOTS_PATH, index=False)
        pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "threshold": list(pr_t) + [1.0],
            }
        ).to_csv(PR_CURVE_PATH, index=False)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_t}).to_csv(
            ROC_CURVE_PATH, index=False
        )

    if verbose:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


# --------------------------------------------------------------------- 推理
@dataclass
class RiskModel:
    """上线推理用的风险模型封装（自助采样集成）。"""

    estimators: list
    feature_names: list[str]
    threshold: float
    kind: str
    preprocess: Preprocess
    high_threshold: float = 0.6
    cv_threshold: float = 0.5
    ensemble_threshold: float = 0.5
    _cache: dict = field(default_factory=dict)

    @staticmethod
    def load(
        model_path: Path | str = MODEL_PATH, preprocess_path: Path | str = PREPROCESS_PATH
    ) -> "RiskModel":
        import joblib

        payload = joblib.load(model_path)
        pre = Preprocess.load(preprocess_path)
        return RiskModel(
            estimators=payload["estimators"],
            feature_names=payload["feature_names"],
            threshold=float(payload["threshold"]),
            high_threshold=float(payload.get("high_threshold", 0.6)),
            cv_threshold=float(payload.get("cv_threshold", payload["threshold"])),
            ensemble_threshold=float(payload.get("ensemble_threshold", payload["threshold"])),
            kind=payload.get("kind", "unknown"),
            preprocess=pre,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if set(self.feature_names) <= set(X.columns):
            Xp = X.loc[:, self.feature_names]
        else:
            Xp = self.preprocess.transform(X)
        return ensemble_proba(self.estimators, Xp)

    def risk_for_row(self, X_raw: pd.DataFrame, row_index: int) -> float:
        return float(self.predict_proba(X_raw.iloc[[row_index]])[0])


def load_metrics(path: Path | str = METRICS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_lots(path: Path | str = LOTS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("尚未训练模型，请先运行 `python train_model.py`")
    return pd.read_csv(path, parse_dates=["timestamp"])
