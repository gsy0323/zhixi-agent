"""模型调参对比：不同正则强度下 XGBoost 在 SECOM 上的表现。

用法：
    python experiments/tune_model.py

结果写入 artifacts/tuning_results.csv，用于给答辩提供"超参是怎么选的"的依据。
"""

from __future__ import annotations

from pathlib import Path

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zhixi.data import build_preprocess, load_secom  # noqa: E402


def tune(y, p):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.02, 0.9, 0.01):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


CONFIGS = {
    "cur": dict(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.9,
                colsample_bytree=0.4, min_child_weight=2, reg_lambda=1.0),
    "reg1": dict(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.3, min_child_weight=5, reg_lambda=2.0),
    "reg2": dict(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.2, min_child_weight=8, reg_lambda=3.0),
    "reg3": dict(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.7,
                 colsample_bytree=0.1, min_child_weight=5, reg_lambda=2.0),
    "reg4": dict(n_estimators=600, max_depth=3, learning_rate=0.03, subsample=0.8,
                 colsample_bytree=0.25, min_child_weight=3, reg_lambda=5.0),
}


def evaluate(Xp, y, pre, kw, spw):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(Xp))
    for tr, va in skf.split(Xp, y):
        m = XGBClassifier(eval_metric="aucpr", tree_method="hist", random_state=42,
                          n_jobs=4, scale_pos_weight=spw, **kw)
        m.fit(Xp.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(Xp.iloc[va])[:, 1]
    pr = average_precision_score(y, oof)
    t, f1 = tune(y.to_numpy(), oof)
    rec = recall_score(y, (oof >= t).astype(int), zero_division=0)
    alarm = float((oof >= t).mean())

    Xtr, ytr = Xp.loc[pre.train_index], y.loc[pre.train_index]
    Xte, yte = Xp.loc[pre.test_index], y.loc[pre.test_index]
    m = XGBClassifier(eval_metric="aucpr", tree_method="hist", random_state=42,
                      n_jobs=4, scale_pos_weight=spw, **kw)
    m.fit(Xtr, ytr)
    pte = m.predict_proba(Xte)[:, 1]
    ho_pr = average_precision_score(yte, pte)
    ho_rec = recall_score(yte, (pte >= t).astype(int), zero_division=0)
    return dict(thr=round(t, 2), alarm=round(alarm, 3), cv_pr=round(pr, 4), cv_f1=round(f1, 4),
                cv_rec=round(rec, 4), ho_pr=round(ho_pr, 4), ho_rec=round(ho_rec, 4),
                of_med=round(float(np.median(oof)), 4), of_p95=round(float(np.percentile(oof, 95)), 4))


def main():
    X, y, ts = load_secom()
    pre = build_preprocess(X, y, ts, seed=42)
    Xp = pre.transform(X)
    spw = (len(y) - y.sum()) / y.sum()
    out = []
    for name, kw in CONFIGS.items():
        r = evaluate(Xp, y, pre, kw, spw)
        r["name"] = name
        out.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    out_path = Path(__file__).resolve().parents[1] / "artifacts" / "tuning_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(out_path, index=False)
    print("saved")


if __name__ == "__main__":
    main()
