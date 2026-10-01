"""可解释性模块。

优先使用 SHAP（TreeExplainer）给出每个批次的关键异常信号及其贡献值；
未安装 shap 时退化为“逻辑回归代理模型”的线性贡献近似，接口保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Contribution:
    feature: str
    group: str
    value: float
    z_score: float
    shap_value: float
    direction: str  # "升高风险" / "降低风险"


class Explainer:
    def __init__(self, model, X_train: pd.DataFrame):
        """model: zhixi.model.RiskModel；X_train: 预处理后的训练集特征。"""

        self.model = model
        self.feature_names = list(model.feature_names)
        self.X_train = X_train.loc[:, self.feature_names]
        self.mean = self.X_train.mean()
        self.std = self.X_train.std().replace(0.0, 1.0)

        self.backend = "surrogate-linear"
        self._shap = None
        self._explainers: list = []
        self._coef = None
        estimators = getattr(model, "estimators", None) or [model.estimator]
        try:
            import shap  # noqa: F401

            self._shap = shap
            self._explainers = [shap.TreeExplainer(e) for e in estimators]
            self.backend = "shap-tree"
        except Exception:
            from sklearn.linear_model import LogisticRegression

            Z = ((self.X_train - self.mean) / self.std).to_numpy()
            y = _pseudo_labels(self.X_train)
            lr = LogisticRegression(max_iter=2000, class_weight="balanced")
            lr.fit(Z, y)
            self._coef = pd.Series(lr.coef_[0], index=self.feature_names)

    def contributions(self, X_raw: pd.DataFrame, row_index: int, top_k: int = 8) -> list[Contribution]:
        row_raw = X_raw.loc[[row_index], self.feature_names]
        row = self.model.preprocess.transform(row_raw).iloc[0]
        z = (row - self.mean) / self.std

        if self.backend == "shap-tree":
            per_model = []
            for expl in self._explainers:
                v = expl.shap_values(row.to_frame().T)
                if isinstance(v, list):  # 兼容旧版 SHAP 的二分类返回
                    v = v[1]
                per_model.append(np.asarray(v).reshape(-1))
            values = np.mean(per_model, axis=0)
        else:
            values = (self._coef * z).to_numpy()

        order = np.argsort(-np.abs(values))[:top_k]
        out: list[Contribution] = []
        for i in order:
            name = self.feature_names[i]
            v = float(values[i])
            out.append(
                Contribution(
                    feature=name,
                    group=self.model.preprocess.group_of(name),
                    value=float(row.iloc[i]),
                    z_score=float(z.iloc[i]),
                    shap_value=round(v, 4),
                    direction="升高风险" if v > 0 else "降低风险",
                )
            )
        return out

    def group_ranking(self, contribs: list[Contribution]) -> list[tuple[str, float]]:
        agg: dict[str, float] = {}
        for c in contribs:
            if c.shap_value > 0:
                agg[c.group] = agg.get(c.group, 0.0) + c.shap_value
        return sorted(agg.items(), key=lambda kv: -kv[1])


def _pseudo_labels(X_train: pd.DataFrame, n: int = 400, seed: int = 0) -> np.ndarray:
    """代理模型没有真实标签时的近似：用风险模型预测作为软标签。"""

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=min(n, len(X_train)), replace=False)
    return (np.abs(_zscore(X_train.iloc[idx])) > 2.5).sum(axis=1).to_numpy() > 0


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std().replace(0.0, 1.0)


def humanize(contribs: list[Contribution]) -> str:
    """把贡献值转成答辩现场能读出来的一句话。"""

    if not contribs:
        return "未发现显著异常信号。"
    parts = [
        f"{c.feature}（{c.group}，贡献 {c.shap_value:+.2f}，当前值偏差 {c.z_score:+.2f}σ）"
        for c in contribs[:3]
        if c.shap_value > 0
    ]
    return "主要异常信号：" + "、".join(parts) if parts else "未发现显著拉升风险的信号。"
