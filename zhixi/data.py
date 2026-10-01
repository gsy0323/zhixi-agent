"""SECOM 数据加载与预处理。

数据集事实（来自 UCI 官方数据集说明）：
  * 1567 条样本 × 591 个变量，每一行 = 一个生产批次(production entity)的工艺信号；
  * 标签 -1 = 良品(pass)，+1 = 异常(fail)；其中异常 104 条，占比约 6.6%；
  * 591 个变量是匿名化的半导体前道工序传感器/过程测量点，变量名未公开；
  * 数据含缺失值(以 NaN 表示)；标签文件附带时间戳；
  * 许可：CC BY 4.0。

因此本项目把每一行视为“一个批次在本厂前道工序段的工艺信号快照”，
用变量聚类把匿名变量归为若干“工艺信号簇”，作为工单派发到“工序站”的依据。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PREPROCESS_PATH, RAW_DIR

N_FEATURES = 591
FEATURE_NAMES = [f"var_{i:03d}" for i in range(1, N_FEATURES + 1)]
GROUP_LABELS = list("ABCDEFGHIJKL")  # 12 个工艺信号簇 -> 工序站 A..L


def load_secom(raw_dir: Path | str = RAW_DIR) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """读取 SECOM 原始文件，返回 (X, y, timestamp)。"""

    raw = Path(raw_dir)
    data_file = raw / "secom.data"
    label_file = raw / "secom_labels.data"
    if not data_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            f"未找到 SECOM 原始数据，请先运行 `python setup_data.py`。期望文件：{data_file}"
        )

    X = pd.read_csv(
        data_file,
        sep=r"\s+",
        header=None,
        names=FEATURE_NAMES,
        na_values=["NaN"],
        engine="python",
    )

    # 标签文件格式： -1 "19/07/2008 11:55:00"
    # 时间戳内部含空格且被引号包裹，直接按空白切分会串列，因此用正则解析。
    import re

    pattern = re.compile(r'^\s*(-?\d+)\s+"([^"]+)"\s*$')
    labels: list[int] = []
    stamps: list[str] = []
    for line in Path(label_file).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        m = pattern.match(line)
        if not m:
            continue
        labels.append(int(m.group(1)))
        stamps.append(m.group(2))

    ts = pd.to_datetime(pd.Series(stamps), format="%d/%m/%Y %H:%M:%S", errors="coerce")
    y = (pd.Series(labels) == 1).astype(int)
    return X, y, ts


def _cluster_features(corr: pd.DataFrame, n_groups: int, seed: int = 42) -> dict[str, str]:
    """按相关性把匿名变量聚成工艺信号簇。"""

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    m = corr.abs().to_numpy(copy=True)
    np.fill_diagonal(m, 1.0)
    dist = 1.0 - m
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    labels = fcluster(z, t=n_groups, criterion="maxclust")

    # 按簇规模从大到小编号，保证簇 A 稳定
    order = (
        pd.Series(labels).value_counts().sort_values(ascending=False).index.tolist()
    )
    remap = {old: new for new, old in enumerate(order)}
    names = corr.columns.tolist()
    return {
        names[i]: f"工序站{GROUP_LABELS[remap[labels[i]] % len(GROUP_LABELS)]}"
        for i in range(len(names))
    }


@dataclass
class Preprocess:
    """可持久化的预处理对象：缺失值填补 + 常数列剔除 + 信号簇映射。"""

    feature_names: list[str]
    medians: pd.Series
    group_map: dict[str, str]
    train_index: list[int]
    test_index: list[int]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.loc[:, self.feature_names]
        return X.fillna(self.medians)

    def group_of(self, feature: str) -> str:
        return self.group_map.get(feature, "工序站A")

    def save(self, path: Path | str = PREPROCESS_PATH) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path | str = PREPROCESS_PATH) -> "Preprocess":
        import joblib

        return joblib.load(path)


def build_preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    *,
    test_ratio: float = 0.3,
    n_groups: int = 12,
    seed: int = 42,
) -> Preprocess:
    """构建预处理对象。

    划分策略：按时间排序后切分，前 70% 训练、后 30% 测试，
    更接近“用历史批次训练、预测未来批次”的真实上线方式。
    """

    order = ts.sort_values(kind="mergesort").index
    n = len(order)
    cut = int(n * (1 - test_ratio))
    train_index = list(order[:cut])
    test_index = list(order[cut:])

    X = X.loc[order]
    train = X.loc[train_index]

    # 剔除训练集中取值为常量的列（SECOM 官方基线也做了这一步）
    keep = [c for c in X.columns if train[c].nunique(dropna=True) > 1]
    X = X.loc[:, keep]

    medians = X.loc[train_index].median()
    X_filled = X.fillna(medians)

    corr = X_filled.loc[train_index].corr()
    group_map = _cluster_features(corr, n_groups=n_groups, seed=seed)

    return Preprocess(
        feature_names=list(X.columns),
        medians=medians,
        group_map=group_map,
        train_index=train_index,
        test_index=test_index,
    )


def load_processed(raw_dir: Path | str = RAW_DIR) -> tuple[pd.DataFrame, pd.Series, pd.Series, Preprocess]:
    """一步到位：读取原始数据 + 加载（或构建并缓存）预处理对象。"""

    X_raw, y, ts = load_secom(raw_dir)
    if PREPROCESS_PATH.exists():
        pre = Preprocess.load()
    else:
        pre = build_preprocess(X_raw, y, ts)
        pre.save()
    X = pre.transform(X_raw)
    return X, y, ts, pre
