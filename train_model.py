"""训练风险模型并生成评估产物。

用法：
    python train_model.py

产物：
    artifacts/risk_model.joblib    上线推理模型
    artifacts/preprocess.joblib    预处理对象（缺失值、信号簇映射、训练/测试划分）
    artifacts/metrics.json         指标（PR-AUC / Recall / F1 / 混淆矩阵等）
    artifacts/lots.csv             全量批次的风险概率与时间戳（供界面回放）
    artifacts/pr_curve.csv         精确率-召回率曲线
    artifacts/roc_curve.csv        ROC 曲线
"""

import json

from zhixi.model import train

if __name__ == "__main__":
    metrics = train(save=True, verbose=False)
    print("模型训练完成 —— 关键指标")
    print(json.dumps(
        {
            "模型": metrics["model_kind"],
            "样本数": metrics["n_samples"],
            "异常样本数": metrics["n_positives"],
            "异常占比": metrics["base_rate"],
            "集成模型数": metrics["ensemble_size"],
            "袋外F1最优阈值": metrics["thresholds"]["cv_threshold"],
            "上线触发阈值": metrics["thresholds"]["operational_threshold"],
            "上线报警比例": metrics["thresholds"]["deployed_alarm_rate"],
            "交叉验证": {
                k: metrics["cv"][k] for k in ["pr_auc", "roc_auc", "recall", "precision", "f1"]
            },
            "时间留出集": {
                k: metrics["holdout"][k] for k in ["pr_auc", "roc_auc", "recall", "precision", "f1"]
            },
        },
        ensure_ascii=False,
        indent=2,
    ))
