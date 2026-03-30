"""
药物相互作用预测：分子指纹 + 逻辑回归基线模型（含可视化）
要求已安装：pandas, numpy, scikit-learn, rdkit-pypi, matplotlib, seaborn
安装命令：pip install pandas numpy scikit-learn rdkit-pypi matplotlib seaborn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ================== 配置参数 ==================
TRAIN_PATH = r".\dataset\drugbankddi\raw\train.csv"  # 训练集文件路径
TEST_PATH = r".\dataset\drugbankddi\raw\test.csv"  # 测试集文件路径
RADIUS = 2  # 摩根指纹半径
NBITS = 2048  # 指纹长度
SAVE_FIGURES = True  # 是否保存图片
FIG_DIR = "figures"  # 图片保存目录


# ================== 辅助函数 ==================
def smiles_to_fingerprint(smiles, radius=RADIUS, nBits=NBITS):
    """将 SMILES 字符串转换为摩根指纹（位向量）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array(fp, dtype=np.uint8)


def prepare_features(df, smiles_col1="smile1", smiles_col2="smile2"):
    """为数据框中的药物对计算特征：两个指纹拼接后的向量"""
    df["fp1"] = df[smiles_col1].apply(smiles_to_fingerprint)
    df["fp2"] = df[smiles_col2].apply(smiles_to_fingerprint)

    original_len = len(df)
    df = df.dropna(subset=["fp1", "fp2"]).reset_index(drop=True)
    if len(df) < original_len:
        print(f"警告：删除了 {original_len - len(df)} 条无效 SMILES 记录")

    X = np.array([np.concatenate([fp1, fp2]) for fp1, fp2 in zip(df["fp1"], df["fp2"])])
    y = df["label"].values
    return X, y


def plot_roc_curve(y_true, y_pred_proba, auc_score, save_path=None):
    """绘制 ROC 曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC 曲线已保存至 {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Interaction (0)', 'Interaction (1)'],
                yticklabels=['No Interaction (0)', 'Interaction (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至 {save_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_importance(model, top_n=20, save_path=None):
    """绘制逻辑回归系数绝对值最大的前 N 个特征（可选，帮助理解）"""
    # 注意：特征数量很多（2*NBITS），只显示前 N 个
    coef = model.coef_[0]
    indices = np.argsort(np.abs(coef))[-top_n:]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), coef[indices], color='steelblue')
    plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
    plt.xlabel('Coefficient')
    plt.title(f'Top {top_n} Important Features (by absolute coefficient)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存至 {save_path}")
    else:
        plt.show()
    plt.close()


# ================== 主程序 ==================
def main():
    # 创建保存目录
    if SAVE_FIGURES and not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    # 1. 读取数据
    print("读取数据...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 检查列名
    required_cols = {"label", "smile1", "smile2"}
    if not required_cols.issubset(train_df.columns):
        raise ValueError(f"训练集列名缺少：{required_cols - set(train_df.columns)}")
    if not required_cols.issubset(test_df.columns):
        raise ValueError(f"测试集列名缺少：{required_cols - set(test_df.columns)}")

    # 2. 准备特征和标签
    print("计算训练集分子指纹...")
    X_train, y_train = prepare_features(train_df)
    print(f"训练集有效样本数: {len(y_train)}")

    print("计算测试集分子指纹...")
    X_test, y_test = prepare_features(test_df)
    print(f"测试集有效样本数: {len(y_test)}")

    # 3. 训练逻辑回归模型
    print("训练逻辑回归模型...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    # 4. 预测并计算指标
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)  # 阈值 0.5
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n测试集 AUC: {auc:.4f}")

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['No Interaction', 'Interaction']))

    # 5. 绘图
    print("\n生成可视化图表...")
    # ROC 曲线
    roc_save = os.path.join(FIG_DIR, "roc_curve.png") if SAVE_FIGURES else None
    plot_roc_curve(y_test, y_pred_proba, auc, save_path=roc_save)

    # 混淆矩阵
    cm_save = os.path.join(FIG_DIR, "confusion_matrix.png") if SAVE_FIGURES else None
    plot_confusion_matrix(y_test, y_pred, save_path=cm_save)

    # 可选：特征重要性（由于特征数量多，绘制前 20 个）
    # 如果需要可以取消注释，但可能不是必须
    # feat_imp_save = os.path.join(FIG_DIR, "feature_importance.png") if SAVE_FIGURES else None
    # plot_feature_importance(model, top_n=20, save_path=feat_imp_save)

    if SAVE_FIGURES:
        print(f"\n所有图表已保存至 {FIG_DIR} 目录")


if __name__ == "__main__":
    main()