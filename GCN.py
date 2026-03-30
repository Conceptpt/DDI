# -*- coding: utf-8 -*-
"""
基础GCN版本：仅使用 GCN 表征进行 DDI 预测，不融合原始分子指纹 pair 特征

与改进GCN模型的区别：
1. 节点输入仍然是 Morgan Fingerprint（因为 GCN 需要节点初始特征）
2. 但 pair decoder 不再拼接原始分子指纹组合项
3. 即仅使用:
   [|h_u-h_v|, h_u*h_v]
4. 仅保存：
   - 最佳模型
   - 部署工件
   - 测试预测结果
   - 测试指标
   - ROC 曲线
   - 混淆矩阵
"""

import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")


# =========================================================
# 全局配置
# =========================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RADIUS = 2
NBITS = 2048

HIDDEN_DIM = 256
DROPOUT = 0.2

LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 150
PATIENCE = 12
EVAL_EVERY = 5

NEG_SAMPLING_RATIO = 1
TRAIN_POS_RATIO = 0.85
VAL_POS_RATIO = 0.15

# 数据路径
TRAIN_CSV = r"D:\DeepL_Project\GnnbasedDrugDrugInteractionPrediction-github\dataset\drugbankddi\raw\train.csv"
TEST_CSV = r"D:\DeepL_Project\GnnbasedDrugDrugInteractionPrediction-github\dataset\drugbankddi\raw\test.csv"

# 输出目录
ARTIFACT_DIR = "GCN_Results"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# =========================================================
# 随机种子
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 基础工具
# =========================================================
def canonicalize_edge(u, v):
    return (u, v) if u < v else (v, u)


def edge_index_to_set(edge_index):
    edge_set = set()
    if edge_index is None or edge_index.numel() == 0:
        return edge_set
    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        edge_set.add(canonicalize_edge(u, v))
    return edge_set


def edge_set_to_index(edge_set):
    if len(edge_set) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edges = list(edge_set)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# =========================================================
# 分子特征
# =========================================================
def smiles_to_fingerprint(smiles, radius=RADIUS, nbits=NBITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
    return np.array(fp, dtype=np.float32)


# =========================================================
# 数据读取与图构建
# =========================================================
def load_csv_checked(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = {"smile1", "smile2", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} 缺少必要列: {required_cols - set(df.columns)}")
    return df


def build_global_drug_vocab(train_df, test_df):
    all_smiles = pd.concat([
        train_df["smile1"], train_df["smile2"],
        test_df["smile1"], test_df["smile2"]
    ]).dropna().unique()

    drug_to_idx = {s: i for i, s in enumerate(all_smiles)}
    idx_to_smiles = {i: s for s, i in drug_to_idx.items()}
    return drug_to_idx, idx_to_smiles


def extract_positive_edge_set(df, drug_to_idx):
    edge_set = set()
    for _, row in df.iterrows():
        if int(row["label"]) != 1:
            continue

        s1, s2 = row["smile1"], row["smile2"]
        if s1 not in drug_to_idx or s2 not in drug_to_idx:
            continue

        u = drug_to_idx[s1]
        v = drug_to_idx[s2]
        if u == v:
            continue

        edge_set.add(canonicalize_edge(u, v))
    return edge_set


def build_test_records(df_test, drug_to_idx, idx_to_smiles_global):
    records = []

    for _, row in df_test.iterrows():
        s1, s2 = row["smile1"], row["smile2"]
        label = int(row["label"])

        if s1 not in drug_to_idx or s2 not in drug_to_idx:
            continue

        u = drug_to_idx[s1]
        v = drug_to_idx[s2]
        if u == v:
            continue

        u, v = canonicalize_edge(u, v)

        records.append({
            "drug1_idx": u,
            "drug2_idx": v,
            "drug1_smiles": idx_to_smiles_global[u],
            "drug2_smiles": idx_to_smiles_global[v],
            "true_label": label
        })

    df_records = pd.DataFrame(records).drop_duplicates(
        subset=["drug1_idx", "drug2_idx", "true_label"]
    ).reset_index(drop=True)

    return df_records


def split_train_val_edges(train_pos_edge_index, train_ratio=TRAIN_POS_RATIO, seed=SEED):
    num_edges = train_pos_edge_index.size(1)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(num_edges, generator=g)

    train_end = int(train_ratio * num_edges)
    train_idx = perm[:train_end]
    val_idx = perm[train_end:]

    edge_train = train_pos_edge_index[:, train_idx]
    edge_val = train_pos_edge_index[:, val_idx]
    return edge_train, edge_val


def compute_node_features(idx_to_smiles):
    num_nodes = len(idx_to_smiles)
    feats = []
    invalid_count = 0

    for i in range(num_nodes):
        smi = idx_to_smiles[i]
        fp = smiles_to_fingerprint(smi)
        if fp is None:
            fp = np.zeros(NBITS, dtype=np.float32)
            invalid_count += 1
        feats.append(fp)

    feats = torch.tensor(np.array(feats), dtype=torch.float)
    print(f"[Info] 节点特征计算完成，非法 SMILES 数量: {invalid_count}")
    return feats


# =========================================================
# 负采样
# =========================================================
def sample_negative_edges(avoid_edge_index, num_nodes, num_samples):
    avoid_set = edge_index_to_set(avoid_edge_index)
    neg_set = set()

    max_trials = max(num_samples * 200, 2000)
    trials = 0

    while len(neg_set) < num_samples and trials < max_trials:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        trials += 1

        if u == v:
            continue

        e = canonicalize_edge(u, v)
        if e in avoid_set or e in neg_set:
            continue

        neg_set.add(e)

    if len(neg_set) < num_samples:
        print(f"[Warning] 负采样不足，期望 {num_samples}，实际 {len(neg_set)}")

    if len(neg_set) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    neg_edges = list(neg_set)
    return torch.tensor(neg_edges, dtype=torch.long).t().contiguous()


# =========================================================
# 模型定义：消融版，不融合原始指纹 pair 特征
# =========================================================
class GCNLinkPredictorAblation(nn.Module):
    """
    消融版本：
    - 节点输入仍是分子指纹 x
    - GCN 编码得到 h
    - decode 时只使用 h 的对称组合：
      [|h_u-h_v|, h_u*h_v]
    不再使用：
      [|x_u-x_v|, x_u*x_v]
    """
    def __init__(self, in_dim, hidden_dim, out_dim=1, dropout=0.2):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = dropout

        pair_dim = 2 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        return h

    def build_pair_feature(self, h, edge_pairs):
        if edge_pairs.numel() == 0:
            return torch.empty((0, 2 * h.size(1)), device=h.device)

        u = edge_pairs[0]
        v = edge_pairs[1]

        h_u = h[u]
        h_v = h[v]

        feat = torch.cat([
            torch.abs(h_u - h_v),
            h_u * h_v
        ], dim=1)

        return feat

    def decode(self, h, edge_pairs):
        if edge_pairs.numel() == 0:
            return torch.empty((0,), device=h.device)

        feat = self.build_pair_feature(h, edge_pairs)
        logit = self.mlp(feat).squeeze(-1)
        return logit

    def forward(self, x, message_edge_index, pos_edge, neg_edge):
        h = self.encode(x, message_edge_index)
        pos_logit = self.decode(h, pos_edge)
        neg_logit = self.decode(h, neg_edge)
        return pos_logit, neg_logit


# =========================================================
# 训练与评估
# =========================================================
def train_one_epoch(model, optimizer, x, edge_train_message, train_pos_all_for_sampling, num_nodes, device):
    model.train()

    num_pos = edge_train_message.size(1)
    num_neg = num_pos * NEG_SAMPLING_RATIO

    neg_edges = sample_negative_edges(
        avoid_edge_index=train_pos_all_for_sampling,
        num_nodes=num_nodes,
        num_samples=num_neg
    )

    x = x.to(device)
    edge_train_message = edge_train_message.to(device)
    pos_edge = edge_train_message.to(device)
    neg_edge = neg_edges.to(device)

    optimizer.zero_grad()

    pos_logit, neg_logit = model(x, edge_train_message, pos_edge, neg_edge)

    pos_label = torch.ones_like(pos_logit)
    neg_label = torch.zeros_like(neg_logit)

    logits = torch.cat([pos_logit, neg_logit], dim=0)
    labels = torch.cat([pos_label, neg_label], dim=0)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def find_best_threshold_by_f1(y_true, y_score):
    candidate_thresholds = np.linspace(0.05, 0.95, 91)

    best_threshold = 0.5
    best_f1 = -1.0

    for thr in candidate_thresholds:
        y_pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    return float(best_threshold), float(best_f1)


@torch.no_grad()
def evaluate_on_positive_edges(
    model,
    x,
    message_edge_index,
    eval_pos_edges,
    avoid_edge_index_for_neg_sampling,
    num_nodes,
    device
):
    model.eval()

    x = x.to(device)
    message_edge_index = message_edge_index.to(device)
    h = model.encode(x, message_edge_index)

    pos_edge = eval_pos_edges.to(device)
    pos_logit = model.decode(h, pos_edge)
    pos_prob = torch.sigmoid(pos_logit).cpu().numpy()

    num_pos = eval_pos_edges.size(1)
    neg_edges = sample_negative_edges(
        avoid_edge_index=avoid_edge_index_for_neg_sampling,
        num_nodes=num_nodes,
        num_samples=num_pos
    ).to(device)

    neg_logit = model.decode(h, neg_edges)
    neg_prob = torch.sigmoid(neg_logit).cpu().numpy()

    y_true = np.concatenate([np.ones(len(pos_prob)), np.zeros(len(neg_prob))])
    y_score = np.concatenate([pos_prob, neg_prob])

    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    best_threshold, best_f1 = find_best_threshold_by_f1(y_true, y_score)
    y_pred = (y_score >= best_threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        "auc": auc,
        "aupr": aupr,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "precision": precision,
        "recall": recall
    }


@torch.no_grad()
def predict_test_dataframe(model, x, message_edge_index, df_test_records, device, threshold=0.5):
    model.eval()

    x = x.to(device)
    message_edge_index = message_edge_index.to(device)
    h = model.encode(x, message_edge_index)

    edge_pairs = torch.tensor(
        df_test_records[["drug1_idx", "drug2_idx"]].values,
        dtype=torch.long
    ).t().contiguous().to(device)

    logits = model.decode(h, edge_pairs)
    probs = torch.sigmoid(logits).cpu().numpy()

    df_pred = df_test_records.copy()
    df_pred["pred_prob"] = probs
    df_pred["pred_label"] = (df_pred["pred_prob"] >= threshold).astype(int)

    return df_pred


def evaluate_test_metrics(df_pred):
    y_true = df_pred["true_label"].values
    y_score = df_pred["pred_prob"].values
    y_pred = df_pred["pred_label"].values

    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        "auc": auc,
        "aupr": aupr,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# =========================================================
# 可视化：只保留 ROC 与混淆矩阵
# =========================================================
def plot_roc_curve(df_pred, save_path):
    y_true = df_pred["true_label"].values
    y_score = df_pred["pred_prob"].values

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}", color="#d62728")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] ROC 曲线: {save_path}")


def plot_confusion_matrix(df_pred, save_path):
    y_true = df_pred["true_label"].values
    y_pred = df_pred["pred_label"].values

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] 混淆矩阵: {save_path}")


# =========================================================
# 工件保存
# =========================================================
def save_artifacts(
    model,
    node_features,
    edge_train,
    edge_val,
    train_pos_all,
    drug_to_idx,
    idx_to_smiles,
    df_test_records,
    best_threshold,
    save_dir=ARTIFACT_DIR
):
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "best_gcn_ablation_model.pth"))
    torch.save(node_features.cpu(), os.path.join(save_dir, "node_features.pt"))
    torch.save(edge_train.cpu(), os.path.join(save_dir, "edge_train.pt"))
    torch.save(edge_val.cpu(), os.path.join(save_dir, "edge_val.pt"))
    torch.save(train_pos_all.cpu(), os.path.join(save_dir, "train_pos_all.pt"))

    with open(os.path.join(save_dir, "drug_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(drug_to_idx, f, ensure_ascii=False, indent=2)

    idx_to_smiles_json = {str(k): v for k, v in idx_to_smiles.items()}
    with open(os.path.join(save_dir, "idx_to_smiles.json"), "w", encoding="utf-8") as f:
        json.dump(idx_to_smiles_json, f, ensure_ascii=False, indent=2)

    df_test_records.to_csv(
        os.path.join(save_dir, "strict_test_records.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    config = {
        "MODEL_NAME": "GCN_Ablation_NoFPFusion",
        "SEED": SEED,
        "RADIUS": RADIUS,
        "NBITS": NBITS,
        "HIDDEN_DIM": HIDDEN_DIM,
        "DROPOUT": DROPOUT,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "EPOCHS": EPOCHS,
        "PATIENCE": PATIENCE,
        "EVAL_EVERY": EVAL_EVERY,
        "NEG_SAMPLING_RATIO": NEG_SAMPLING_RATIO,
        "TRAIN_POS_RATIO": TRAIN_POS_RATIO,
        "VAL_POS_RATIO": VAL_POS_RATIO,
        "BEST_THRESHOLD_FROM_VAL": best_threshold,
        "ABLATION_NOTE": "Only use GCN pair embedding [|h_u-h_v|, h_u*h_v], without raw fingerprint fusion in decoder."
    }

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[Saved] 模型与工件目录: {save_dir}")


# =========================================================
# 主流程
# =========================================================
def main():
    set_seed(SEED)

    print("=" * 70)
    print("1. 读取训练集与测试集")
    print("=" * 70)
    df_train = load_csv_checked(TRAIN_CSV)
    df_test = load_csv_checked(TEST_CSV)

    print(f"train.csv 样本数: {len(df_train)}")
    print(f"test.csv 样本数 : {len(df_test)}")

    print("\n" + "=" * 70)
    print("2. 构建全局药物词表")
    print("=" * 70)
    drug_to_idx, idx_to_smiles = build_global_drug_vocab(df_train, df_test)
    num_nodes = len(drug_to_idx)
    print(f"药物节点数: {num_nodes}")

    print("\n" + "=" * 70)
    print("3. 从训练集提取正边，并划分训练/验证")
    print("=" * 70)
    train_pos_set = extract_positive_edge_set(df_train, drug_to_idx)
    train_pos_all = edge_set_to_index(train_pos_set)

    if train_pos_all.size(1) == 0:
        raise ValueError("train.csv 中未提取到正边，请检查 label=1 样本。")

    edge_train, edge_val = split_train_val_edges(train_pos_all, train_ratio=TRAIN_POS_RATIO, seed=SEED)

    print(f"训练集正边总数: {train_pos_all.size(1)}")
    print(f"edge_train 数量 : {edge_train.size(1)}")
    print(f"edge_val 数量   : {edge_val.size(1)}")

    print("\n" + "=" * 70)
    print("4. 构建严格独立测试记录")
    print("=" * 70)
    df_test_records = build_test_records(df_test, drug_to_idx, idx_to_smiles)
    if len(df_test_records) == 0:
        raise ValueError("test.csv 未生成有效测试记录，请检查数据。")

    print(f"测试记录数: {len(df_test_records)}")
    print("测试标签分布：")
    print(df_test_records["true_label"].value_counts(dropna=False).sort_index())

    print("\n" + "=" * 70)
    print("5. 计算节点特征（Morgan Fingerprint）")
    print("=" * 70)
    node_features = compute_node_features(idx_to_smiles)
    print(f"节点特征维度: {tuple(node_features.shape)}")

    print("\n" + "=" * 70)
    print("6. 初始化消融模型")
    print("=" * 70)
    model = GCNLinkPredictorAblation(
        in_dim=NBITS,
        hidden_dim=HIDDEN_DIM,
        out_dim=1,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_model_path = os.path.join(ARTIFACT_DIR, "best_gcn_ablation_model.pth")
    best_val_auc = 0.0
    best_val_aupr = 0.0
    best_threshold = 0.5
    patience_counter = 0

    print("\n" + "=" * 70)
    print("7. 开始训练")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            x=node_features,
            edge_train_message=edge_train,
            train_pos_all_for_sampling=train_pos_all,
            num_nodes=num_nodes,
            device=DEVICE
        )

        if epoch % EVAL_EVERY == 0:
            val_result = evaluate_on_positive_edges(
                model=model,
                x=node_features,
                message_edge_index=edge_train,
                eval_pos_edges=edge_val,
                avoid_edge_index_for_neg_sampling=train_pos_all,
                num_nodes=num_nodes,
                device=DEVICE
            )

            val_auc = val_result["auc"]
            val_aupr = val_result["aupr"]

            print(
                f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f} | "
                f"BestThr: {val_result['best_threshold']:.2f} | "
                f"F1: {val_result['best_f1']:.4f}"
            )

            improved = (val_aupr > best_val_aupr) or (
                np.isclose(val_aupr, best_val_aupr) and val_auc > best_val_auc
            )

            if improved:
                best_val_auc = val_auc
                best_val_aupr = val_aupr
                best_threshold = val_result["best_threshold"]
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"[Info] 保存最佳模型 -> {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("[Info] Early stopping triggered.")
                    break

    print(f"\n最佳验证集 AUC : {best_val_auc:.4f}")
    print(f"最佳验证集 AUPR: {best_val_aupr:.4f}")
    print(f"最佳验证阈值   : {best_threshold:.2f}")

    print("\n" + "=" * 70)
    print("8. 加载最佳模型")
    print("=" * 70)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    print("\n" + "=" * 70)
    print("9. 严格独立测试评估")
    print("=" * 70)
    df_pred = predict_test_dataframe(
        model=model,
        x=node_features,
        message_edge_index=edge_train,
        df_test_records=df_test_records,
        device=DEVICE,
        threshold=best_threshold
    )

    test_metrics = evaluate_test_metrics(df_pred)
    print(f"测试集 AUC      : {test_metrics['auc']:.4f}")
    print(f"测试集 AUPR     : {test_metrics['aupr']:.4f}")
    print(f"测试集 F1       : {test_metrics['f1']:.4f}")
    print(f"测试集 Precision: {test_metrics['precision']:.4f}")
    print(f"测试集 Recall   : {test_metrics['recall']:.4f}")

    print("\n分类报告：")
    print(classification_report(df_pred["true_label"], df_pred["pred_label"], digits=4))

    print("\n" + "=" * 70)
    print("10. 保存模型与工件")
    print("=" * 70)
    save_artifacts(
        model=model,
        node_features=node_features,
        edge_train=edge_train,
        edge_val=edge_val,
        train_pos_all=train_pos_all,
        drug_to_idx=drug_to_idx,
        idx_to_smiles=idx_to_smiles,
        df_test_records=df_test_records,
        best_threshold=best_threshold,
        save_dir=ARTIFACT_DIR
    )

    pred_csv_path = os.path.join(ARTIFACT_DIR, "test_predictions.csv")
    df_pred.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] 测试集预测结果: {pred_csv_path}")

    metrics_path = os.path.join(ARTIFACT_DIR, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    print(f"[Saved] 测试集指标: {metrics_path}")

    print("\n" + "=" * 70)
    print("11. 绘制 ROC / 混淆矩阵")
    print("=" * 70)
    roc_path = os.path.join(ARTIFACT_DIR, "roc_curve.png")
    cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")

    plot_roc_curve(df_pred, roc_path)
    plot_confusion_matrix(df_pred, cm_path)

    print("\n" + "=" * 70)
    print("全部完成")
    print("=" * 70)
    print(f"输出目录: {os.path.abspath(ARTIFACT_DIR)}")


if __name__ == "__main__":
    main()