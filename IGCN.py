# -*- coding: utf-8 -*-
"""
融合分子指纹特征的GCN 药物相互作用（DDI）预测

1. 严格无信息泄漏：
   - 仅使用 train.csv 中的正样本边构建训练图
   - 从 train 的正边中切分验证边
   - test.csv 仅用于独立测试，不参与图构建、不参与训练
2. 可视化：
   - 对称式 pair decoder，更适合无向 DDI
   - 验证集自动选择最优阈值（基于 F1）
   - 增加 AUPR / PR 曲线
   - 增加 pair embedding t-SNE（按真实标签双颜色可视化）
   - t-SNE 可视化支持类别均衡采样，使图更易解释
3. 保存模型方便后续界面部署：
   - 保存最佳模型
   - 保存测试预测结果
   - 保存 ROC / PR / 混淆矩阵 / t-SNE
   - 保存案例分析
   - 保存部署工件
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
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

# t-SNE 可视化配置
TSNE_BALANCED_VIS = True          # 是否对 t-SNE 可视化做类别均衡抽样（仅用于画图）
TSNE_MAX_POINTS_PER_CLASS = 1000  # 每类最多可视化多少点
TSNE_PCA_DIM = 50                 # t-SNE 前先 PCA 到多少维
TSNE_PERPLEXITY = 30

# 数据路径
TRAIN_CSV = r"D:\DeepL_Project\GnnbasedDrugDrugInteractionPrediction-github\dataset\drugbankddi\raw\train.csv"
TEST_CSV = r"D:\DeepL_Project\GnnbasedDrugDrugInteractionPrediction-github\dataset\drugbankddi\raw\test.csv"

# 输出目录
ARTIFACT_DIR = "I_GCN_Results"
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
    """无向边规范化"""
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


def merge_edge_indices(*edge_indices):
    merged = set()
    for edge_index in edge_indices:
        merged |= edge_index_to_set(edge_index)
    return edge_set_to_index(merged)


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
    """
    构建全局药物词表：
    - 包含 train/test 中出现的所有药物节点
    - 不包含任何测试边标签信息
    """
    all_smiles = pd.concat([
        train_df["smile1"], train_df["smile2"],
        test_df["smile1"], test_df["smile2"]
    ]).dropna().unique()

    drug_to_idx = {s: i for i, s in enumerate(all_smiles)}
    idx_to_smiles = {i: s for s, i in drug_to_idx.items()}
    return drug_to_idx, idx_to_smiles


def extract_positive_edge_set(df, drug_to_idx):
    """
    从 df 中提取 label=1 的无向正边集合
    """
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


def build_test_records(df_test, drug_to_idx):
    """
    严格独立测试记录：
    - 保留 test.csv 原始样本
    - 对无向边做规范化，避免 (u,v) 与 (v,u) 顺序影响预测
    """
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
    """
    仅对 train.csv 的正边做划分：
    - 一部分用于 message passing + 监督训练
    - 一部分作为验证正边
    """
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
    """
    负采样，避开 avoid_edge_index 中所有正边
    """
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
# 模型定义
# =========================================================
class GCNLinkPredictor(nn.Module):
    """
    论文友好增强点：
    1. GCN 两层编码
    2. 使用 BatchNorm 提升稳定性
    3. 使用对称式 pair representation，适配无向 DDI：
       [|h_u-h_v|, h_u*h_v, |x_u-x_v|, x_u*x_v]
    """
    def __init__(self, in_dim, hidden_dim, out_dim=1, dropout=0.2):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = dropout

        pair_dim = 2 * hidden_dim + 2 * in_dim
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

    def build_pair_feature(self, h, x, edge_pairs):
        if edge_pairs.numel() == 0:
            return torch.empty((0, 2 * h.size(1) + 2 * x.size(1)), device=h.device)

        u = edge_pairs[0]
        v = edge_pairs[1]

        h_u = h[u]
        h_v = h[v]
        x_u = x[u]
        x_v = x[v]

        feat = torch.cat([
            torch.abs(h_u - h_v),
            h_u * h_v,
            torch.abs(x_u - x_v),
            x_u * x_v
        ], dim=1)

        return feat

    def decode(self, h, x, edge_pairs):
        if edge_pairs.numel() == 0:
            return torch.empty((0,), device=h.device)

        feat = self.build_pair_feature(h, x, edge_pairs)
        logit = self.mlp(feat).squeeze(-1)
        return logit

    def forward(self, x, message_edge_index, pos_edge, neg_edge):
        h = self.encode(x, message_edge_index)
        pos_logit = self.decode(h, x, pos_edge)
        neg_logit = self.decode(h, x, neg_edge)
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


@torch.no_grad()
def get_node_embeddings(model, x, message_edge_index, device):
    model.eval()
    x = x.to(device)
    message_edge_index = message_edge_index.to(device)
    h = model.encode(x, message_edge_index)
    return h


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
    """
    验证集评估：
    - 正样本：验证正边
    - 负样本：随机负采样，数量与正样本相同
    - 输出 AUC / AUPR / 最佳阈值 / F1
    """
    model.eval()

    x = x.to(device)
    message_edge_index = message_edge_index.to(device)
    h = model.encode(x, message_edge_index)

    pos_edge = eval_pos_edges.to(device)
    pos_logit = model.decode(h, x, pos_edge)
    pos_prob = torch.sigmoid(pos_logit).cpu().numpy()

    num_pos = eval_pos_edges.size(1)
    neg_edges = sample_negative_edges(
        avoid_edge_index=avoid_edge_index_for_neg_sampling,
        num_nodes=num_nodes,
        num_samples=num_pos
    ).to(device)

    neg_logit = model.decode(h, x, neg_edges)
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
    """
    对 test.csv 中每一条样本进行预测
    """
    model.eval()

    x = x.to(device)
    message_edge_index = message_edge_index.to(device)
    h = model.encode(x, message_edge_index)

    edge_pairs = torch.tensor(
        df_test_records[["drug1_idx", "drug2_idx"]].values,
        dtype=torch.long
    ).t().contiguous().to(device)

    logits = model.decode(h, x, edge_pairs)
    probs = torch.sigmoid(logits).cpu().numpy()

    df_pred = df_test_records.copy()
    df_pred["pred_prob"] = probs
    df_pred["pred_label"] = (df_pred["pred_prob"] >= threshold).astype(int)

    return df_pred


def find_best_threshold_by_f1(y_true, y_score):
    """
    在验证集上搜索最佳二分类阈值
    """
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
# 可视化
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


def plot_pr_curve(df_pred, save_path):
    y_true = df_pred["true_label"].values
    y_score = df_pred["pred_prob"].values

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, linewidth=2, label=f"AUPR = {ap:.4f}", color="#1f77b4")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] PR 曲线: {save_path}")


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


@torch.no_grad()
def visualize_pair_tsne(
    model,
    x,
    message_edge_index,
    df_records,
    save_csv,
    save_fig,
    device,
    balanced_vis=True,
    max_points_per_class=1000
):
    """
    论文友好 t-SNE：
    - 可视化对象：pair embedding，而不是 node embedding
    - 用 true_label 上色
    - 支持类别均衡抽样，仅用于可视化展示
    """
    vis_df = df_records.copy()

    if balanced_vis:
        pos_df = vis_df[vis_df["true_label"] == 1]
        neg_df = vis_df[vis_df["true_label"] == 0]

        if len(pos_df) > 0 and len(neg_df) > 0:
            n_each = min(len(pos_df), len(neg_df), max_points_per_class)
            pos_df = pos_df.sample(n=n_each, random_state=SEED) if len(pos_df) > n_each else pos_df
            neg_df = neg_df.sample(n=n_each, random_state=SEED) if len(neg_df) > n_each else neg_df
            vis_df = pd.concat([pos_df, neg_df], axis=0).sample(frac=1, random_state=SEED).reset_index(drop=True)
    else:
        vis_df_pos = vis_df[vis_df["true_label"] == 1]
        vis_df_neg = vis_df[vis_df["true_label"] == 0]
        if len(vis_df_pos) > max_points_per_class:
            vis_df_pos = vis_df_pos.sample(n=max_points_per_class, random_state=SEED)
        if len(vis_df_neg) > max_points_per_class:
            vis_df_neg = vis_df_neg.sample(n=max_points_per_class, random_state=SEED)
        vis_df = pd.concat([vis_df_pos, vis_df_neg], axis=0).sample(frac=1, random_state=SEED).reset_index(drop=True)

    model.eval()

    x = x.to(device)
    message_edge_index = message_edge_index.to(device)
    h = model.encode(x, message_edge_index)

    edge_pairs = torch.tensor(
        vis_df[["drug1_idx", "drug2_idx"]].values,
        dtype=torch.long
    ).t().contiguous().to(device)

    pair_feat = model.build_pair_feature(h, x, edge_pairs).cpu().numpy()

    # 先 PCA 再 t-SNE，提高稳定性
    pca_dim = min(TSNE_PCA_DIM, pair_feat.shape[0] - 1, pair_feat.shape[1])
    if pca_dim >= 2:
        pair_feat_reduced = PCA(n_components=pca_dim, random_state=SEED).fit_transform(pair_feat)
    else:
        pair_feat_reduced = pair_feat

    n_samples = pair_feat_reduced.shape[0]
    perplexity = min(TSNE_PERPLEXITY, max(5, n_samples // 10))
    if perplexity >= n_samples:
        perplexity = max(2, n_samples - 1)

    tsne = TSNE(
        n_components=2,
        random_state=SEED,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto"
    )
    pair_2d = tsne.fit_transform(pair_feat_reduced)

    df_tsne = vis_df.copy()
    df_tsne["tsne_1"] = pair_2d[:, 0]
    df_tsne["tsne_2"] = pair_2d[:, 1]
    df_tsne.to_csv(save_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_tsne,
        x="tsne_1",
        y="tsne_2",
        hue="true_label",
        palette={0: "#1f77b4", 1: "#d62728"},
        alpha=0.75,
        s=35,
        edgecolor=None
    )
    plt.title("t-SNE Visualization of Pair Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="True Label", labels=["No Interaction", "Interaction"])
    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    plt.close()

    print(f"[Saved] Pair t-SNE 坐标: {save_csv}")
    print(f"[Saved] Pair t-SNE 图像: {save_fig}")

    return df_tsne


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

    torch.save(model.state_dict(), os.path.join(save_dir, "best_gcn_model.pth"))
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
        "TSNE_BALANCED_VIS": TSNE_BALANCED_VIS,
        "TSNE_MAX_POINTS_PER_CLASS": TSNE_MAX_POINTS_PER_CLASS,
        "TSNE_PCA_DIM": TSNE_PCA_DIM,
        "TSNE_PERPLEXITY": TSNE_PERPLEXITY
    }

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[Saved] 模型与工件目录: {save_dir}")


# =========================================================
# 案例分析
# =========================================================
def analyze_cases(df_pred, save_txt_path, top_k=5):
    lines = []

    def add_section(title, df_case):
        lines.append(f"\n{'=' * 20} {title} {'=' * 20}")
        if df_case.empty:
            lines.append("无案例")
        else:
            lines.append(df_case.to_string(index=False))

    cols = ["drug1_smiles", "drug2_smiles", "true_label", "pred_prob", "pred_label"]

    tp = df_pred[
        (df_pred["true_label"] == 1) & (df_pred["pred_label"] == 1)
    ].sort_values("pred_prob", ascending=False).head(top_k)

    tn = df_pred[
        (df_pred["true_label"] == 0) & (df_pred["pred_label"] == 0)
    ].sort_values("pred_prob", ascending=True).head(top_k)

    fp = df_pred[
        (df_pred["true_label"] == 0) & (df_pred["pred_label"] == 1)
    ].sort_values("pred_prob", ascending=False).head(top_k)

    fn = df_pred[
        (df_pred["true_label"] == 1) & (df_pred["pred_label"] == 0)
    ].sort_values("pred_prob", ascending=True).head(top_k)

    add_section("高置信正确预测（真阳性）", tp[cols] if not tp.empty else tp)
    add_section("高置信正确预测（真阴性）", tn[cols] if not tn.empty else tn)
    add_section("高置信错误预测（假阳性）", fp[cols] if not fp.empty else fp)
    add_section("高置信错误预测（假阴性）", fn[cols] if not fn.empty else fn)

    lines.append("\n" + "=" * 20 + " 案例分析说明 " + "=" * 20)
    lines.append("1. 真阳性说明模型能够识别测试集中真实存在相互作用的药物对。")
    lines.append("2. 真阴性说明模型能够排除部分无相互作用药物对。")
    lines.append("3. 假阳性可能来自结构相似性、图传播偏差、潜在未标注相互作用。")
    lines.append("4. 假阴性可能说明相互作用模式复杂，单纯依赖图结构和分子指纹仍有局限。")

    text = "\n".join(lines)
    print(text)

    with open(save_txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[Saved] 案例分析: {save_txt_path}")


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
    global idx_to_smiles_global
    drug_to_idx, idx_to_smiles = build_global_drug_vocab(df_train, df_test)
    idx_to_smiles_global = idx_to_smiles
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
    df_test_records = build_test_records(df_test, drug_to_idx)
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
    print("6. 初始化模型")
    print("=" * 70)
    model = GCNLinkPredictor(
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

    best_model_path = os.path.join(ARTIFACT_DIR, "best_gcn_model.pth")
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

            # 以 AUPR 优先、AUC 辅助做模型选择（更适合不均衡场景）
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

    embeddings = get_node_embeddings(model, node_features, edge_train, DEVICE)
    emb_path = os.path.join(ARTIFACT_DIR, "drug_embeddings.pt")
    torch.save(embeddings.cpu(), emb_path)
    print(f"[Saved] 节点嵌入: {emb_path}")

    pred_csv_path = os.path.join(ARTIFACT_DIR, "test_predictions.csv")
    df_pred.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] 测试集预测结果: {pred_csv_path}")

    metrics_path = os.path.join(ARTIFACT_DIR, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    print(f"[Saved] 测试集指标: {metrics_path}")

    print("\n" + "=" * 70)
    print("11. 绘制 ROC / PR / 混淆矩阵")
    print("=" * 70)
    roc_path = os.path.join(ARTIFACT_DIR, "roc_curve.png")
    pr_path = os.path.join(ARTIFACT_DIR, "pr_curve.png")
    cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")

    plot_roc_curve(df_pred, roc_path)
    plot_pr_curve(df_pred, pr_path)
    plot_confusion_matrix(df_pred, cm_path)

    print("\n" + "=" * 70)
    print("12. Pair Embedding t-SNE 可视化（双颜色）")
    print("=" * 70)
    tsne_csv_path = os.path.join(ARTIFACT_DIR, "pair_tsne_embeddings.csv")
    tsne_fig_path = os.path.join(ARTIFACT_DIR, "pair_tsne_plot.png")
    visualize_pair_tsne(
        model=model,
        x=node_features,
        message_edge_index=edge_train,
        df_records=df_test_records,
        save_csv=tsne_csv_path,
        save_fig=tsne_fig_path,
        device=DEVICE,
        balanced_vis=TSNE_BALANCED_VIS,
        max_points_per_class=TSNE_MAX_POINTS_PER_CLASS
    )

    print("\n" + "=" * 70)
    print("13. 案例分析")
    print("=" * 70)
    case_txt_path = os.path.join(ARTIFACT_DIR, "case_analysis.txt")
    analyze_cases(df_pred, save_txt_path=case_txt_path, top_k=5)

    print("\n" + "=" * 70)
    print("全部完成")
    print("=" * 70)
    print(f"输出目录: {os.path.abspath(ARTIFACT_DIR)}")
    print("\n建议论文中重点汇报：")
    print("1. 测试集 AUC / AUPR / F1 / Precision / Recall")
    print("2. ROC 与 PR 曲线")
    print("3. Pair embedding 的 t-SNE 可视化")
    print("4. 案例分析（TP/TN/FP/FN）")


# =========================================================
# 入口
# =========================================================
if __name__ == "__main__":
    # 为了兼容 build_test_records 中使用全局 idx_to_smiles_global
    idx_to_smiles_global = {}
    main()