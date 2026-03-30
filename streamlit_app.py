# -*- coding: utf-8 -*-
"""
Streamlit 应用：对应融合分子指纹特征的 GCN 的 DDI 预测界面
"""

import os
import io
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")


# =========================================================
# 页面设置
# =========================================================
st.set_page_config(
    page_title="DDI Prediction System",
    page_icon="🧪",
    layout="wide"
)

# =========================================================
# 路径配置
# 建议这里先用绝对路径，最稳
# ============================================
# =============
ARTIFACT_DIR = r"./I_GCN_Results"

MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_gcn_model.pth")
NODE_FEATURES_PATH = os.path.join(ARTIFACT_DIR, "node_features.pt")
EDGE_TRAIN_PATH = os.path.join(ARTIFACT_DIR, "edge_train.pt")
DRUG_TO_IDX_PATH = os.path.join(ARTIFACT_DIR, "drug_to_idx.json")
IDX_TO_SMILES_PATH = os.path.join(ARTIFACT_DIR, "idx_to_smiles.json")
CONFIG_PATH = os.path.join(ARTIFACT_DIR, "config.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 工具函数
# =========================================================
def canonicalize_edge(u, v):
    return (u, v) if u < v else (v, u)


def mol_from_smiles(smiles):
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None
    return Chem.MolFromSmiles(smiles)


def validate_smiles(smiles):
    return mol_from_smiles(smiles) is not None


def smiles_to_fingerprint(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
    return np.array(fp, dtype=np.float32)


def draw_molecule(smiles, size=(300, 300)):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =========================================================
# 模型定义（与训练代码一致）
# =========================================================
class GCNLinkPredictor(nn.Module):
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


# =========================================================
# 加载模型与工件
# =========================================================
def load_artifacts():
    required_files = [
        MODEL_PATH,
        NODE_FEATURES_PATH,
        EDGE_TRAIN_PATH,
        DRUG_TO_IDX_PATH,
        IDX_TO_SMILES_PATH,
        CONFIG_PATH
    ]

    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("缺少以下工件文件：\n" + "\n".join(missing))

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(DRUG_TO_IDX_PATH, "r", encoding="utf-8") as f:
        drug_to_idx = json.load(f)

    with open(IDX_TO_SMILES_PATH, "r", encoding="utf-8") as f:
        idx_to_smiles_raw = json.load(f)
    idx_to_smiles = {int(k): v for k, v in idx_to_smiles_raw.items()}

    node_features = torch.load(NODE_FEATURES_PATH, map_location=DEVICE).float()
    edge_train = torch.load(EDGE_TRAIN_PATH, map_location=DEVICE).long()

    model = GCNLinkPredictor(
        in_dim=int(config["NBITS"]),
        hidden_dim=int(config["HIDDEN_DIM"]),
        out_dim=1,
        dropout=float(config["DROPOUT"])
    ).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "node_features": node_features,
        "edge_train": edge_train,
        "drug_to_idx": drug_to_idx,
        "idx_to_smiles": idx_to_smiles,
        "config": config,
        "best_threshold": float(config.get("BEST_THRESHOLD_FROM_VAL", 0.5))
    }


# =========================================================
# 动态扩展特征
# =========================================================
def prepare_features_for_smiles(smiles_list, artifacts):
    base_features = artifacts["node_features"]
    drug_to_idx = dict(artifacts["drug_to_idx"])
    idx_to_smiles = dict(artifacts["idx_to_smiles"])
    config = artifacts["config"]

    x = base_features.clone()

    for smi in smiles_list:
        if smi in drug_to_idx:
            continue

        fp = smiles_to_fingerprint(
            smi,
            radius=int(config["RADIUS"]),
            nbits=int(config["NBITS"])
        )
        if fp is None:
            raise ValueError(f"非法 SMILES：{smi}")

        new_idx = len(drug_to_idx)
        drug_to_idx[smi] = new_idx
        idx_to_smiles[new_idx] = smi

        fp_tensor = torch.tensor(fp, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        x = torch.cat([x, fp_tensor], dim=0)

    return x, drug_to_idx, idx_to_smiles


# =========================================================
# 预测函数
# =========================================================
@torch.no_grad()
def predict_pair(smiles1, smiles2, artifacts):
    model = artifacts["model"]
    edge_train = artifacts["edge_train"]
    threshold = artifacts["best_threshold"]

    x, drug_to_idx, _ = prepare_features_for_smiles([smiles1, smiles2], artifacts)
    h = model.encode(x, edge_train)

    u = drug_to_idx[smiles1]
    v = drug_to_idx[smiles2]
    u, v = canonicalize_edge(u, v)

    edge_pair = torch.tensor([[u], [v]], dtype=torch.long, device=DEVICE)
    logit = model.decode(h, x, edge_pair)
    prob = torch.sigmoid(logit).item()
    pred = int(prob >= threshold)

    return prob, pred, threshold


@torch.no_grad()
def predict_batch(df_input, artifacts):
    if "smile1" not in df_input.columns or "smile2" not in df_input.columns:
        raise ValueError("CSV 文件必须包含 smile1 和 smile2 两列。")

    df = df_input.copy()
    df["smile1"] = df["smile1"].astype(str).str.strip()
    df["smile2"] = df["smile2"].astype(str).str.strip()

    valid_mask = df["smile1"].apply(validate_smiles) & df["smile2"].apply(validate_smiles)
    df["is_valid_smiles"] = valid_mask.astype(int)

    valid_smiles = pd.concat([
        df.loc[valid_mask, "smile1"],
        df.loc[valid_mask, "smile2"]
    ]).unique().tolist()

    if len(valid_smiles) == 0:
        df["pred_prob"] = np.nan
        df["pred_label"] = np.nan
        df["pred_class"] = "Invalid SMILES"
        return df

    model = artifacts["model"]
    edge_train = artifacts["edge_train"]
    threshold = artifacts["best_threshold"]

    x, drug_to_idx, _ = prepare_features_for_smiles(valid_smiles, artifacts)
    h = model.encode(x, edge_train)

    pred_probs = []
    pred_labels = []
    pred_classes = []

    for _, row in df.iterrows():
        s1, s2 = row["smile1"], row["smile2"]

        if not validate_smiles(s1) or not validate_smiles(s2):
            pred_probs.append(np.nan)
            pred_labels.append(np.nan)
            pred_classes.append("Invalid SMILES")
            continue

        u = drug_to_idx[s1]
        v = drug_to_idx[s2]
        u, v = canonicalize_edge(u, v)

        edge_pair = torch.tensor([[u], [v]], dtype=torch.long, device=DEVICE)
        logit = model.decode(h, x, edge_pair)
        prob = torch.sigmoid(logit).item()
        pred = int(prob >= threshold)

        pred_probs.append(prob)
        pred_labels.append(pred)
        pred_classes.append("Interaction" if pred == 1 else "No Interaction")

    df["pred_prob"] = pred_probs
    df["pred_label"] = pred_labels
    df["pred_class"] = pred_classes

    return df


# =========================================================
# 主页面
# =========================================================
def main():
    st.title("🧪 药物相互作用预测系统（GCN）")
    st.markdown("基于融合分子指纹特征的GCN模型，支持 **单条预测** 与 **批量预测**。")

    with st.spinner("正在加载模型与工件，请稍候..."):
        try:
            artifacts = load_artifacts()
        except Exception as e:
            st.error(f"模型加载失败：{e}")
            st.stop()

    st.sidebar.header("系统信息")
    st.sidebar.write(f"设备：{DEVICE}")
    st.sidebar.write(f"最佳阈值：{artifacts['best_threshold']:.2f}")
    st.sidebar.write(f"指纹维度：{artifacts['config']['NBITS']}")
    st.sidebar.write(f"隐藏维度：{artifacts['config']['HIDDEN_DIM']}")
    st.sidebar.write(f"训练图边数：{artifacts['edge_train'].shape[1]}")
    st.sidebar.write(f"节点数：{artifacts['node_features'].shape[0]}")

    tab1, tab2, tab3 = st.tabs(["单条预测", "批量预测", "系统说明"])

    with tab1:
        st.subheader("手动输入两个药物的 SMILES")

        c1, c2 = st.columns(2)
        with c1:
            smiles1 = st.text_area(
                "药物 1 的 SMILES",
                value="CC(=O)OC1=CC=CC=C1C(=O)O",
                height=120
            )
        with c2:
            smiles2 = st.text_area(
                "药物 2 的 SMILES",
                value="CN1CCC[C@H]1C2=CN=CC=C2",
                height=120
            )

        if st.button("开始预测", key="single_predict"):
            s1 = smiles1.strip()
            s2 = smiles2.strip()

            if not validate_smiles(s1):
                st.error("药物 1 的 SMILES 无效。")
            elif not validate_smiles(s2):
                st.error("药物 2 的 SMILES 无效。")
            else:
                with st.spinner("正在预测..."):
                    prob, pred, thr = predict_pair(s1, s2, artifacts)

                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**药物 1 分子结构**")
                    img1 = draw_molecule(s1)
                    if img1:
                        st.image(img1, use_container_width=True)
                    st.code(s1)

                with d2:
                    st.markdown("**药物 2 分子结构**")
                    img2 = draw_molecule(s2)
                    if img2:
                        st.image(img2, use_container_width=True)
                    st.code(s2)

                m1, m2, m3 = st.columns(3)
                m1.metric("相互作用概率", f"{prob:.4f}")
                m2.metric("判定阈值", f"{thr:.2f}")
                m3.metric("预测标签", "Interaction" if pred == 1 else "No Interaction")

                if pred == 1:
                    st.success("预测结果：两药存在相互作用。")
                else:
                    st.info("预测结果：两药不存在明显相互作用。")

    with tab2:
        st.subheader("上传 CSV 文件进行批量预测")
        st.write("CSV 至少需要两列：`smile1` 和 `smile2`。")

        uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"])

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.markdown("**原始数据预览**")
                st.dataframe(df_upload, use_container_width=True)
            except Exception as e:
                st.error(f"CSV 读取失败：{e}")
                st.stop()

            if st.button("批量预测", key="batch_predict"):
                with st.spinner("正在批量预测，请稍候..."):
                    try:
                        df_result = predict_batch(df_upload, artifacts)
                    except Exception as e:
                        st.error(f"批量预测失败：{e}")
                        st.stop()

                st.markdown("**预测结果**")
                st.dataframe(df_result, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                valid_count = int(df_result["is_valid_smiles"].sum())
                interaction_count = int((df_result["pred_label"] == 1).sum())
                no_interaction_count = int((df_result["pred_label"] == 0).sum())

                c1.metric("有效样本数", valid_count)
                c2.metric("预测为 Interaction", interaction_count)
                c3.metric("预测为 No Interaction", no_interaction_count)

                csv_data = df_result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="下载预测结果 CSV",
                    data=csv_data,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv"
                )

    with tab3:
        st.subheader("系统说明")
        st.markdown(
            """
### 功能简介
本系统用于药物相互作用（DDI）预测，支持：

1. **单条预测**
   - 输入两个药物的 SMILES
   - RDKit 校验分子合法性
   - 预测相互作用概率

2. **批量预测**
   - 上传包含 `smile1`、`smile2` 的 CSV 文件
   - 前端展示数据
   - 进行批量推理
   - 下载预测结果

### 注意
- 本系统输出为模型预测结果，仅供科研参考；
- 不能替代真实药理实验或临床验证。
            """
        )


if __name__ == "__main__":
    main()