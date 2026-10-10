import os
import json
import pm4py
import torch
import random
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# =====================================================
# 固定随机种子
# =====================================================

seed = 99

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["PYTHONHASHSEED"] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =====================================================
# 路径
# =====================================================

BASE_DIR = r"/pm_cp_adaptation-main/pm_cp_adaptation-main/Data/Output/SingleDisease/Method/Road"

SAVE_DIR = r"/pm_cp_adaptation-main/pm_cp_adaptation-main/MyCleanModels/Road"

os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_XES = os.path.join(BASE_DIR, "L_TR.xes")

# =====================================================
# 参数
# =====================================================

MASK_PROB = 0.15

MAX_LEN = 64

BATCH_SIZE = 32

FEATURE_BATCH = 256

EPOCHS = 15

KEEP_RATIO = 0.70

MIN_ACTIVITY_COUNT = 10

NOISE_THRESHOLD = 0.30 #

# 对比学习权重，先使用较保守的0.10
CONTRASTIVE_WEIGHT = 0.08

# =====================================================
# TGPD-V2 参数
# =====================================================

# 低于该分数的直接跟随关系视为弱结构关系
EDGE_THRESHOLD = 0.20

# TGPD结构异常分数在总异常分数中的权重
TGPD_STRUCTURE_WEIGHT = 0.20

# Heuristics Miner动态dependency threshold范围
# 不建议低于0.50，否则容易增加弱边
HM_DEP_MIN = 0.52
HM_DEP_MAX = 0.62


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\n使用设备: {device}")

# =====================================================
# Dataset
# =====================================================

class ActivityDataset(Dataset):

    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):

        return torch.tensor(
            self.seqs[idx],
            dtype=torch.long
        )

# =====================================================
# Transformer
# =====================================================

class TransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=32,
        nhead=2,
        num_layers=2,
        max_len=64
    ):

        super().__init__()

        self.embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_len, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.token_head = nn.Linear(
            embed_dim,
            vocab_size
        )

    def encode(self, x, padding_mask):

        x = self.embed(x)

        x = x + self.pos_embed[:, :x.size(1), :]

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        return self.norm(x)

    def encode_with_attention(self, x, padding_mask):
        """
        在保持 TransformerEncoderLayer 原始前向传播逻辑的同时，
        提取每一层的 self-attention 权重。
        """

        x = self.embed(x)

        x = x + self.pos_embed[:, :x.size(1), :]

        layer_attns = []

        for layer in self.transformer.layers:

            # -------------------------------------------------
            # 兼容不同PyTorch版本的MultiheadAttention
            # -------------------------------------------------

            def run_self_attention(attn_input):

                try:

                    attn_out, attn_w = layer.self_attn(
                        attn_input,
                        attn_input,
                        attn_input,
                        key_padding_mask=padding_mask,
                        need_weights=True,
                        average_attn_weights=False
                    )

                except TypeError:

                    attn_out, attn_w = layer.self_attn(
                        attn_input,
                        attn_input,
                        attn_input,
                        key_padding_mask=padding_mask,
                        need_weights=True
                    )

                # 新版PyTorch可能返回:
                # [batch, heads, seq_len, seq_len]
                if attn_w is not None and attn_w.dim() == 4:
                    attn_w = attn_w.mean(dim=1)

                return attn_out, attn_w

            # -------------------------------------------------
            # norm_first=True：Pre-Norm Transformer
            # -------------------------------------------------

            if getattr(layer, "norm_first", False):

                norm_x = layer.norm1(x)

                attn_out, attn_w = run_self_attention(norm_x)

                x = x + layer.dropout1(attn_out)

                x = x + layer._ff_block(
                    layer.norm2(x)
                )

            # -------------------------------------------------
            # norm_first=False：PyTorch默认Post-Norm
            # -------------------------------------------------

            else:

                attn_out, attn_w = run_self_attention(x)

                x = layer.norm1(
                    x + layer.dropout1(attn_out)
                )

                x = layer.norm2(
                    x + layer._ff_block(x)
                )

            layer_attns.append(attn_w)

        return self.norm(x), layer_attns

    def forward(self, x, padding_mask):

        x = self.encode(x, padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1).float()

        x = (
            (x * valid_mask).sum(dim=1)
            / valid_mask.sum(dim=1).clamp_min(1.0)
        )

        return x

# =====================================================
# 对比损失
# =====================================================

def contrastive_loss(z1, z2, temperature=0.2):

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.t()) / temperature

    labels = torch.arange(
        z1.size(0),
        device=z1.device
    )

    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )

# =====================================================
# 特征学习
# =====================================================

def get_trace_features(log):
    traces = []

    case_ids = []

    for trace_index, trace in enumerate(log):

        acts = []

        for e in trace:
            acts.append(e["concept:name"])

        traces.append(acts)

        case_id = trace.attributes.get(
            "concept:name",
            f"case_{trace_index}"
        )

        case_ids.append(str(case_id))

    # =====================================================
    # 活动编码
    # =====================================================

    all_acts = sorted(list({
        a for t in traces for a in t
    }))

    act2id = {
        a: i + 1
        for i, a in enumerate(all_acts)
    }

    mask_id = len(all_acts) + 1

    # =====================================================
    # 截断 + padding
    # =====================================================

    seqs = []

    for t in traces:

        ids = [
            act2id[a]
            for a in t[:MAX_LEN]
        ]

        ids += [0] * (MAX_LEN - len(ids))

        seqs.append(ids)

    # =====================================================
    # DataLoader
    # =====================================================

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        ActivityDataset(seqs),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g
    )

    # =====================================================
    # 模型
    # =====================================================

    model = TransformerEncoder(
        len(all_acts) + 2,
        max_len=MAX_LEN
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    ce = nn.CrossEntropyLoss()

    # =====================================================
    # 训练
    # =====================================================

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0

        for batch in loader:

            batch = batch.to(device)

            pad_mask = batch.eq(0)

            mask_pos = (
                (~pad_mask)
                & (
                    torch.rand_like(batch.float())
                    < MASK_PROB
                )
            )

            if not mask_pos.any():
                mask_pos[0, 0] = True

            inp = batch.clone()

            inp[mask_pos] = mask_id

            opt.zero_grad()

            rep = model.encode(
                inp,
                pad_mask
            )

            loss_mlm = ce(
                model.token_head(rep)[mask_pos],
                batch[mask_pos]
            )

            h1 = model(inp, pad_mask)

            h2 = model(batch, pad_mask)

            loss = (
                loss_mlm
                + CONTRASTIVE_WEIGHT *
                contrastive_loss(h1, h2)
            )

            loss.backward()

            opt.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss: {total_loss:.4f}"
        )

    # =====================================================
    # 分batch提取特征（防OOM）
    # =====================================================

    model.eval()

    all_features = []

    with torch.no_grad():

        for i in range(0, len(seqs), FEATURE_BATCH):

            batch_seqs = seqs[
                i:i + FEATURE_BATCH
            ]

            tensor = torch.tensor(
                batch_seqs,
                dtype=torch.long
            ).to(device)

            feats = model(
                tensor,
                tensor.eq(0)
            )

            all_features.append(
                feats.cpu().numpy()
            )

    features = np.concatenate(
        all_features,
        axis=0
    )

    id2act = {i: a for a, i in act2id.items()}

    return features,model,act2id,id2act,seqs,case_ids


# =====================================================
# TGPD: Attention Edge Score & Activity Importance
# =====================================================

def extract_tgpd_scores(model, seqs, id2act):

    print("\n===== TGPD: 提取直接跟随 Attention 分数 =====")

    model.eval()

    edge_accum = defaultdict(float)
    edge_count = defaultdict(int)

    act_in_sum = defaultdict(float)
    act_in_cnt = defaultdict(int)

    attn_heatmap = defaultdict(float)
    heatmap_cnt = defaultdict(int)

    with torch.no_grad():

        for i in range(0, len(seqs), FEATURE_BATCH):

            batch_seqs = seqs[
                i:i + FEATURE_BATCH
            ]

            tensor = torch.tensor(
                batch_seqs,
                dtype=torch.long
            ).to(device)

            pad_mask = tensor.eq(0)

            _, layer_attns = model.encode_with_attention(
                tensor,
                pad_mask
            )

            valid_layer_attns = [
                attn
                for attn in layer_attns
                if attn is not None
            ]

            if len(valid_layer_attns) == 0:
                continue

            # 所有Transformer层取平均
            attn = torch.stack(
                valid_layer_attns,
                dim=0
            ).mean(dim=0)

            # 兼容仍然含head维度的情况
            if attn.dim() == 4:
                attn = attn.mean(dim=1)

            for b in range(tensor.size(0)):

                valid_len = int(
                    (~pad_mask[b]).sum().item()
                )

                if valid_len < 1:
                    continue

                acts = [
                    id2act[int(tensor[b, j].item())]
                    for j in range(valid_len)
                ]

                mat = attn[
                    b,
                    :valid_len,
                    :valid_len
                ].cpu().numpy()

                # -----------------------------------------
                # Activity importance
                # 使用平均被关注程度，避免长trace天然分数更高
                # -----------------------------------------

                for pos in range(valid_len):

                    received_attention = (
                        mat[:, pos].sum()
                        - mat[pos, pos]
                    ) / max(valid_len - 1, 1)

                    act = acts[pos]

                    act_in_sum[act] += float(
                        received_attention
                    )

                    act_in_cnt[act] += 1

                # -----------------------------------------
                # Edge importance
                # 只统计真实的直接跟随关系
                # -----------------------------------------

                for pos in range(valid_len - 1):

                    src = acts[pos]
                    dst = acts[pos + 1]

                    # Transformer不是因果Attention，
                    # 因此使用相邻位置双向Attention的平均值
                    pair_attention = 0.5 * (
                        float(mat[pos, pos + 1])
                        + float(mat[pos + 1, pos])
                    )

                    edge_key = f"{src}->{dst}"

                    edge_accum[edge_key] += pair_attention
                    edge_count[edge_key] += 1

                    heat_key = (src, dst)

                    attn_heatmap[heat_key] += pair_attention
                    heatmap_cnt[heat_key] += 1

    # =====================================================
    # Activity score
    # =====================================================

    raw_activity_score = {}

    for act, total in act_in_sum.items():

        raw_activity_score[act] = (
            total
            / max(act_in_cnt[act], 1)
        )

    # =====================================================
    # Edge score
    # =====================================================

    raw_edge_weight = {}

    for key, total in edge_accum.items():

        raw_edge_weight[key] = (
            total
            / max(edge_count[key], 1)
        )

    # =====================================================
    # 使用5%-95%分位数做稳健归一化
    # 避免单个最大值把其他边全部压得很低
    # =====================================================

    def robust_normalize(score_dict):

        if not score_dict:
            return {}

        values = np.array(
            list(score_dict.values()),
            dtype=np.float32
        )

        low = float(
            np.percentile(values, 5)
        )

        high = float(
            np.percentile(values, 95)
        )

        if high - low < 1e-8:

            return {
                key: 1.0
                for key in score_dict
            }

        normalized = {}

        for key, value in score_dict.items():

            score = (
                float(value) - low
            ) / (high - low)

            score = float(
                np.clip(score, 0.0, 1.0)
            )

            normalized[key] = round(
                score,
                4
            )

        return normalized

    activity_score = robust_normalize(
        raw_activity_score
    )

    edge_weight = robust_normalize(
        raw_edge_weight
    )

    # 热力图保留原始平均Attention
    heatmap_matrix = {}

    for key, total in attn_heatmap.items():

        heatmap_matrix[key] = (
            total
            / max(heatmap_cnt[key], 1)
        )

    strong_edge_count = sum(
        value >= EDGE_THRESHOLD
        for value in edge_weight.values()
    )

    print(
        f"直接跟随边数量: {len(edge_weight)}"
    )

    print(
        f"强Attention边数量: {strong_edge_count}"
    )

    return (
        activity_score,
        edge_weight,
        heatmap_matrix
    )

def save_tgpd_json(activity_score, edge_weight):

    with open(
        os.path.join(SAVE_DIR, "activity_score.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            activity_score,
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(
        os.path.join(SAVE_DIR, "edge_weight.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            edge_weight,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("TGPD 权重已保存: activity_score.json, edge_weight.json")

def save_attention_heatmap(heatmap_matrix, activity_score):

    activities = sorted(
        activity_score.keys(),
        key=lambda a: activity_score[a],
        reverse=True
    )

    n = len(activities)

    if n == 0:
        return

    mat = np.zeros((n, n))

    act_idx = {a: i for i, a in enumerate(activities)}

    for (src, dst), val in heatmap_matrix.items():

        if src in act_idx and dst in act_idx:
            mat[act_idx[src], act_idx[dst]] = val

    row_max = mat.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    mat = mat / row_max

    fig, ax = plt.subplots(
        figsize=(max(8, n * 0.35), max(6, n * 0.35))
    )

    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(activities, rotation=90, fontsize=7)
    ax.set_yticklabels(activities, fontsize=7)
    ax.set_xlabel("Target Activity")
    ax.set_ylabel("Source Activity")
    ax.set_title("Transformer Attention Heatmap (TGPD)")

    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()

    plt.savefig(
        os.path.join(SAVE_DIR, "attention_heatmap.png"),
        dpi=150
    )

    plt.close()

    print("Attention 热力图已保存: attention_heatmap.png")

def discover_petri_net_tgpd(
        clean_log,
        edge_weight
):

    print(
        "\nTGPD Guided HM"
    )

    # =========================
    # Transformer Attention
    # =========================

    att_values = list(
        edge_weight.values()
    )

    if len(att_values) == 0:

        avg_attention = 0.5

    else:

        avg_attention = np.mean(
            att_values
        )

    # =========================
    # 动态HM阈值
    # =========================

    dep_threshold = compute_dynamic_threshold(
        edge_weight
    )

    dep_threshold = min(
        dep_threshold,
        0.60
    )

    print(
        f"Average Attention: "
        f"{avg_attention:.4f}"
    )

    print(
        f"Dependency Threshold: "
        f"{dep_threshold:.4f}"
    )

    net, im, fm = pm4py.discover_petri_net_heuristics(
        clean_log,
        dependency_threshold=dep_threshold,
        and_threshold=0.82
    )

    return net, im, fm

def save_weighted_dfg(clean_log, edge_weight):

    print("\nTGPD: 生成加权 DFG ...")

    dfg, start_acts, end_acts = pm4py.discover_dfg(clean_log)

    weighted_dfg = {}

    for (src, dst), freq in dfg.items():

        key = f"{src}->{dst}"

        w = float(
            edge_weight.get(key, 0.0)
        )

        # Weighted DFG只展示TGPD认为可靠的结构关系
        if w < EDGE_THRESHOLD:
            continue

        weighted_dfg[(src, dst)] = round(
            float(freq) * (
                    1.0 + 0.5 * w
            ),
            4
        )

    dfg_json = {
        f"{src}->{dst}": v
        for (src, dst), v in weighted_dfg.items()
    }

    with open(
        os.path.join(SAVE_DIR, "weighted_dfg.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dfg_json,
            f,
            indent=2,
            ensure_ascii=False
        )

    try:
        gviz = pm4py.visualization.dfg.visualizer.apply(
            weighted_dfg,
            parameters={
                "start_activities": start_acts,
                "end_activities": end_acts,
            }
        )

        pm4py.visualization.dfg.visualizer.save(
            gviz,
            os.path.join(SAVE_DIR, "weighted_dfg.png")
        )

        print("加权 DFG 已保存: weighted_dfg.json, weighted_dfg.png")

    except Exception as exc:

        print(f"加权 DFG 可视化跳过: {exc}")

def compute_dynamic_threshold(edge_weight):
    """
    根据直接跟随边Attention分布，
    动态确定Heuristics Miner dependency threshold。

    Attention越集中、强弱边区分越明显，
    结构可信度越高，阈值可接近0.50。

    Attention越分散，
    使用更严格的阈值，最高到0.62。
    """

    values = np.array(
        list(edge_weight.values()),
        dtype=np.float32
    )

    if len(values) == 0:

        print(
            "\n没有可用Attention边，"
            "使用默认dependency threshold=0.55"
        )

        return 0.55

    if len(values) == 1:
        return 0.55

    mean_att = float(
        values.mean()
    )

    std_att = float(
        values.std()
    )

    # =====================================================
    # Attention entropy
    # =====================================================

    probability = (
        values + 1e-8
    ) / (
        values.sum() + 1e-8
    )

    entropy = -np.sum(
        probability
        * np.log(probability + 1e-8)
    )

    entropy = entropy / max(
        np.log(len(values)),
        1e-8
    )

    entropy = float(
        np.clip(entropy, 0.0, 1.0)
    )

    # =====================================================
    # 强边和普通边的分离程度
    # =====================================================

    q80 = float(
        np.percentile(values, 80)
    )

    strong_values = values[
        values >= q80
    ]

    weak_values = values[
        values < q80
    ]

    strong_mean = float(
        strong_values.mean()
    ) if len(strong_values) else mean_att

    weak_mean = float(
        weak_values.mean()
    ) if len(weak_values) else mean_att

    separation = float(
        np.clip(
            strong_mean - weak_mean,
            0.0,
            1.0
        )
    )

    # =====================================================
    # Process confidence
    # =====================================================

    confidence = (
        0.60 * separation
        + 0.40 * (1.0 - entropy)
    )

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0
        )
    )

    # confidence高时接近0.50
    # confidence低时接近0.62
    dep_threshold = (
        HM_DEP_MAX
        - (
            HM_DEP_MAX - HM_DEP_MIN
        ) * confidence
    )

    dep_threshold = float(
        np.clip(
            dep_threshold,
            HM_DEP_MIN,
            HM_DEP_MAX
        )
    )

    print("\n===== TGPD Adaptive Threshold =====")

    print(
        f"Mean Attention       : {mean_att:.4f}"
    )

    print(
        f"Std Attention        : {std_att:.4f}"
    )

    print(
        f"Attention Entropy    : {entropy:.4f}"
    )

    print(
        f"Strong/Weak Separation: {separation:.4f}"
    )

    print(
        f"Process Confidence   : {confidence:.4f}"
    )

    print(
        f"Dependency Threshold : {dep_threshold:.4f}"
    )

    return dep_threshold


# =====================================================
# 异常过滤
# =====================================================

def filter_by_dl_anomaly_score(
        df,
        features,
        case_ids,
        seqs,
        id2act,
        edge_weight
):

    print("\n开始TGPD联合异常检测...")

    n_samples = len(features)

    if n_samples == 0:
        raise ValueError("features为空，无法进行异常检测")

    if len(case_ids) != n_samples:
        raise ValueError(
            "case_ids数量与features数量不一致"
        )

    # =====================================================
    # 1. KNN距离异常分数
    # =====================================================

    knn_neighbors = min(
        10,
        n_samples
    )

    nbrs = NearestNeighbors(
        n_neighbors=knn_neighbors
    ).fit(features)

    distances = nbrs.kneighbors(
        features
    )[0]

    # 第一列通常是样本自身，距离为0
    if distances.shape[1] > 1:

        knn_scores = distances[
            :,
            1:
        ].mean(axis=1)

    else:

        knn_scores = distances.mean(axis=1)

    # =====================================================
    # 2. LOF连续异常分数
    # =====================================================

    if n_samples >= 3:

        lof_neighbors = min(
            20,
            n_samples - 1
        )

        lof = LocalOutlierFactor(
            n_neighbors=lof_neighbors
        )

        lof.fit_predict(features)

        # 数值越大，越异常
        lof_scores = (
            -lof.negative_outlier_factor_
        )

    else:

        lof_scores = np.zeros(
            n_samples,
            dtype=np.float32
        )

    # =====================================================
    # 3. TGPD流程结构一致性
    # =====================================================

    structure_scores = []

    for seq in seqs:

        valid_ids = [
            int(activity_id)
            for activity_id in seq
            if int(activity_id) != 0
        ]

        if len(valid_ids) < 2:

            structure_scores.append(0.0)
            continue

        trace_edge_scores = []

        for left_id, right_id in zip(
                valid_ids[:-1],
                valid_ids[1:]
        ):

            src = id2act[left_id]
            dst = id2act[right_id]

            key = f"{src}->{dst}"

            score = float(
                edge_weight.get(key, 0.0)
            )

            # 弱Attention关系按0处理
            if score < EDGE_THRESHOLD:
                score = 0.0

            trace_edge_scores.append(score)

        if trace_edge_scores:

            structure_score = float(
                np.mean(trace_edge_scores)
            )

        else:

            structure_score = 0.0

        structure_scores.append(
            structure_score
        )

    structure_scores = np.array(
        structure_scores,
        dtype=np.float32
    )

    # =====================================================
    # 4. 稳健标准化
    # =====================================================

    def robust_zscore(values):

        values = np.asarray(
            values,
            dtype=np.float32
        )

        median = np.median(values)

        mad = np.median(
            np.abs(values - median)
        )

        scale = 1.4826 * mad

        if scale < 1e-8:

            std = values.std()

            if std < 1e-8:
                return np.zeros_like(values)

            return (
                values - values.mean()
            ) / std

        return (
            values - median
        ) / scale

    knn_normalized = robust_zscore(
        knn_scores
    )

    lof_normalized = robust_zscore(
        lof_scores
    )

    # structure越低，结构异常程度越高
    structure_anomaly = robust_zscore(
        1.0 - structure_scores
    )

    # =====================================================
    # 5. 联合异常分数
    # 分数越低越正常
    # =====================================================

    scores = (
        knn_normalized
        + 0.5 * lof_normalized
        + TGPD_STRUCTURE_WEIGHT
        * structure_anomaly
    )

    k = max(
        1,
        int(n_samples * KEEP_RATIO)
    )

    sorted_idx = np.argsort(scores)

    keep_idx = sorted_idx[:k]

    keep_cases = {
        str(case_ids[index])
        for index in keep_idx
    }

    print(
        f"原始Case数: {n_samples}"
    )

    print(
        f"保留Case数: {len(keep_cases)}"
    )

    print(
        "平均结构一致性: "
        f"{structure_scores.mean():.4f}"
    )

    print(
        "保留Trace平均结构一致性: "
        f"{structure_scores[keep_idx].mean():.4f}"
    )

    case_column = (
        df["case:concept:name"]
        .astype(str)
    )

    return df[
        case_column.isin(keep_cases)
    ].copy()


# =====================================================
# 活动过滤
# =====================================================

def filter_low_freq_acts(df):

    cnt = df["concept:name"].value_counts()

    removed = cnt[
        cnt < MIN_ACTIVITY_COUNT
    ]

    keep = cnt[
        cnt >= MIN_ACTIVITY_COUNT
    ].index

    print("\n===== 低频活动过滤 =====")
    print(f"MIN_ACTIVITY_COUNT: {MIN_ACTIVITY_COUNT}")
    print(f"过滤前活动数: {len(cnt)}")
    print(f"删除活动数: {len(removed)}")

    if len(removed) > 0:
        print("删除的活动及频次:")
        for act, count in removed.items():
            print(f"  {act}: {count}")

    df_filtered = df[
        df["concept:name"].isin(keep)
    ].copy()

    cases_before = df_filtered[
        "case:concept:name"
    ].nunique()

    df_filtered = df_filtered.groupby(
        "case:concept:name"
    ).filter(lambda x: len(x) > 1)

    cases_after = df_filtered[
        "case:concept:name"
    ].nunique()

    print(f"因只剩1个事件而删除的Case数: {cases_before - cases_after}")
    print(f"过滤后活动数: {df_filtered['concept:name'].nunique()}")

    return df_filtered


# =====================================================
# 主流程
# =====================================================

def process_dataset():

    print("\n===== 读取日志 =====")

    log = xes_importer.apply(TRAIN_XES)

    df = pm4py.convert_to_dataframe(log)

    print(f"Trace数量: {len(log)}")
    print(f"Event数量: {len(df)}")

    # =====================================================
    # 深度特征学习
    # =====================================================

    (
        feats,
        model,
        act2id,
        id2act,
        seqs,
        case_ids
    ) = get_trace_features(log)

    # =====================================================
    # TGPD: Attention → Activity / Edge 分数
    # =====================================================

    activity_score, edge_weight, heatmap_matrix = (
        extract_tgpd_scores(model, seqs, id2act)
    )

    save_tgpd_json(activity_score, edge_weight)

    save_attention_heatmap(heatmap_matrix, activity_score)

    top_edges = sorted(
        edge_weight.items(),
        key=lambda x: x[1],
        reverse=True
    )[:8]

    print("\nTop Attention Edge Scores:")

    for key, val in top_edges:
        print(f"  {key} = {val}")

    top_acts = sorted(
        activity_score.items(),
        key=lambda x: x[1],
        reverse=True
    )[:8]

    print("\nTop Activity Importance:")

    for act, val in top_acts:
        print(f"  {act} = {val}")

    # =====================================================
    # TGPD联合异常过滤（KNN + 连续LOF + 结构一致性）
    # =====================================================

    df_clean = filter_by_dl_anomaly_score(
        df=df,
        features=feats,
        case_ids=case_ids,
        seqs=seqs,
        id2act=id2act,
        edge_weight=edge_weight
    )

    # =====================================================
    # 低频活动过滤
    # 与Transformer baseline保持一致，保证公平比较
    # =====================================================

    df_clean = filter_low_freq_acts(
        df_clean
    )

    # =====================================================
    # TGPD 活动过滤 (替代 MIN_ACTIVITY_COUNT)
    # =====================================================

    # df_clean = filter_by_activity_score(
    #     df_clean,
    #     activity_score
    # )

    print(f"\n清洗后Event数: {len(df_clean)}")

    print(
        f"清洗后Trace数: "
        f"{df_clean['case:concept:name'].nunique()}"
    )

    print(
        f"清洗后活动数: "
        f"{df_clean['concept:name'].nunique()}"
    )

    # =====================================================
    # 保存保留活动（新增）
    # =====================================================

    keep_activities = sorted(
        df_clean["concept:name"]
        .unique()
        .tolist()
    )

    print("\n保留活动:")
    print(keep_activities)

    # =====================================================
    # 转日志
    # =====================================================

    clean_log = log_converter.apply(df_clean)

    # =====================================================
    # 流程发现
    # =====================================================

    print("\n开始 TGPD 流程发现...")

    # clean_log = inject_attention_edges(
    #     clean_log,
    #     edge_weight
    # )

    net, im, fm = discover_petri_net_tgpd(
        clean_log,
        edge_weight
    )

    save_weighted_dfg(clean_log, edge_weight)

    # =====================================================
    # 保存PNML
    # =====================================================

    pm4py.write_pnml(
        net,
        im,
        fm,
        os.path.join(SAVE_DIR, "model.pnml")
    )

    # =====================================================
    # 保存流程图
    # =====================================================

    gviz = pm4py.visualization.petri_net.visualizer.apply(
        net,
        im,
        fm,
        parameters={"layout": "lr",
                    "format": "png",
        "debug": False}
    )

    pm4py.visualization.petri_net.visualizer.save(
        gviz,
        os.path.join(SAVE_DIR, "model.png")
    )

    # =====================================================
    # 保存配置（新增 keep_activities）
    # =====================================================

    cfg = {

        "method": "Explainable TGPD-V2",

        "keep_activities": keep_activities,

        "keep_ratio": KEEP_RATIO,

        "min_activity": MIN_ACTIVITY_COUNT,

        "noise_threshold": NOISE_THRESHOLD,

        "edge_threshold": EDGE_THRESHOLD,

        "tgpd_structure_weight": TGPD_STRUCTURE_WEIGHT,

        "hm_dependency_min": HM_DEP_MIN,

        "hm_dependency_max": HM_DEP_MAX,

        "epochs": EPOCHS,

        "contrastive_weight": CONTRASTIVE_WEIGHT,

        "mask_prob": MASK_PROB,

        "max_len": MAX_LEN,

        "batch_size": BATCH_SIZE,

        "feature_batch": FEATURE_BATCH,

        "tgpd_outputs": [
            "activity_score.json",
            "edge_weight.json",
            "attention_heatmap.png",
            "weighted_dfg.json",
            "weighted_dfg.png"
        ]
    }

    with open(
        os.path.join(
            SAVE_DIR,
            "train_config.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            cfg,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n配置文件已保存")

    print("\n===== 完成 =====")

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    process_dataset()