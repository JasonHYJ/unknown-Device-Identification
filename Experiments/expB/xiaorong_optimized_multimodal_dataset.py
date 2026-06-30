# xiaorong_optimized_multimodal_dataset.py
# 📦 多模态 IoT 数据集加载（支持 F0–F12；自监督128维自动对齐到64维；恒定287维输入）
#
# 输入组织（来自 CSV 的绝对路径）：
#   - stat_feature_path :  对比阶段的归一化统计特征 CSV（31维，第31维为 is_behavior）
#   - seq_embed_feature_path :  对比学习(64) 的序列嵌入 .npy
#   - raw_embed_feature_path :  对比学习(64) 的原始字节嵌入 .npy
# 当选择 *-Embed（自监督）时，内部按规则把 “10_contrastive_*” 替换成 “9_learned_*”，
# 并将 128 维嵌入降到 64 维（相邻两维求均值）。
#
# 输出向量固定 287 维：
#   [31 stat, 128 idle_embed(64 seq + 64 raw), 128 beh_embed(64 seq + 64 raw)]
# 若未启用 Stat，则 stat 前30维=0，仅第31维写 is_behavior。
# 嵌入按 is_behavior 放到 idle/behavior 一侧，另一侧全0（门控才有作用）。
#
# 4) 标签读取自 label dict（type2idx/brand2idx/device2idx），查不到则返回全零 one-hot（并统计告警）。
#
# feat_combo（与训练脚本 F0–F12 对应）：
#   "Stat", "Seq-Embed", "Raw-Embed", "Seq-Embed+Raw-Embed",
#   "Stat+Seq-Embed", "Stat+Raw-Embed", "Stat+Seq-Embed+Raw-Embed",
#   "Seq-CL", "Raw-CL", "Seq-CL+Raw-CL",
#   "Stat+Seq-CL", "Stat+Raw-CL", "Stat+Seq-CL+Raw-CL"
#
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import Counter

# —— 这三个根前缀仅用于“按规则替换为自监督路径”时使用 —— #
_CONTRASTIVE_SEQ_KEY = "/10_contrastive_embeddings/10_contrastive_sequence_embeddings/"
_CONTRASTIVE_RAW_KEY = "/10_contrastive_embeddings/10_contrastive_rawbyte_embeddings/"
_LEARNED_SEQ_KEY     = "/9_learned_embeddings/9_learned_sequence_embeddings/"
_LEARNED_RAW_KEY     = "/9_learned_embeddings/9_learned_rawbyte_embeddings/"

class MultiModalIoTDataset(Dataset):
    """
    支持消融组合的多模态 IoT 数据集。

    feat_combo 取值（与训练脚本 F0–F12 对应）：
      - "Stat"                                # F0
      - "Seq-Embed", "Raw-Embed", "Seq-Embed+Raw-Embed"                 # F1,F2,F3（自监督）
      - "Stat+Seq-Embed", "Stat+Raw-Embed", "Stat+Seq-Embed+Raw-Embed"  # F4,F5,F6
      - "Seq-CL", "Raw-CL", "Seq-CL+Raw-CL"                             # F7,F8,F9（对比）
      - "Stat+Seq-CL", "Stat+Raw-CL", "Stat+Seq-CL+Raw-CL"              # F10,F11,F12（默认跑）
    """
    def __init__(self, csv_path, label_dict_dir, feat_combo="Stat+Seq-CL+Raw-CL"):
        super().__init__()
        self.df = pd.read_csv(csv_path)

        # 必要列校验
        need = [
            "type_label","brand_label","device_label","is_behavior",
            "stat_feature_path","seq_embed_feature_path","raw_embed_feature_path"
        ]
        miss = [c for c in need if c not in self.df.columns]
        if miss:
            raise ValueError(f"CSV 缺少必要列: {miss}")

        self.feat_combo = feat_combo

        # 标签字典
        label_dir = Path(label_dict_dir)
        self.type2idx   = json.loads(Path(label_dir/"type2idx.json").read_text())
        self.brand2idx  = json.loads(Path(label_dir/"brand2idx.json").read_text())
        self.device2idx = json.loads(Path(label_dir/"device2idx.json").read_text())
        self.num_type, self.num_brand, self.num_device = len(self.type2idx), len(self.brand2idx), len(self.device2idx)

        # 维度与常量
        self.stat_dim = 31
        self.embed64  = 64   # 目标对齐维度
        self.side_dim = 2*self.embed64  # 128
        self.total_dim= self.stat_dim + self.side_dim + self.side_dim  # 31 + 128 + 128 = 287

        # 组合标志：Stat / 自监督(Embed) / 对比(CL)
        self.use_stat    = ("Stat" in feat_combo)
        self.use_seq_cl  = ("Seq-CL" in feat_combo)
        self.use_raw_cl  = ("Raw-CL" in feat_combo)
        self.use_seq_emb = ("Seq-Embed" in feat_combo)
        self.use_raw_emb = ("Raw-Embed" in feat_combo)

        # 统计无效标签（每 N 条打印一次）
        self._invalid = Counter()
        self._warn_every = 200

        print(f"[Dataset] feat_combo={self.feat_combo} | use_stat={self.use_stat} | "
              f"seq: CL={self.use_seq_cl}, EMB={self.use_seq_emb} | "
              f"raw: CL={self.use_raw_cl}, EMB={self.use_raw_emb}")
        print(f"[Dataset] 样本数={len(self.df)} | label sizes: type={self.num_type}, brand={self.num_brand}, device={self.num_device}")

    def __len__(self): 
        return len(self.df)

    # --- 工具：路径替换为自监督 --- #
    def _swap_to_learned_seq(self, p:str)->str:
        return p.replace(_CONTRASTIVE_SEQ_KEY, _LEARNED_SEQ_KEY) if _CONTRASTIVE_SEQ_KEY in p else p
    def _swap_to_learned_raw(self, p:str)->str:
        return p.replace(_CONTRASTIVE_RAW_KEY, _LEARNED_RAW_KEY) if _CONTRASTIVE_RAW_KEY in p else p

    # --- 工具：把 128 维降到 64 维；其它异常维度兜底 --- #
    def _to_64dim(self, vec: np.ndarray) -> np.ndarray:
        v = vec.reshape(-1).astype(np.float32)
        d = v.shape[0]
        if d == 64:
            return v
        if d == 128:
            try:
                return v.reshape(64, 2).mean(axis=1).astype(np.float32)
            except Exception:
                return v[:64].astype(np.float32)
        if d > 64:
            return v[:64].astype(np.float32)
        # d < 64
        z = np.zeros(64, dtype=np.float32); z[:d] = v
        return z

    def _load_stat(self, path: str, isb: int) -> np.ndarray:
        """
        读取 31 维 stat；若未启用 Stat，则返回全 0，但把 is_behavior 写入第 31 维（索引 30）。
        """
        if not self.use_stat:
            s = np.zeros(self.stat_dim, dtype=np.float32)
            s[-1] = float(isb)  # 仍把 is_behavior 写进最后一维，供门控使用
            return s
        try:
            # 归一化后的 stat CSV：首行 header，第二行是数据
            v = pd.read_csv(path, skiprows=1, header=None).iloc[0, :self.stat_dim].values.astype(np.float32)
            if int(v[-1]) != int(isb):
                print(f"[WARN] is_behavior mismatch: csv={isb}, stat[-1]={v[-1]} | {path}")
            return v
        except Exception as e:
            print(f"[WARN] 读取 stat 失败：{path} -> {e}")
            z = np.zeros(self.stat_dim, dtype=np.float32); z[-1] = float(isb)
            return z

    def _load_embed(self, path: str) -> np.ndarray:
        try:
            arr = np.load(path)
            return self._to_64dim(arr)
        except Exception as e:
            print(f"[WARN] 读取 embed 失败：{path} -> {e}")
            return np.zeros(64, dtype=np.float32)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isb = int(row["is_behavior"])

        # —— 解析三类路径（CSV 给的是“对比学习”路径） —— #
        stat_path = str(row["stat_feature_path"])
        seq_path  = str(row["seq_embed_feature_path"])
        raw_path  = str(row["raw_embed_feature_path"])

        # 按组合选择“对比/自监督”路径
        if self.use_seq_emb and not self.use_seq_cl:
            seq_path = self._swap_to_learned_seq(seq_path)
        if self.use_raw_emb and not self.use_raw_cl:
            raw_path = self._swap_to_learned_raw(raw_path)

        stat = self._load_stat(stat_path, isb)
        seq  = self._load_embed(seq_path) if (self.use_seq_cl or self.use_seq_emb) else np.zeros(64, np.float32)
        raw  = self._load_embed(raw_path) if (self.use_raw_cl or self.use_raw_emb) else np.zeros(64, np.float32)

        # 组装 idle/behavior 两侧（按 is_behavior 放一侧，另一侧置零）
        if isb == 1:
            idle = np.zeros(128, dtype=np.float32)
            beh  = np.concatenate([seq, raw], axis=0).astype(np.float32)
        else:
            idle = np.concatenate([seq, raw], axis=0).astype(np.float32)
            beh  = np.zeros(128, dtype=np.float32)

        # —— 拼成 287 维输入 —— #
        x = np.concatenate([stat, idle, beh], axis=0).astype(np.float32)

        # one-hot（查不到就全零）
        def _ix(d, k):
            try: return d[k]
            except KeyError:
                self._invalid[f"missing:{k}"] += 1
                return -1

        ti = _ix(self.type2idx,   row["type_label"])
        bi = _ix(self.brand2idx,  row["brand_label"])
        di = _ix(self.device2idx, row["device_label"])

        def _oh(n, i):
            if i < 0: return np.zeros(n, dtype=np.float32)
            z = np.zeros(n, dtype=np.float32); z[i] = 1.0; return z

        y_type   = _oh(self.num_type,  ti)
        y_brand  = _oh(self.num_brand, bi)
        y_device = _oh(self.num_device,di)

        # 偶尔打印无效标签统计
        if (idx+1) % self._warn_every == 0 or idx == len(self.df)-1:
            for k,v in list(self._invalid.items()):
                if v>0: print(f"[WARN] invalid label {k}: {v} times (up to {idx})")
            self._invalid.clear()

        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y_type), torch.tensor(y_brand), torch.tensor(y_device)
