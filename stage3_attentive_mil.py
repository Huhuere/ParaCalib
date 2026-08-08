"""
Stage 3: Attention-based MIL classification over utterance-level SC-Para vectors.

Usage:
  python stage3_attentive_mil.py \
    --root_dir caption_analysis_results_train --labels_csv labels.csv \
    --val_root_dir caption_analysis_results_dev --val_labels_csv labels_dev.csv \
    --test_root_dir caption_analysis_results_test --test_labels_csv labels_test.csv \
    --seeds "50,150,250,350,450" --output_csv results_attn_pooling.csv \
    --run_log run_logs --no-verbose-skip --device cuda
"""

import os, sys, json, glob, re, argparse, random, time
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)

import torch
import torch.nn as nn

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


# =============================
# Config / Constants
# =============================
DEFAULT_TEST_THRESHOLD   = 0.5
PROSODY_CAT_MAX          = 2.0
DEPRESSION_SCORE_MAX     = 100.0
LABELS_CSV_CONV_ID       = "conv_id"
LABELS_CSV_LABEL         = "label"
CLASS_NAMES              = ("negative", "positive")
CLF_REPORT_DIGITS        = 4
RUN_LOG_PREFIX           = "attn_pooling"
MODEL_SAVE_PREFIX        = "attn_pooling"
DEFAULT_TRAIN_VAL_SPLIT  = 0.8
EARLY_STOPPING_PATIENCE  = 10
LR_SCHEDULER_PATIENCE    = 3
LR_SCHEDULER_FACTOR      = 0.5
GRAD_CLIP_NORM           = 1.0
WEIGHT_DECAY             = 1e-4
EPOCHS                   = 50
BATCH_SIZE               = 4
LEARNING_RATE            = 0.0005
ATTN_HIDDEN_DIM          = 64
DROPOUT                  = 0.3
TABLE_WIDTH_SHORT        = 60
TABLE_WIDTH_LONG         = 80
TABLE_WIDTH_METRICS      = 73
DEFAULT_DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


# =============================
# Prosody Feature Mapping
# =============================
CAT_MAP = {
    "pitch_variability": {"flat": 0, "moderate": 1, "high": 2, "unknown": -1},
    "speaking_rate":     {"slow": 0, "medium": 1, "fast": 2, "unknown": -1},
    "rhythm":            {"steady": 0, "irregular": 1, "unknown": -1},
    "pauses":            {"few": 0, "moderate": 1, "many": 2, "unknown": -1},
    "energy":            {"low": 0, "medium": 1, "high": 2, "unknown": -1},
}

FEATURE_KEYS = [
    ("prosody", "pitch_variability"),
    ("prosody", "speaking_rate"),
    ("prosody", "rhythm"),
    ("prosody", "pauses"),
    ("prosody", "energy"),
]


def safe_get(d: Dict, path: Tuple[str, str], default="unknown"):
    a, b = path
    return d.get(a, {}).get(b, default)


def parse_audio_meta(audio_path: Optional[str]) -> Tuple[str, Optional[int]]:
    if not audio_path:
        return "unknown", None
    base = os.path.basename(audio_path)
    m = re.search(r"(\d+)_s?(\d+)", base, re.IGNORECASE)
    if m:
        conv_id = m.group(1)
        try:
            utt_idx = int(m.group(2))
        except (TypeError, ValueError):
            utt_idx = None
        return conv_id, utt_idx
    stem = os.path.splitext(base)[0]
    conv_id = stem.split("_")[0] if "_" in stem else stem
    return conv_id, None


def json_to_feature_vec(j: Dict) -> Optional[np.ndarray]:
    """Skip utterance if prosody unknown or depression score missing."""
    feats = []
    for sec, key in FEATURE_KEYS:
        v = safe_get(j, (sec, key), "unknown")
        mapped = CAT_MAP[key].get(v, -1)
        if v == "unknown" or mapped == -1:
            return None
        feats.append(mapped)

    sc = j.get("depression_acoustic_score", {})
    value = sc.get("value", np.nan)
    conf  = sc.get("confidence", np.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float("nan")
    if np.isnan(value):
        return None  # missing depression score -> exclude
    feats.extend([value, conf])

    feats = np.array(feats, dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0)
    n_prosody = len(FEATURE_KEYS)
    for i in range(n_prosody):
        feats[i] = feats[i] / PROSODY_CAT_MAX
    feats[-2] = feats[-2] / DEPRESSION_SCORE_MAX
    return feats


# =============================
# Dataset
# =============================
class ConversationDataset:
    def __init__(
        self,
        root_dir: str,
        labels_csv: Optional[str] = None,
        verbose_skip: bool = True,
    ):
        self.root_dir      = root_dir
        self._verbose_skip = verbose_skip

        conv_dirs  = sorted([d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])
        flat_jsons = sorted(glob.glob(os.path.join(root_dir, "*.json")))

        self.mode = "folders" if conv_dirs else "flat"
        self.conv_to_files = defaultdict(list)

        if self.mode == "folders":
            self.conv_dirs = conv_dirs
            self.conv_ids  = [os.path.basename(d) for d in self.conv_dirs]
        else:
            if not flat_jsons:
                raise RuntimeError(f"No JSON files found under {root_dir}")
            for fp in flat_jsons:
                with open(fp, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                audio_path = raw.get("audio_file") or raw.get("audio_path")
                if not audio_path and isinstance(raw.get("analysis_result"), dict):
                    audio_path = raw["analysis_result"].get("audio_file")
                conv_id, utt_idx = parse_audio_meta(audio_path)
                self.conv_to_files[conv_id].append((utt_idx, fp))
            self.conv_ids = sorted(self.conv_to_files.keys())

        self.labels = None
        if labels_csv is not None and os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv, dtype={LABELS_CSV_CONV_ID: str})
            self.labels = {r[LABELS_CSV_CONV_ID]: float(r[LABELS_CSV_LABEL])
                           for _, r in df.iterrows()}

    def __len__(self):
        return len(self.conv_ids)

    def __getitem__(self, idx: int) -> HeteroData:
        return self.get(idx)

    def _sorted_files_for_conv(self, conv_id: str) -> List[str]:
        if self.mode == "folders":
            conv_dir = self.conv_dirs[self.conv_ids.index(conv_id)]
            files = sorted(glob.glob(os.path.join(conv_dir, "*.json")))
            if not files:
                raise RuntimeError(f"No json found in {conv_dir}")
            return files
        items = self.conv_to_files[conv_id]
        items = sorted(items, key=lambda t: (t[0] is None, t[0], t[1]))
        return [fp for _, fp in items]

    def get(self, idx: int) -> HeteroData:
        conv_id = self.conv_ids[idx]
        files   = self._sorted_files_for_conv(conv_id)

        utt_feats        = []
        raw_strengths    = []
        skipped          = 0
        skipped_no_score = 0

        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if "analysis" in raw and isinstance(raw["analysis"], dict):
                j = raw["analysis"]
            elif "analysis_result" in raw and isinstance(raw["analysis_result"], dict):
                j = raw["analysis_result"].get("analysis", raw["analysis_result"])
            else:
                j = raw

            sc_val = j.get("depression_acoustic_score", {}).get("value", np.nan)
            try:
                sc_float = float(sc_val)
            except (TypeError, ValueError):
                sc_float = float("nan")

            try:
                _missing = np.isnan(float(sc_val))
            except (TypeError, ValueError):
                _missing = True
            if _missing:
                skipped_no_score += 1

            feat_vec = json_to_feature_vec(j)
            if feat_vec is None:
                skipped += 1
                continue

            utt_feats.append(feat_vec)
            raw_strengths.append(sc_float)

        if len(utt_feats) == 0:
            raise RuntimeError(f"Conversation {conv_id}: no valid utterances")

        if skipped_no_score > 0 and self._verbose_skip:
            print(f"[Info] {conv_id}: {skipped_no_score}/{len(files)} utterances "
                  f"missing depression score, excluded")
        if skipped > 0 and self._verbose_skip:
            print(f"[Info] {conv_id}: {skipped}/{len(files)} utterances skipped")

        utt_features = torch.tensor(np.stack(utt_feats, axis=0), dtype=torch.float)

        data = HeteroData()
        data['utterance'].x         = utt_features
        data['utterance'].num_nodes = utt_features.size(0)
        data['utterance'].scores    = torch.tensor(raw_strengths, dtype=torch.float)

        if self.labels is not None and conv_id in self.labels:
            data.y = torch.tensor([self.labels[conv_id]], dtype=torch.float)
        else:
            data.y = None

        data.conv_id = conv_id
        return data


# =============================
# Model: Attention Pooling
# =============================
class AttentionPoolingModel(nn.Module):
    """
    Additive attention with learnable query — no graph, no message passing.

        α_i = softmax( v^T tanh(W x_i + b) )
        z   = Σ_i α_i · x_i
        pred = MLP(z)

    Parameters:
        W ∈ R^{F' × F}   (Linear layer, includes bias b)
        v ∈ R^{F'}        (learnable query vector)
        F  = utt_in_dim  (input feature dim, 7 in this project)
        F' = attn_dim    (attention projection dim, controlled by --hidden)
    """

    def __init__(self, utt_in_dim: int, attn_dim: int = 32,
                 dropout: float = 0.3, out_dim: int = 1):
        super().__init__()
        self.W = nn.Linear(utt_in_dim, attn_dim, bias=True)
        self.v = nn.Parameter(torch.empty(attn_dim))
        nn.init.xavier_uniform_(self.v.unsqueeze(0))

        self.mlp = nn.Sequential(
            nn.Linear(utt_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_dim),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x     = data["utterance"].x          # [N, F]
        batch = (data["utterance"].batch
                 if hasattr(data["utterance"], "batch")
                 else torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        scores = torch.tanh(self.W(x)) @ self.v   # [N]

        batch_size  = batch.max().item() + 1 if batch.numel() > 0 else 1
        graph_reprs = []
        for b in range(batch_size):
            mask  = batch == b
            x_b   = x[mask]
            alpha = torch.softmax(scores[mask], dim=0)
            z     = (alpha.unsqueeze(1) * x_b).sum(dim=0)
            graph_reprs.append(z)
        graph_reprs = torch.stack(graph_reprs, dim=0)

        was_training = self.training
        if was_training and graph_reprs.size(0) == 1:
            self.eval()
        out = self.mlp(graph_reprs)
        if was_training and graph_reprs.size(0) == 1:
            self.train()
        return out

    def get_attention_weights(self, data: HeteroData) -> np.ndarray:
        """Return per-utterance attention weights α ∈ [0,1]^N for a single conversation."""
        self.eval()
        with torch.no_grad():
            x      = data["utterance"].x.to(next(self.parameters()).device)
            scores = torch.tanh(self.W(x)) @ self.v
            alpha  = torch.softmax(scores, dim=0)
        return alpha.cpu().numpy()

    def get_loss(self, data: HeteroData, criterion: nn.Module):
        out = self(data)
        y   = _get_graph_y(data)
        if y is None:
            raise RuntimeError("No labels found")
        loss = criterion(out.view(-1), y.view(-1))
        return loss, {"main_loss": loss.item()}


# =============================
# Utilities
# =============================
def hetero_collate(data_list):
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(data_list)
    ys = []
    for d in data_list:
        y = _get_graph_y(d)
        if y is None:
            raise ValueError("HeteroData has no label")
        ys.append(y.view(1))
    y_tensor = torch.cat(ys, dim=0)
    batch.y = y_tensor
    if hasattr(batch, "_global_store"):
        batch._global_store["y"] = y_tensor
    return batch


def _get_graph_y(hetero_data):
    gs = getattr(hetero_data, "_global_store", None)
    if gs is not None:
        y = gs.get("y", None)
        if y is not None:
            return y
    try:
        y = getattr(hetero_data, "y", None)
        return y
    except (AttributeError, KeyError):
        return None


def _batch_to_device(batch, device):
    y = _get_graph_y(batch)
    batch = batch.to(device)
    if y is not None:
        y = y.to(device)
        gs = getattr(batch, "_global_store", None)
        if gs is not None:
            gs["y"] = y
        try:
            object.__setattr__(batch, "y", y)
        except Exception:
            pass
    return batch


def _init_run_log(run_log_arg: Optional[str]) -> Optional[str]:
    if not run_log_arg:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.path.isdir(run_log_arg) or not run_log_arg.endswith(".log"):
        base = (run_log_arg or "run_logs").rstrip(os.sep)
        run_log_dir  = f"{base}_{stamp}"
        os.makedirs(run_log_dir, exist_ok=True)
        run_log_path = os.path.join(run_log_dir, f"{RUN_LOG_PREFIX}_{stamp}.log")
    else:
        run_log_path = run_log_arg
        os.makedirs(os.path.dirname(os.path.abspath(run_log_path)) or ".", exist_ok=True)
    with open(run_log_path, "w", encoding="utf-8") as f:
        f.write(f"[RunLog] Model: attention_pooling - {datetime.now().isoformat()}\n")
        f.write(f"[RunLog] Command: {' '.join(sys.argv)}\n")
    return run_log_path


def _write_run_log(path: Optional[str], line: str):
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line if line.endswith("\n") else line + "\n")


def _set_strict_reproducibility(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def _make_loader_generator(seed: int, offset: int = 0) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + offset)
    return gen


# =============================
# Evaluation
# =============================
def eval_binary_metrics(model, loader, device, threshold=0.5,
                        run_log_path: Optional[str] = None):
    model.eval()
    ys, ps, probs_list = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch  = _batch_to_device(batch, device)
            logits = model(batch).view(-1)
            prob   = torch.sigmoid(logits)
            pred   = (prob >= threshold).long().cpu().numpy()
            y      = _get_graph_y(batch)
            if y is None:
                raise RuntimeError("Batch has no labels")
            ys.append(y.view(-1).cpu().numpy())
            ps.append(pred)
            probs_list.append(prob.cpu().numpy())

    if not ys:
        return {"accuracy": 0.0, "f1": 0.0, "auc_roc": 0.0}

    y_true  = np.concatenate(ys)
    y_pred  = np.concatenate(ps)
    y_probs = np.concatenate(probs_list)

    def _log(s):
        print(s)
        _write_run_log(run_log_path, s)

    _log(f"[Debug] Positive samples: {y_true.sum()}/{len(y_true)} ({y_true.mean()*100:.1f}%)")
    _log(f"[Debug] Positive predictions: {y_pred.sum()}/{len(y_pred)} ({y_pred.mean()*100:.1f}%)")
    _log(f"[Debug] Prob range: [{y_probs.min():.4f}, {y_probs.max():.4f}]")
    _log(f"[Debug] Prob mean: {y_probs.mean():.4f}, median: {np.median(y_probs):.4f}")

    prec_cls = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_cls  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_cls   = f1_score(y_true, y_pred, average=None, zero_division=0)
    clf_rep  = classification_report(y_true, y_pred,
                                     target_names=list(CLASS_NAMES),
                                     digits=CLF_REPORT_DIGITS, zero_division=0)

    return {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "precision":          float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":             float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":                 float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":            float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0,
        "precision_negative": float(prec_cls[0]) if len(prec_cls) > 0 else 0.0,
        "recall_negative":    float(rec_cls[0])  if len(rec_cls)  > 0 else 0.0,
        "f1_negative":        float(f1_cls[0])   if len(f1_cls)   > 0 else 0.0,
        "precision_positive": float(prec_cls[1]) if len(prec_cls) > 1 else 0.0,
        "recall_positive":    float(rec_cls[1])  if len(rec_cls)  > 1 else 0.0,
        "f1_positive":        float(f1_cls[1])   if len(f1_cls)   > 1 else 0.0,
        "precision_macro":    float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":       float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":           float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": clf_rep,
    }


# =============================
# Training
# =============================
def train_model(
    root_dir: str,
    labels_csv: str,
    val_root_dir: Optional[str]    = None,
    val_labels_csv: Optional[str]  = None,
    test_root_dir: Optional[str]   = None,
    test_labels_csv: Optional[str] = None,
    random_seed: Optional[int]     = None,
    run_log_path: Optional[str]    = None,
    verbose_skip: bool             = True,
    device_str: str                = DEFAULT_DEVICE,
):
    if random_seed is None:
        raise ValueError("Reproducibility requires an explicit random seed.")

    _set_strict_reproducibility(random_seed)
    print(f"[Info] Model: attention_pooling | Seed: {random_seed}")
    if device_str == "cuda":
        print("[Warning] GPU mode: best-effort determinism only.")

    _write_run_log(run_log_path, f"[RunLog] Seed: {random_seed}")

    # ── Datasets ────────────────────────────────────────────────────────
    train_ds = ConversationDataset(root_dir=root_dir, labels_csv=labels_csv,
                                   verbose_skip=verbose_skip)
    print(f"[Info] Training set: {len(train_ds)} conversations")

    if val_root_dir and val_labels_csv:
        val_ds = ConversationDataset(root_dir=val_root_dir, labels_csv=val_labels_csv,
                                     verbose_skip=verbose_skip)
        print(f"[Info] Validation set: {len(val_ds)} conversations")
    else:
        full = train_ds
        # ── Stratified split: preserve positive/negative ratio ────
        all_data   = [full.get(i) for i in range(len(full))]
        labels_all = np.array([
            int(_get_graph_y(d).item()) if _get_graph_y(d) is not None else 0
            for d in all_data
        ])
        pos_idx = np.where(labels_all == 1)[0]
        neg_idx = np.where(labels_all == 0)[0]
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        pos_split = int(DEFAULT_TRAIN_VAL_SPLIT * len(pos_idx))
        neg_split = int(DEFAULT_TRAIN_VAL_SPLIT * len(neg_idx))
        train_idx = np.concatenate([pos_idx[:pos_split], neg_idx[:neg_split]])
        val_idx   = np.concatenate([pos_idx[pos_split:], neg_idx[neg_split:]])
        train_ds  = [all_data[i] for i in train_idx]
        val_ds    = [all_data[i] for i in val_idx]
        print(f"[Info] Stratified split: "
              f"{len(train_ds)} train ({int(labels_all[train_idx].sum())}/{len(train_ds)} pos), "
              f"{len(val_ds)} val ({int(labels_all[val_idx].sum())}/{len(val_ds)} pos)")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=hetero_collate, num_workers=0,
                              generator=_make_loader_generator(random_seed, offset=0))
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=hetero_collate, num_workers=0,
                              generator=_make_loader_generator(random_seed, offset=1))

    # ── Model ────────────────────────────────────────────────────────────
    first_data = train_ds[0] if isinstance(train_ds, list) else train_ds.get(0)
    utt_in_dim = first_data["utterance"].x.size(1)

    model  = AttentionPoolingModel(utt_in_dim=utt_in_dim, attn_dim=ATTN_HIDDEN_DIM, dropout=DROPOUT)
    device = torch.device(device_str)
    print(f"[Info] Device: {device}")
    model  = model.to(device)

    # ── Class weights ────────────────────────────────────────────────────
    all_labels = []
    data_list  = train_ds if isinstance(train_ds, list) else [train_ds.get(i) for i in range(len(train_ds))]
    for data in data_list:
        y = _get_graph_y(data)
        if y is not None:
            all_labels.append(y.item())

    pos_weight = None
    if all_labels:
        pos_count = sum(all_labels)
        neg_count = len(all_labels) - pos_count
        if pos_count > 0:
            pos_weight = torch.tensor([neg_count / pos_count], device=device)
            print(f"[Info] Class dist: {int(pos_count)}/{len(all_labels)} positive, "
                  f"pos_weight={pos_weight.item():.4f}")

    # ── Optimizer / loss ─────────────────────────────────────────────────
    opt       = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None \
                else nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)

    # ── Training loop ────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience         = 0
    best_model_state = None

    _train_start_time = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            total_loss, _ = model.get_loss(batch, criterion)
            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            losses.append(total_loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _batch_to_device(batch, device)
                pred  = model(batch).view(-1)
                y     = _get_graph_y(batch).view(-1)
                val_losses.append(criterion(pred, y).item())

        val_loss = np.mean(val_losses) if val_losses else 0.0
        line = f"Epoch {ep:02d} | train={np.mean(losses):.4f} | val={val_loss:.4f}"
        print(line)
        _write_run_log(run_log_path, f"[RunLog] {line}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience         = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best (val_loss={val_loss:.4f})")
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"[Info] Early stopping at epoch {ep}")
                break

    _train_elapsed_sec = time.time() - _train_start_time
    _train_gpu_hours   = _train_elapsed_sec / 3600.0
    print(f"[Timing] Training wall-clock: {_train_elapsed_sec:.1f}s  ({_train_gpu_hours:.4f} GPU-h)")

    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"\n[Info] Restored best model (val_loss={best_val_loss:.4f})")

    # ── Evaluate train / val for overfitting analysis ────────────────────
    _train_eval_data = train_ds if isinstance(train_ds, list) else [train_ds.get(i) for i in range(len(train_ds))]
    _train_eval_loader = DataLoader(
        _train_eval_data, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=hetero_collate, num_workers=0,
        generator=_make_loader_generator(random_seed, offset=10),
    )
    print("\n[Info] Evaluating train set for overfitting analysis ...")
    _train_metrics = eval_binary_metrics(
        model, _train_eval_loader, device,
        threshold=DEFAULT_TEST_THRESHOLD, run_log_path=run_log_path,
    )

    print("[Info] Evaluating val set for overfitting analysis ...")
    _val_metrics = eval_binary_metrics(
        model, val_loader, device,
        threshold=DEFAULT_TEST_THRESHOLD, run_log_path=run_log_path,
    )

    # ── Test ─────────────────────────────────────────────────────────────
    if test_root_dir and test_labels_csv:
        test_ds = ConversationDataset(root_dir=test_root_dir, labels_csv=test_labels_csv,
                                      verbose_skip=verbose_skip)
        print(f"\n[Info] Test set: {len(test_ds)} conversations")
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=hetero_collate, num_workers=0,
                                 generator=_make_loader_generator(random_seed, offset=2))

        sep60 = "=" * TABLE_WIDTH_SHORT
        print(f"\n{sep60}\nTest (threshold={DEFAULT_TEST_THRESHOLD})\n{sep60}")
        metrics = eval_binary_metrics(model, test_loader, device,
                                      threshold=DEFAULT_TEST_THRESHOLD, run_log_path=run_log_path)

        for line in [
            f"  Overall    - Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f} | AUC={metrics['auc_roc']:.4f}",
            f"  Negative   - Prec={metrics['precision_negative']:.4f} | Rec={metrics['recall_negative']:.4f} | F1={metrics['f1_negative']:.4f}",
            f"  Positive   - Prec={metrics['precision_positive']:.4f} | Rec={metrics['recall_positive']:.4f} | F1={metrics['f1_positive']:.4f}",
            f"  Macro Avg  - Prec={metrics['precision_macro']:.4f} | Rec={metrics['recall_macro']:.4f} | F1={metrics['f1_macro']:.4f}",
        ]:
            print(line)
            _write_run_log(run_log_path, line)

        print(f"\n[Model End] AUC-ROC = {metrics['auc_roc']:.4f}")
        for line in metrics.get("classification_report", "").strip().split("\n"):
            print(f"  {line}")
            _write_run_log(run_log_path, f"  {line}")

        # ── Overfitting analysis: Train / Val / Test ─────────────────────
        ov_cols   = ["accuracy", "f1_macro", "auc_roc", "recall_positive", "recall_negative"]
        ov_labels = ["Accuracy", "F1_Macro", "AUC-ROC", "Sensitivity", "Specificity"]
        col_w_ov  = 12
        sep_ov    = "-" * (10 + col_w_ov * len(ov_cols))
        hdr_ov    = f"{'Split':<10}" + "".join(f"{l:>{col_w_ov}}" for l in ov_labels)
        ov_title  = f"\nOverfitting Analysis (seed={random_seed}, threshold={DEFAULT_TEST_THRESHOLD})"
        for _ln in [ov_title, hdr_ov, sep_ov]:
            print(_ln); _write_run_log(run_log_path, _ln)
        for split_name, split_m in [("Train", _train_metrics),
                                     ("Val",   _val_metrics),
                                     ("Test",  metrics)]:
            vals = "".join(f"{split_m.get(c, float('nan')):>{col_w_ov}.4f}"
                           for c in ov_cols)
            _ln  = f"{split_name:<10}" + vals
            print(_ln); _write_run_log(run_log_path, _ln)
        print(sep_ov); _write_run_log(run_log_path, sep_ov)

        # ── Attach train/val metrics to returned dict ────────────────────
        for _k, _v in _train_metrics.items():
            if _k != "classification_report":
                metrics[f"train_{_k}"] = _v
        for _k, _v in _val_metrics.items():
            if _k != "classification_report":
                metrics[f"val_{_k}"] = _v

        save_dir  = os.path.dirname(run_log_path) if run_log_path else "."
        save_path = os.path.join(save_dir,
                                 f"{MODEL_SAVE_PREFIX}_seed{random_seed}"
                                 f"_f1{metrics['f1']:.4f}.pt")
        torch.save({"model_state_dict": model.state_dict(),
                    "metrics": metrics}, save_path)
        print(f"[Info] Model saved to {save_path}")
        _write_run_log(run_log_path, f"[RunLog] End {datetime.now().isoformat()}")

        metrics["threshold"] = DEFAULT_TEST_THRESHOLD
        metrics["seed"]      = random_seed
        metrics["gpu_hours"] = round(_train_gpu_hours, 6)
        metrics["train_sec"] = round(_train_elapsed_sec, 2)
        return model, metrics

    return model, None


# =============================
# CLI
# =============================
def _cli():
    parser = argparse.ArgumentParser(
        description="Attention Pooling model for depression detection")
    parser.add_argument("--root_dir",        default="caption_analysis_results_train")
    parser.add_argument("--labels_csv",      default="labels.csv")
    parser.add_argument("--val_root_dir",    default=None)
    parser.add_argument("--val_labels_csv",  default=None)
    parser.add_argument("--test_root_dir",   default=None)
    parser.add_argument("--test_labels_csv", default=None)
    parser.add_argument("--seeds",           type=str, default="42")
    parser.add_argument("--output_csv",      type=str, default="results_attn_pooling.csv")
    parser.add_argument("--run_log",         type=str, default=None)
    parser.add_argument("--no-verbose-skip", dest="verbose_skip", action="store_false")
    parser.add_argument("--device",          choices=["cpu", "cuda"], default=DEFAULT_DEVICE)
    args = parser.parse_args()

    run_log_path = _init_run_log(args.run_log)
    if run_log_path:
        print(f"[Info] Run log: {run_log_path}")

    seed_list = [int(s.strip()) for s in args.seeds.split(",")]

    sep80 = "=" * TABLE_WIDTH_LONG
    if len(seed_list) > 1:
        print(f"\n{sep80}")
        print(f"Model: attention_pooling | Seeds: {seed_list}")
        print(f"{sep80}\n")

    all_results = []
    for seed in seed_list:
        print(f"\n{'#'*80}\n# SEED {seed}\n{'#'*80}\n")
        _, metrics = train_model(
            root_dir=args.root_dir,
            labels_csv=args.labels_csv,
            val_root_dir=args.val_root_dir,
            val_labels_csv=args.val_labels_csv,
            test_root_dir=args.test_root_dir,
            test_labels_csv=args.test_labels_csv,
            random_seed=seed,
            run_log_path=run_log_path,
            verbose_skip=args.verbose_skip,
            device_str=args.device,
        )
        if metrics is not None:
            all_results.append(metrics)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output_csv, index=False)
        print(f"\n[Info] Results saved to {args.output_csv}")

        print(f"\n{sep80}")
        print(f"SUMMARY — attention_pooling ({len(all_results)} seeds)")
        print(f"{sep80}")

        # ── Per-seed results table ────────────────────────────────────────
        key_cols   = ["auc_roc", "recall_positive", "recall_negative",
                      "f1_macro", "f1_negative", "f1_positive"]
        col_labels = ["auc_roc", "sensitivity", "specificity",
                      "f1_macro", "f1_negative", "f1_positive"]
        col_w = 12
        sep_wide = "-" * (8 + col_w * len(key_cols) + len(key_cols))
        header = f"{'Seed':<8}" + "".join(f"{lbl:>{col_w}}" for lbl in col_labels)
        print(f"\nPer-seed results:")
        print(sep_wide)
        print(header)
        print(sep_wide)
        for _, row in df.iterrows():
            seed_val = int(row["seed"]) if "seed" in row else "-"
            line = f"{seed_val:<8}" + "".join(
                f"{row[c]:>{col_w}.4f}" if c in row else f"{'N/A':>{col_w}}"
                for c in key_cols
            )
            print(line)
        print(sep_wide)

        # ── Mean ± Std row ────────────────────────────────────────────────
        mean_line = f"{'Mean':<8}" + "".join(
            f"{df[c].mean():>{col_w}.4f}" if c in df.columns else f"{'N/A':>{col_w}}"
            for c in key_cols
        )
        std_line  = f"{'Std':<8}" + "".join(
            f"{df[c].std():>{col_w}.4f}" if c in df.columns else f"{'N/A':>{col_w}}"
            for c in key_cols
        )
        print(mean_line)
        print(std_line)
        print(sep_wide)

        # ── Compact summary ───────────────────────────────────────────────
        sep73 = "-" * TABLE_WIDTH_METRICS
        print(f"\n{'Metric':<25} {'Mean':>10} {'Std':>10}")
        print(sep73)
        for col, lbl in zip(key_cols, col_labels):
            if col in df.columns:
                print(f"{lbl:<25} {df[col].mean():>10.4f} {df[col].std():>10.4f}")
        print(sep73)
        print(f"\n  Macro F1    : {df['f1_macro'].mean():.4f} ± {df['f1_macro'].std():.4f}")
        print(f"  AUC-ROC     : {df['auc_roc'].mean():.4f} ± {df['auc_roc'].std():.4f}")
        print(f"  Sensitivity : {df['recall_positive'].mean():.4f} ± {df['recall_positive'].std():.4f}")
        print(f"  Specificity : {df['recall_negative'].mean():.4f} ± {df['recall_negative'].std():.4f}\n")

        # ── Overfitting Analysis (averaged across seeds) ──────────────────
        ov_splits_cli = [
            ("Train", ["train_accuracy", "train_f1_macro", "train_auc_roc",
                       "train_recall_positive", "train_recall_negative"]),
            ("Val",   ["val_accuracy",   "val_f1_macro",   "val_auc_roc",
                       "val_recall_positive",   "val_recall_negative"]),
            ("Test",  ["accuracy",       "f1_macro",       "auc_roc",
                       "recall_positive",       "recall_negative"]),
        ]
        ov_col_labels_cli = ["Accuracy", "F1_Macro", "AUC-ROC", "Sensitivity", "Specificity"]
        col_w_cli = 12
        sep_cli   = "-" * (10 + col_w_cli * len(ov_col_labels_cli))
        hdr_cli   = f"{'Split':<10}" + "".join(f"{l:>{col_w_cli}}" for l in ov_col_labels_cli)
        print(f"\n{sep80}")
        print(f"Overfitting Analysis — Mean ± Std across {len(all_results)} seeds")
        print(f"{sep80}")
        print(hdr_cli)
        print(sep_cli)
        for split_name, col_keys in ov_splits_cli:
            present = [k for k in col_keys if k in df.columns]
            if not present:
                continue
            vals = "".join(
                f"{df[k].mean():>{col_w_cli}.4f}" if k in df.columns else f"{'N/A':>{col_w_cli}}"
                for k in col_keys
            )
            stds = "".join(
                f"±{df[k].std():>{col_w_cli-1}.4f}" if k in df.columns else f"{'':>{col_w_cli}}"
                for k in col_keys
            )
            print(f"{split_name:<10}" + vals)
            print(f"{'':10}" + stds)
        print(sep_cli)


if __name__ == "__main__":
    _cli()
