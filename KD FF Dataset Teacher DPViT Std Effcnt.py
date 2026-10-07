"""
Knowledge Distillation: DPViT (Teacher) -> EfficientNet-B0 (Student)
FaceForensics++ Deepfake Detection

KD Loss = alpha * L_soft + beta * L_feat + gamma * L_hard
  L_soft : Soft-label KD (Hinton 2015)
  L_feat : Feature KD — student projection matches teacher 1536-d fused space
  L_hard : Hard-label BCE with label smoothing
"""

import os
import cv2
import time
import math
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm.notebook import tqdm
import timm

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
    average_precision_score, precision_recall_curve,
    classification_report,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_FRAMES    = "/kaggle/input/datasets/syedazmulhasansabbir/face-forensics-frames/Face Forensics Frames"
REAL_FOLDER_NAME  = 'orginal'
FAKE_FOLDER_NAMES = ['Face2Face', 'Deepfakes', 'NeuralTextures', 'FaceShifter', 'FaceSwap']

TEACHER_PATH = "/kaggle/input/models/syedazmulhasansabbir/dp-vit-ff/tensorflow2/default/1/best_dpvit after 10 epoch ff.pth"

BATCH_SIZE      = 32
EPOCHS          = 15
IMG_SIZE        = 224
SEED            = 42

LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
WARMUP_EPOCHS   = 2
PATIENCE        = 4

KD_TEMPERATURE  = 4.0
ALPHA           = 0.50
BETA            = 0.30
GAMMA           = 0.20
LABEL_SMOOTHING = 0.05

TEACHER_FEAT_DIM = 1536

OUT_DIR      = Path("/kaggle/working/kd_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RUN_TIME = 11.25 * 3600
START_TIME   = time.time()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
print(f"KD config -> T={KD_TEMPERATURE}  alpha={ALPHA}  beta={BETA}  gamma={GAMMA}")


# =============================================================================
# DATASET
# =============================================================================

class FaceForensicsDataset(Dataset):
    def __init__(self, root_dir: str):
        valid_exts      = {'.jpg', '.jpeg', '.png'}
        self.real_paths = []
        self.fake_paths = []
        self.path_to_video: dict[str, str] = {}

        print("Scanning directories...")
        real_dir = os.path.join(root_dir, REAL_FOLDER_NAME)
        if os.path.exists(real_dir):
            for root, _, files in os.walk(real_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        fp = os.path.join(root, f)
                        self.real_paths.append(fp)
                        self.path_to_video[fp] = os.path.basename(root)

        for fake_name in FAKE_FOLDER_NAMES:
            fake_dir = os.path.join(root_dir, fake_name)
            if os.path.exists(fake_dir):
                for root, _, files in os.walk(fake_dir):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in valid_exts:
                            fp = os.path.join(root, f)
                            self.fake_paths.append(fp)
                            self.path_to_video[fp] = f"{fake_name}_{os.path.basename(root)}"

        self.all_paths = self.real_paths + self.fake_paths
        self.labels    = [0] * len(self.real_paths) + [1] * len(self.fake_paths)

        print(f"  REAL : {len(self.real_paths):,}")
        print(f"  FAKE : {len(self.fake_paths):,}")
        print(f"  TOTAL: {len(self.all_paths):,}")
        if not self.all_paths:
            raise ValueError("No images found — check DATASET_FRAMES path.")

    def __len__(self):
        return len(self.all_paths)

    @staticmethod
    def _high_freq(img: Image.Image) -> Image.Image:
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        lap = cv2.convertScaleAbs(cv2.Laplacian(arr, cv2.CV_64F))
        return Image.fromarray(cv2.cvtColor(lap, cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        img_rgb = Image.open(self.all_paths[idx]).convert('RGB')
        return img_rgb, self._high_freq(img_rgb), self.labels[idx]


class KDTransformWrapper(Dataset):
    def __init__(self, subset, train_tf, eval_tf, is_train: bool, full_dataset):
        self.subset       = subset
        self.tf           = train_tf if is_train else eval_tf
        self.full_dataset = full_dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_rgb, img_hf, label = self.subset[idx]
        rgb_t = self.tf(img_rgb)
        hf_t  = self.tf(img_hf)
        rgb_s = self.tf(img_rgb)
        return rgb_t, hf_t, rgb_s, torch.tensor(label, dtype=torch.float32)


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_dataset = FaceForensicsDataset(root_dir=DATASET_FRAMES)
total   = len(full_dataset)
n_train = int(0.80 * total)
n_val   = int(0.10 * total)
n_test  = total - n_train - n_val

gen = torch.Generator().manual_seed(SEED)
train_sub, val_sub, test_sub = random_split(
    full_dataset, [n_train, n_val, n_test], generator=gen
)
print(f"\nSplit -> train: {n_train:,}  |  val: {n_val:,}  |  test: {n_test:,}")

train_data = KDTransformWrapper(train_sub, train_tf, eval_tf, is_train=True,  full_dataset=full_dataset)
val_data   = KDTransformWrapper(val_sub,   train_tf, eval_tf, is_train=False, full_dataset=full_dataset)
test_data  = KDTransformWrapper(test_sub,  train_tf, eval_tf, is_train=False, full_dataset=full_dataset)

train_labels   = [full_dataset.labels[i] for i in train_sub.indices]
class_counts   = [train_labels.count(0), train_labels.count(1)]
class_weights  = [1.0 / c for c in class_counts]
sample_weights = [class_weights[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)


# =============================================================================
# TEACHER MODEL (frozen DPViT)
# =============================================================================

class DPViT(nn.Module):
    def __init__(self, model_name: str = 'vit_base_patch16_224', pretrained: bool = False):
        super().__init__()
        self.rgb_branch = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.hf_branch  = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_feat = self.rgb_branch.num_features + self.hf_branch.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_feat, 512),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 1),
        )

    def forward(self, rgb_x, hf_x):
        feats = torch.cat([self.rgb_branch(rgb_x), self.hf_branch(hf_x)], dim=1)
        return self.classifier(feats)

    def get_fused_features(self, rgb_x, hf_x):
        return torch.cat([self.rgb_branch(rgb_x), self.hf_branch(hf_x)], dim=1)

    def get_logits(self, rgb_x, hf_x):
        return self.forward(rgb_x, hf_x)


print(f"\nLoading teacher weights from: {TEACHER_PATH}")
if not os.path.exists(TEACHER_PATH):
    raise FileNotFoundError(f"Teacher checkpoint not found: {TEACHER_PATH}")

teacher = DPViT(pretrained=False).to(DEVICE)
teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print("Teacher loaded and frozen.")

with torch.no_grad():
    _d = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    _t = teacher.get_fused_features(_d, _d)
    assert _t.shape == (2, TEACHER_FEAT_DIM), \
        f"Unexpected teacher feature dim: {_t.shape[1]} (expected {TEACHER_FEAT_DIM})"
print(f"Teacher feature dim confirmed: {TEACHER_FEAT_DIM}")


# =============================================================================
# STUDENT MODEL (EfficientNet-B0 + projection head)
# =============================================================================

class EfficientNetStudent(nn.Module):
    """
    EfficientNet-B0 student.
    backbone   : EfficientNet-B0 (feature extractor, output dim=1280)
    classifier : 1280 -> 256 -> 1
    proj_head  : 1280 -> 1536 (aligns with teacher fused feature space)
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        student_feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(student_feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        self.proj_head = nn.Sequential(
            nn.LayerNorm(student_feat_dim),
            nn.Linear(student_feat_dim, TEACHER_FEAT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)

    def get_features_and_logits(self, x: torch.Tensor):
        feats = self.backbone(x)
        logit = self.classifier(feats)
        proj  = self.proj_head(feats)
        return proj, logit


student = EfficientNetStudent().to(DEVICE)

teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
student_params = sum(p.numel() for p in student.parameters()) / 1e6
print(f"\nTeacher params : {teacher_params:.1f} M  (frozen)")
print(f"Student params : {student_params:.1f} M  (trainable)")
print(f"Compression    : {teacher_params / student_params:.1f}x")


# =============================================================================
# KD LOSS
# =============================================================================

class KDLoss(nn.Module):
    """
    L_soft : KL-divergence between teacher and student temperature-scaled sigmoid outputs
    L_feat : Cosine similarity loss between student projection and teacher fused features
    L_hard : Label-smoothed BCE with ground-truth labels
    Total  : alpha * L_soft + beta * L_feat + gamma * L_hard
    """
    def __init__(self, T: float, alpha: float, beta: float,
                 gamma: float, smoothing: float = 0.05):
        super().__init__()
        self.T        = T
        self.alpha    = alpha
        self.beta     = beta
        self.gamma    = gamma
        self.smoothing = smoothing
        self.bce      = nn.BCEWithLogitsLoss()
        self.cos_loss = nn.CosineEmbeddingLoss()

    def forward(self,
                student_logit: torch.Tensor,
                student_proj:  torch.Tensor,
                teacher_logit: torch.Tensor,
                teacher_feats: torch.Tensor,
                true_labels:   torch.Tensor,
                ) -> tuple[torch.Tensor, dict]:

        soft_teacher_targets = torch.sigmoid(
            teacher_logit.float() / self.T
        ).detach()
        L_soft = F.binary_cross_entropy_with_logits(
            student_logit.float() / self.T,
            soft_teacher_targets,
        ) * (self.T ** 2)

        target = torch.ones(student_proj.size(0), device=student_proj.device)
        L_feat = self.cos_loss(student_proj, teacher_feats.detach(), target)

        t_smooth = true_labels * (1 - self.smoothing) + 0.5 * self.smoothing
        L_hard = self.bce(student_logit, t_smooth)

        total = self.alpha * L_soft + self.beta * L_feat + self.gamma * L_hard

        return total, {
            'L_soft': L_soft.item(),
            'L_feat': L_feat.item(),
            'L_hard': L_hard.item(),
        }


kd_criterion = KDLoss(
    T=KD_TEMPERATURE, alpha=ALPHA, beta=BETA,
    gamma=GAMMA, smoothing=LABEL_SMOOTHING
)


# =============================================================================
# OPTIMISER & SCHEDULER
# =============================================================================

backbone_params = list(student.backbone.parameters())
head_params     = list(student.classifier.parameters()) + list(student.proj_head.parameters())

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': LEARNING_RATE / 10},
    {'params': head_params,     'lr': LEARNING_RATE},
], weight_decay=WEIGHT_DECAY)


def lr_lambda(epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler    = torch.cuda.amp.GradScaler()


# =============================================================================
# TRAINING STATE
# =============================================================================

start_epoch       = 0
best_val_acc      = 0.0
epochs_no_improve = 0

history = {
    'train_loss':  [float('nan')] * EPOCHS,
    'train_acc':   [float('nan')] * EPOCHS,
    'val_loss':    [float('nan')] * EPOCHS,
    'val_acc':     [float('nan')] * EPOCHS,
    'lr_backbone': [float('nan')] * EPOCHS,
    'lr_head':     [float('nan')] * EPOCHS,
    'L_soft':      [float('nan')] * EPOCHS,
    'L_feat':      [float('nan')] * EPOCHS,
    'L_hard':      [float('nan')] * EPOCHS,
}


# =============================================================================
# HELPERS
# =============================================================================

def evaluate_loader(loader, desc: str = "VAL") -> tuple[float, float]:
    student.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for rgb_t, hf_t, rgb_s, labels in tqdm(loader, desc=desc, leave=False):
            rgb_t  = rgb_t.to(DEVICE)
            hf_t   = hf_t.to(DEVICE)
            rgb_s  = rgb_s.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)

            with torch.amp.autocast(device_type='cuda'):
                student_proj, student_logit = student.get_features_and_logits(rgb_s)
                teacher_logit = teacher.get_logits(rgb_t, hf_t)
                teacher_feats = teacher.get_fused_features(rgb_t, hf_t)
                loss, _ = kd_criterion(student_logit, student_proj,
                                       teacher_logit, teacher_feats, labels)

            total_loss += loss.item()
            preds   = (torch.sigmoid(student_logit) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return total_loss / len(loader), correct / total


def run_inference(loader, desc: str = "Infer") -> tuple[np.ndarray, np.ndarray]:
    student.eval()
    labels_all, probs_all = [], []
    with torch.no_grad():
        for rgb_t, hf_t, rgb_s, labels in tqdm(loader, desc=desc, leave=False):
            rgb_s = rgb_s.to(DEVICE)
            with torch.amp.autocast(device_type='cuda'):
                logit = student(rgb_s)
            probs = torch.sigmoid(logit).squeeze(1).cpu().numpy()
            probs_all.extend(probs.tolist())
            labels_all.extend(labels.numpy().tolist())
    return np.array(labels_all), np.array(probs_all)


def compute_frame_metrics(labels: np.ndarray, probs: np.ndarray,
                           threshold: float = 0.5) -> dict:
    preds  = (probs >= threshold).astype(int)
    cm     = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        'frame_accuracy':    accuracy_score(labels, preds),
        'frame_f1':          f1_score(labels, preds, zero_division=0),
        'frame_precision':   precision_score(labels, preds, zero_division=0),
        'frame_recall':      recall_score(labels, preds, zero_division=0),
        'frame_specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'frame_auc_roc':     roc_auc_score(labels, probs),
        'frame_ap':          average_precision_score(labels, probs),
        'frame_tp': int(tp), 'frame_fp': int(fp),
        'frame_tn': int(tn), 'frame_fn': int(fn),
    }


def compute_video_metrics(labels: np.ndarray, probs: np.ndarray,
                           subset, threshold: float = 0.5) -> tuple[dict, np.ndarray, np.ndarray]:
    video_probs  = defaultdict(list)
    video_labels = defaultdict(list)
    for i, gi in enumerate(subset.indices):
        path   = full_dataset.all_paths[gi]
        vid_id = full_dataset.path_to_video.get(path, f"unk_{gi}")
        video_probs[vid_id].append(probs[i])
        video_labels[vid_id].append(labels[i])

    vid_true, vid_prob_mean = [], []
    for vid_id in video_probs:
        vid_true.append(1 if 1 in set(video_labels[vid_id]) else 0)
        vid_prob_mean.append(float(np.mean(video_probs[vid_id])))

    vid_true      = np.array(vid_true)
    vid_prob_mean = np.array(vid_prob_mean)
    vid_pred      = (vid_prob_mean >= threshold).astype(int)
    n_classes     = len(np.unique(vid_true))

    metrics = {
        'video_count':     len(vid_true),
        'video_accuracy':  accuracy_score(vid_true, vid_pred),
        'video_f1':        f1_score(vid_true, vid_pred, zero_division=0),
        'video_precision': precision_score(vid_true, vid_pred, zero_division=0),
        'video_recall':    recall_score(vid_true, vid_pred, zero_division=0),
        'video_auc_roc':   roc_auc_score(vid_true, vid_prob_mean) if n_classes > 1 else float('nan'),
        'video_ap':        average_precision_score(vid_true, vid_prob_mean) if n_classes > 1 else float('nan'),
    }
    return metrics, vid_true, vid_prob_mean


# =============================================================================
# TRAINING LOOP
# =============================================================================

print(f"\n{'='*65}")
print(f"  KD Training  |  device: {DEVICE}")
print(f"  Epochs: 1-{EPOCHS}  |  early-stop patience: {PATIENCE}")
print(f"  Loss = {ALPHA}*L_soft + {BETA}*L_feat + {GAMMA}*L_hard  (T={KD_TEMPERATURE})")
print(f"{'='*65}\n")

timed_out = False

for epoch in range(EPOCHS):
    if timed_out:
        break

    student.train()
    run_loss  = 0.0
    run_soft  = run_feat = run_hard = 0.0
    correct   = total = 0
    pbar      = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")

    for rgb_t, hf_t, rgb_s, labels in pbar:
        if time.time() - START_TIME > MAX_RUN_TIME:
            print("\nTime limit reached — saving emergency checkpoint...")
            timed_out = True
            break

        rgb_t  = rgb_t.to(DEVICE)
        hf_t   = hf_t.to(DEVICE)
        rgb_s  = rgb_s.to(DEVICE)
        labels = labels.unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            student_proj, student_logit = student.get_features_and_logits(rgb_s)
            with torch.no_grad():
                teacher_logit = teacher.get_logits(rgb_t, hf_t)
                teacher_feats = teacher.get_fused_features(rgb_t, hf_t)
            loss, loss_parts = kd_criterion(
                student_logit, student_proj,
                teacher_logit, teacher_feats, labels
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        bs         = labels.size(0)
        run_loss  += loss.item()
        run_soft  += loss_parts['L_soft']
        run_feat  += loss_parts['L_feat']
        run_hard  += loss_parts['L_hard']
        preds      = (torch.sigmoid(student_logit) >= 0.5).float()
        correct   += (preds == labels).sum().item()
        total     += bs
        pbar.set_postfix(
            loss=f"{run_loss/total:.4f}",
            acc=f"{correct/total:.4f}",
            Ls=f"{run_soft/(total/bs):.3f}",
            Lf=f"{run_feat/(total/bs):.3f}",
        )

    if timed_out:
        fname = OUT_DIR / f'EMERGENCY_kd_student_epoch{epoch+1:02d}.pth'
        torch.save({
            'epoch': epoch + 1, 'state_dict': student.state_dict(),
            'optimizer': optimizer.state_dict(), 'val_acc': best_val_acc,
            'history': history,
        }, fname)
        print(f"Emergency checkpoint saved: {fname.name}")
        break

    train_acc  = correct / total
    n_batches  = len(train_loader)
    train_loss = run_loss / n_batches

    val_loss, val_acc = evaluate_loader(val_loader, desc=f"Epoch {epoch+1} [VAL]")

    scheduler.step()
    lr_bb   = optimizer.param_groups[0]['lr']
    lr_head = optimizer.param_groups[1]['lr']

    history['train_loss'][epoch]  = train_loss
    history['train_acc'][epoch]   = train_acc
    history['val_loss'][epoch]    = val_loss
    history['val_acc'][epoch]     = val_acc
    history['lr_backbone'][epoch] = lr_bb
    history['lr_head'][epoch]     = lr_head
    history['L_soft'][epoch]      = run_soft / n_batches
    history['L_feat'][epoch]      = run_feat / n_batches
    history['L_hard'][epoch]      = run_hard / n_batches

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train acc: {train_acc:.4f}  loss: {train_loss:.4f} | "
          f"Val acc: {val_acc:.4f}  loss: {val_loss:.4f} | "
          f"LR_head: {lr_head:.2e}")

    ckpt_path = OUT_DIR / f'kd_student_epoch{epoch+1:02d}.pth'
    torch.save({
        'epoch':      epoch + 1,
        'state_dict': student.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'val_acc':    val_acc,
        'history':    history,
    }, ckpt_path)
    print(f"  Saved: {ckpt_path.name}")

    if val_acc > best_val_acc:
        best_val_acc      = val_acc
        epochs_no_improve = 0
        torch.save(student.state_dict(), OUT_DIR / 'best_kd_student.pth')
        print(f"  New best -> best_kd_student.pth  (val_acc={best_val_acc:.4f})")
    else:
        epochs_no_improve += 1
        print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")

    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping after epoch {epoch+1}.")
        break

    print("-" * 65)


# =============================================================================
# TEST EVALUATION
# =============================================================================

print(f"\n{'='*65}")
print("  Loading best student model for final TEST evaluation...")
best_path = OUT_DIR / 'best_kd_student.pth'
if best_path.exists():
    student.load_state_dict(torch.load(best_path, map_location=DEVICE))
else:
    print("  best_kd_student.pth not found — using current weights.")

test_labels, test_probs = run_inference(test_loader, desc="TEST  inference")
val_labels,  val_probs  = run_inference(val_loader,  desc="VAL   inference")

fpr_val, tpr_val, thresh_val = roc_curve(val_labels, val_probs)
best_thresh = float(thresh_val[np.argmax(tpr_val - fpr_val)])
print(f"\n  Optimal threshold (Youden's J on val): {best_thresh:.4f}")

frame_metrics = compute_frame_metrics(test_labels, test_probs, threshold=best_thresh)
print("\n-- Frame-level --")
for k, v in frame_metrics.items():
    print(f"  {k:<25}: {v:.4f}" if isinstance(v, float) else f"  {k:<25}: {v}")

video_metrics, vid_true, vid_prob_mean = compute_video_metrics(
    test_labels, test_probs, test_sub, threshold=best_thresh
)
print("\n-- Video-level --")
for k, v in video_metrics.items():
    print(f"  {k:<25}: {v:.4f}" if isinstance(v, float) else f"  {k:<25}: {v}")

preds_test = (test_probs >= best_thresh).astype(int)
print("\n-- Classification Report (frame-level) --")
print(classification_report(test_labels, preds_test,
                             target_names=['REAL', 'FAKE'], digits=4))


# =============================================================================
# EXPORT CSV FILES
# =============================================================================

all_metrics = {
    'model': 'EfficientNet-B0 (KD student)',
    'teacher': 'DPViT',
    'kd_temperature': KD_TEMPERATURE,
    'alpha': ALPHA, 'beta': BETA, 'gamma': GAMMA,
    'optimal_threshold': best_thresh,
    **{f'test_{k}': v for k, v in frame_metrics.items()},
    **{f'test_{k}': v for k, v in video_metrics.items()},
    'best_val_acc': best_val_acc,
}
pd.DataFrame([all_metrics]).to_csv(OUT_DIR / 'kd_metrics_summary.csv', index=False)
print("Saved: kd_metrics_summary.csv")

valid_ep = [e for e in range(EPOCHS) if not math.isnan(history['train_loss'][e])]
hist_df = pd.DataFrame({
    'epoch':       [e + 1 for e in valid_ep],
    'train_loss':  [history['train_loss'][e]  for e in valid_ep],
    'train_acc':   [history['train_acc'][e]   for e in valid_ep],
    'val_loss':    [history['val_loss'][e]     for e in valid_ep],
    'val_acc':     [history['val_acc'][e]      for e in valid_ep],
    'lr_backbone': [history['lr_backbone'][e]  for e in valid_ep],
    'lr_head':     [history['lr_head'][e]      for e in valid_ep],
    'L_soft':      [history['L_soft'][e]       for e in valid_ep],
    'L_feat':      [history['L_feat'][e]       for e in valid_ep],
    'L_hard':      [history['L_hard'][e]       for e in valid_ep],
})
hist_df.to_csv(OUT_DIR / 'kd_training_history.csv', index=False)
print("Saved: kd_training_history.csv")

pd.DataFrame({
    'file_path':  [full_dataset.all_paths[i] for i in test_sub.indices],
    'true_label': test_labels.astype(int),
    'prob_fake':  np.round(test_probs, 6),
    'pred_label': preds_test,
    'correct':    (test_labels == preds_test).astype(int),
}).to_csv(OUT_DIR / 'kd_test_frame_predictions.csv', index=False)
print("Saved: kd_test_frame_predictions.csv")


# =============================================================================
# VISUALISATIONS
# =============================================================================

plt.rcParams.update({
    'figure.facecolor':  '#0a0e1a',
    'axes.facecolor':    '#111827',
    'axes.edgecolor':    '#2d3748',
    'axes.labelcolor':   '#e2e8f0',
    'text.color':        '#e2e8f0',
    'xtick.color':       '#94a3b8',
    'ytick.color':       '#94a3b8',
    'grid.color':        '#1e2a3a',
    'grid.linestyle':    '--',
    'grid.alpha':        0.7,
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'legend.facecolor':  '#111827',
    'legend.edgecolor':  '#2d3748',
    'savefig.facecolor': '#0a0e1a',
    'savefig.dpi':       200,
    'savefig.bbox':      'tight',
})

C_BLUE   = '#38bdf8'
C_RED    = '#f87171'
C_GREEN  = '#4ade80'
C_YELLOW = '#fbbf24'
C_PURPLE = '#c084fc'
C_GREY   = '#475569'
epochs_x = hist_df['epoch'].tolist()


fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ax = axes[0]
ax.plot(epochs_x, hist_df['train_loss'], color=C_BLUE,  lw=2.5, marker='o', ms=5, label='Train')
ax.plot(epochs_x, hist_df['val_loss'],   color=C_RED,   lw=2.5, marker='s', ms=5, label='Val', linestyle='--')
ax.set_title('Total KD Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True); ax.set_xticks(epochs_x)
ax = axes[1]
ax.plot(epochs_x, hist_df['L_soft'], color=C_BLUE,   lw=2, marker='o', ms=4, label=f'L_soft (alpha={ALPHA})')
ax.plot(epochs_x, hist_df['L_feat'], color=C_GREEN,  lw=2, marker='s', ms=4, label=f'L_feat (beta={BETA})',  linestyle='--')
ax.plot(epochs_x, hist_df['L_hard'], color=C_YELLOW, lw=2, marker='D', ms=4, label=f'L_hard (gamma={GAMMA})', linestyle=':')
ax.set_title('KD Loss Component Breakdown'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True); ax.set_xticks(epochs_x)
fig.suptitle('Knowledge Distillation — Loss Curves', fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig1_kd_loss_curves.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs_x, hist_df['train_acc'], color=C_GREEN,  lw=2.5, marker='o', ms=5, label='Train Acc')
ax.plot(epochs_x, hist_df['val_acc'],   color=C_YELLOW, lw=2.5, marker='s', ms=5, label='Val Acc', linestyle='--')
ax.axhline(frame_metrics['frame_accuracy'], color=C_RED, lw=1.8, linestyle=':',
           label=f"Test Acc ({frame_metrics['frame_accuracy']:.4f})")
ax.set_title('Student Accuracy — Train / Val / Test')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.05)
ax.legend(); ax.grid(True); ax.set_xticks(epochs_x)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig2_accuracy_curve.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs_x, hist_df['lr_head'],     color=C_BLUE,   lw=2.5, marker='o', ms=5, label='Head LR')
ax.plot(epochs_x, hist_df['lr_backbone'], color=C_PURPLE, lw=2.5, marker='s', ms=5, label='Backbone LR (÷10)', linestyle='--')
ax.set_title('Differential Learning Rate Schedule'); ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
ax.set_yscale('log'); ax.legend(); ax.grid(True); ax.set_xticks(epochs_x)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig3_lr_schedule.png')
plt.close(fig)

cm      = confusion_matrix(test_labels, preds_test)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, data, fmt, title in zip(
    axes, [cm, cm_norm], ['d', '.3f'],
    ['Confusion Matrix — Raw Counts', 'Confusion Matrix — Normalised'],
):
    sns.heatmap(data, annot=True, fmt=fmt, ax=ax, cmap='Blues',
                xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'],
                linewidths=0.5, linecolor='#2d3748',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(title, pad=10); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
fig.suptitle('Frame-Level Confusion Matrix (EfficientNet-B0 Student)', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig4_confusion_matrix.png')
plt.close(fig)

fpr_f, tpr_f, _ = roc_curve(test_labels, test_probs)
auc_f            = roc_auc_score(test_labels, test_probs)
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr_f, tpr_f, color=C_BLUE, lw=2.5, label=f'Frame AUC = {auc_f:.4f}')
if not math.isnan(video_metrics['video_auc_roc']):
    fpr_v, tpr_v, _ = roc_curve(vid_true, vid_prob_mean)
    ax.plot(fpr_v, tpr_v, color=C_GREEN, lw=2.5, linestyle='--',
            label=f"Video AUC = {video_metrics['video_auc_roc']:.4f}")
ax.plot([0, 1], [0, 1], color=C_GREY, lw=1.5, linestyle=':')
opt_idx = np.argmax(tpr_f - fpr_f)
ax.scatter(fpr_f[opt_idx], tpr_f[opt_idx], color=C_RED, s=100, zorder=5,
           label=f'Optimal point (tau={best_thresh:.3f})')
ax.set_title('ROC-AUC Curve — EfficientNet-B0 Student')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
ax.legend(loc='lower right'); ax.grid(True)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig5_roc_auc.png')
plt.close(fig)

prec_f, rec_f, _ = precision_recall_curve(test_labels, test_probs)
ap_f             = average_precision_score(test_labels, test_probs)
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(rec_f, prec_f, color=C_YELLOW, lw=2.5, label=f'Frame AP = {ap_f:.4f}')
if not math.isnan(video_metrics['video_ap']):
    prec_v, rec_v, _ = precision_recall_curve(vid_true, vid_prob_mean)
    ax.plot(rec_v, prec_v, color=C_GREEN, lw=2.5, linestyle='--',
            label=f"Video AP = {video_metrics['video_ap']:.4f}")
baseline = test_labels.mean()
ax.axhline(baseline, color=C_GREY, lw=1.5, linestyle=':', label=f'Baseline = {baseline:.3f}')
ax.set_title('Precision-Recall Curve'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
ax.legend(); ax.grid(True)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig6_precision_recall.png')
plt.close(fig)

real_probs = test_probs[test_labels == 0]
fake_probs = test_probs[test_labels == 1]
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(real_probs, bins=60, color=C_GREEN, alpha=0.65, label='REAL', density=True)
ax.hist(fake_probs, bins=60, color=C_RED,   alpha=0.65, label='FAKE', density=True)
ax.axvline(best_thresh, color=C_YELLOW, lw=2, linestyle='--',
           label=f'Threshold = {best_thresh:.3f}')
ax.set_title('Predicted Probability Distribution — Real vs Fake')
ax.set_xlabel('P(fake)'); ax.set_ylabel('Density')
ax.legend(); ax.grid(True)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig7_score_distribution.png')
plt.close(fig)

manip_acc = {}
for manip in FAKE_FOLDER_NAMES + [REAL_FOLDER_NAME]:
    idxs = [i for i, gi in enumerate(test_sub.indices)
            if manip in full_dataset.all_paths[gi]]
    if idxs:
        manip_acc[manip] = accuracy_score(test_labels[idxs], preds_test[idxs])
if manip_acc:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = [C_GREEN if k == REAL_FOLDER_NAME else C_RED for k in manip_acc]
    bars = ax.bar(list(manip_acc.keys()), list(manip_acc.values()),
                  color=colors_bar, edgecolor='#2d3748', linewidth=0.8)
    ax.bar_label(bars, fmt='%.4f', padding=4, color='#e2e8f0', fontsize=10)
    ax.axhline(frame_metrics['frame_accuracy'], color=C_YELLOW, lw=1.8, linestyle='--',
               label=f"Overall = {frame_metrics['frame_accuracy']:.4f}")
    ax.set_title('Per-Manipulation Type Accuracy (Student)')
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.12)
    ax.legend(); ax.grid(True, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig8_per_manipulation_accuracy.png')
    plt.close(fig)

fig = plt.figure(figsize=(22, 15))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(epochs_x, hist_df['train_loss'], color=C_BLUE,  lw=2, marker='o', ms=4, label='Train')
ax0.plot(epochs_x, hist_df['val_loss'],   color=C_RED,   lw=2, marker='s', ms=4, label='Val', linestyle='--')
ax0.set_title('Total Loss'); ax0.set_xlabel('Epoch'); ax0.legend(fontsize=9); ax0.grid(True)

ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(epochs_x, hist_df['train_acc'], color=C_GREEN,  lw=2, marker='o', ms=4, label='Train')
ax1.plot(epochs_x, hist_df['val_acc'],   color=C_YELLOW, lw=2, marker='s', ms=4, label='Val', linestyle='--')
ax1.axhline(frame_metrics['frame_accuracy'], color=C_RED, lw=1.5, linestyle=':', label='Test')
ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=9); ax1.grid(True)

ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(epochs_x, hist_df['L_soft'], color=C_BLUE,   lw=2, marker='o', ms=4, label=f'L_soft alpha={ALPHA}')
ax2.plot(epochs_x, hist_df['L_feat'], color=C_GREEN,  lw=2, marker='s', ms=4, label=f'L_feat beta={BETA}',  linestyle='--')
ax2.plot(epochs_x, hist_df['L_hard'], color=C_YELLOW, lw=2, marker='D', ms=4, label=f'L_hard gamma={GAMMA}', linestyle=':')
ax2.set_title('Loss Components'); ax2.set_xlabel('Epoch')
ax2.legend(fontsize=8); ax2.grid(True)

ax3 = fig.add_subplot(gs[1, 0])
sns.heatmap(cm_norm, annot=True, fmt='.3f', ax=ax3, cmap='Blues',
            xticklabels=['REAL','FAKE'], yticklabels=['REAL','FAKE'],
            annot_kws={'size':11,'weight':'bold'}, linewidths=0.5, linecolor='#2d3748')
ax3.set_title('Confusion Matrix (norm.)'); ax3.set_xlabel('Predicted'); ax3.set_ylabel('True')

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(fpr_f, tpr_f, color=C_BLUE, lw=2, label=f'AUC={auc_f:.4f}')
ax4.plot([0,1],[0,1], color=C_GREY, lw=1, linestyle=':')
ax4.set_title('ROC-AUC'); ax4.set_xlabel('FPR'); ax4.set_ylabel('TPR')
ax4.legend(fontsize=9); ax4.grid(True)

ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(rec_f, prec_f, color=C_YELLOW, lw=2, label=f'AP={ap_f:.4f}')
ax5.set_title('Precision-Recall'); ax5.set_xlabel('Recall'); ax5.set_ylabel('Precision')
ax5.legend(fontsize=9); ax5.grid(True)

ax6 = fig.add_subplot(gs[2, :2])
ax6.hist(real_probs, bins=50, color=C_GREEN, alpha=0.65, label='REAL', density=True)
ax6.hist(fake_probs, bins=50, color=C_RED,   alpha=0.65, label='FAKE', density=True)
ax6.axvline(best_thresh, color=C_YELLOW, lw=2, linestyle='--', label=f'tau={best_thresh:.3f}')
ax6.set_title('Score Distribution'); ax6.set_xlabel('P(fake)'); ax6.set_ylabel('Density')
ax6.legend(fontsize=9); ax6.grid(True)

ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
table_data = [
    ['Metric',      'Frame',                                              'Video'],
    ['Accuracy',    f"{frame_metrics['frame_accuracy']:.4f}",            f"{video_metrics['video_accuracy']:.4f}"],
    ['F1',          f"{frame_metrics['frame_f1']:.4f}",                  f"{video_metrics['video_f1']:.4f}"],
    ['Precision',   f"{frame_metrics['frame_precision']:.4f}",           f"{video_metrics['video_precision']:.4f}"],
    ['Recall',      f"{frame_metrics['frame_recall']:.4f}",              f"{video_metrics['video_recall']:.4f}"],
    ['Specificity', f"{frame_metrics['frame_specificity']:.4f}",         '—'],
    ['AUC-ROC',     f"{frame_metrics['frame_auc_roc']:.4f}",             f"{video_metrics['video_auc_roc']:.4f}" if not math.isnan(video_metrics['video_auc_roc']) else 'N/A'],
    ['AP',          f"{frame_metrics['frame_ap']:.4f}",                  f"{video_metrics['video_ap']:.4f}" if not math.isnan(video_metrics['video_ap']) else 'N/A'],
    ['TP/FP/TN/FN', f"{frame_metrics['frame_tp']}/{frame_metrics['frame_fp']}/{frame_metrics['frame_tn']}/{frame_metrics['frame_fn']}", '—'],
]
tbl = ax7.table(cellText=table_data[1:], colLabels=table_data[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.1, 1.55)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#111827' if r % 2 == 0 else '#0a0e1a')
    cell.set_edgecolor('#2d3748')
    cell.set_text_props(color='#e2e8f0')
ax7.set_title('Evaluation Summary', pad=12)

fig.suptitle(
    f'EfficientNet-B0 (KD Student) — Full Evaluation Dashboard\n'
    f'Teacher: DPViT  |  T={KD_TEMPERATURE}  alpha={ALPHA}  beta={BETA}  gamma={GAMMA}',
    fontsize=14, y=1.01, color='#e2e8f0'
)
fig.savefig(OUT_DIR / 'fig9_summary_dashboard.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved: fig9_summary_dashboard.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{'='*65}")
print(f"  ALL OUTPUTS -> {OUT_DIR}")
print(f"{'='*65}")
print(f"\n  Student frame AUC  : {frame_metrics['frame_auc_roc']:.4f}")
print(f"  Student frame Acc  : {frame_metrics['frame_accuracy']:.4f}")
print(f"  Student frame F1   : {frame_metrics['frame_f1']:.4f}")
print(f"  Student video Acc  : {video_metrics['video_accuracy']:.4f}")
print(f"  Student video AUC  : {video_metrics['video_auc_roc']:.4f}")
print(f"\n  Teacher params : {teacher_params:.1f} M  ->  Student params : {student_params:.1f} M")
print(f"  Model compression  : {teacher_params / student_params:.1f}x smaller")
print(f"\n  Done. Use best_kd_student.pth for deployment.")