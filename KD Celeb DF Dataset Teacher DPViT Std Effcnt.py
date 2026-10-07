import os, cv2, gc, time, math, warnings, json
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, update_bn
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm.auto import tqdm
import timm
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, average_precision_score, precision_recall_curve,
    matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score,
    brier_score_loss, log_loss, classification_report
)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

FRAME_ROOT = "/kaggle/input/datasets/syedazmulhasansabbir/celeb-df-frames/content/drive/MyDrive/dataset/celeb_df_frames"
TEACHER_PATH = "/kaggle/input/models/syedazmulhasansabbir/dp-vit-celeb-latest/tensorflow2/default/1/epoch_5.pth"
STUDENT_CKPT = "/kaggle/input/models/syedazmulhasan2001/ep28/tensorflow2/default/1/kd_student_epoch28.pth"

OUT_DIR = Path("/kaggle/working/kd_phase2_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PHASE2_EPOCHS = 20
WARMUP_EPOCHS = 1
N_RESTARTS = 3
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
SWA_START_EPOCH = 5
SWA_LR = 5e-6

ALPHA, BETA, GAMMA = 0.30, 0.20, 0.50
KD_TEMPERATURE = 4.0
LABEL_SMOOTHING = 0.05

BATCH_SIZE = 32
IMG_SIZE = 224
SEED = 42
GRAD_CLIP = 1.0
PATIENCE = 7
DROP_THRESHOLD = 0.04
TTA_ENABLED = True
TTA_N_AUGMENTS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# DATASET & TRANSFORMS
# ──────────────────────────────────────────────────────────────────────────────

class CelebDFDataset(Dataset):
    def __init__(self, root_dir: str):
        valid_exts = {'.jpg', '.jpeg', '.png'}
        self.real_paths, self.fake_paths = [], []
        self.path_to_video, self.path_to_manip = {}, {}

        for cls_name in ["real", "fake"]:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir): continue
            for vid_id in sorted(os.listdir(cls_dir)):
                vid_dir = os.path.join(cls_dir, vid_id)
                if not os.path.isdir(vid_dir): continue
                for f in os.listdir(vid_dir):
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        fp = os.path.join(vid_dir, f)
                        if cls_name == "real":
                            self.real_paths.append(fp)
                        else:
                            self.fake_paths.append(fp)
                        self.path_to_video[fp] = f"{cls_name}_{vid_id}"
                        self.path_to_manip[fp] = cls_name

        self.all_paths = self.real_paths + self.fake_paths
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)

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
    def __init__(self, subset, tf, is_train: bool):
        self.subset = subset
        self.tf = tf

    def __len__(self): return len(self.subset)

    def __getitem__(self, idx):
        img_rgb, img_hf, label = self.subset[idx]
        return self.tf(img_rgb), self.tf(img_hf), self.tf(img_rgb), torch.tensor(label, dtype=torch.float32)


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_dataset = CelebDFDataset(FRAME_ROOT)
n_train = int(0.8 * len(full_dataset))
n_val = int(0.1 * len(full_dataset))
n_test = len(full_dataset) - n_train - n_val
train_sub, val_sub, test_sub = random_split(full_dataset, [n_train, n_val, n_test],
                                            generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(KDTransformWrapper(train_sub, train_tf, True), batch_size=BATCH_SIZE,
                          sampler=WeightedRandomSampler([1.0 / len(train_sub)] * len(train_sub), len(train_sub)),
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(KDTransformWrapper(val_sub, eval_tf, False), batch_size=BATCH_SIZE, num_workers=4)
test_loader = DataLoader(KDTransformWrapper(test_sub, eval_tf, False), batch_size=BATCH_SIZE, num_workers=4)


# ──────────────────────────────────────────────────────────────────────────────
# MODELS (TEACHER & STUDENT)
# ──────────────────────────────────────────────────────────────────────────────

def get_teacher_model(path):
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    embed_dim = next(v.shape[-1] for k, v in sd.items() if 'cls_token' in k)
    vit_map = {192: 'vit_tiny_patch16_224', 384: 'vit_small_patch16_224', 768: 'vit_base_patch16_224'}
    variant = vit_map[embed_dim]

    class Teacher(nn.Module):
        def __init__(self, variant):
            super().__init__()
            self.rgb_branch = timm.create_model(variant, pretrained=False, num_classes=0)
            self.hf_branch = timm.create_model(variant, pretrained=False, num_classes=0)
            self.classifier = nn.Sequential(nn.Linear(embed_dim * 2, 512), nn.GELU(), nn.Linear(512, 2))

        def forward(self, r, f):
            feats = torch.cat([self.rgb_branch(r), self.hf_branch(f)], dim=1)
            return self.classifier(feats)

        def get_logits(self, r, f): return self.forward(r, f)[:, 1:2]

        def get_fused_features(self, r, f): return torch.cat([self.rgb_branch(r), self.hf_branch(f)], dim=1)

    model = Teacher(variant)
    # Flexible key remapping
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace('rgb.', 'rgb_branch.').replace('freq.', 'hf_branch.').replace('head.', 'classifier.')
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False)
    return model.to(DEVICE).eval(), embed_dim * 2


class EfficientNetStudent(nn.Module):
    def __init__(self, t_feat_dim):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        s_feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(s_feat_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.proj_head = nn.Sequential(nn.LayerNorm(s_feat_dim), nn.Linear(s_feat_dim, t_feat_dim))

    def forward(self, x): return self.classifier(self.backbone(x))

    def get_features_and_logits(self, x):
        f = self.backbone(x)
        return self.proj_head(f), self.classifier(f)


teacher, TEACHER_FEAT_DIM = get_teacher_model(TEACHER_PATH)
student = EfficientNetStudent(TEACHER_FEAT_DIM).to(DEVICE)
student.load_state_dict(torch.load(STUDENT_CKPT, map_location=DEVICE)['state_dict'], strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# LOSS & OPTIMIZER
# ──────────────────────────────────────────────────────────────────────────────

class KDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.cos = nn.CosineEmbeddingLoss()

    def forward(self, s_logit, s_proj, t_logit, t_feat, labels):
        L_soft = F.binary_cross_entropy_with_logits(s_logit / KD_TEMPERATURE,
                                                    torch.sigmoid(t_logit / KD_TEMPERATURE).detach()) * (
                             KD_TEMPERATURE ** 2)
        L_feat = self.cos(s_proj, t_feat.detach(), torch.ones(s_logit.size(0), device=DEVICE))
        L_hard = self.bce(s_logit, labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING)
        return ALPHA * L_soft + BETA * L_feat + GAMMA * L_hard, {'L_soft': L_soft.item(), 'L_feat': L_feat.item(),
                                                                 'L_hard': L_hard.item()}


criterion = KDLoss()
optimizer = optim.AdamW([{'params': student.backbone.parameters(), 'lr': LR_BACKBONE},
                         {'params': student.classifier.parameters(), 'lr': LR_HEAD}], weight_decay=1e-4)


class CosineWarmRestart:
    def __init__(self, optimizer, base_lrs):
        self.opt = optimizer
        self.base_lrs = base_lrs
        self.cycle = PHASE2_EPOCHS // N_RESTARTS
        self.current = 0

    def step(self):
        cyc_idx = self.current % self.cycle
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            if cyc_idx < WARMUP_EPOCHS:
                lr = base * (cyc_idx + 1) / WARMUP_EPOCHS
            else:
                ratio = (cyc_idx - WARMUP_EPOCHS) / (self.cycle - WARMUP_EPOCHS)
                lr = base * 0.01 + 0.5 * (base - base * 0.01) * (1 + math.cos(math.pi * ratio))
            g['lr'] = lr
        self.current += 1


scheduler = CosineWarmRestart(optimizer, [LR_BACKBONE, LR_HEAD])
swa_model = AveragedModel(student)
scaler = torch.cuda.amp.GradScaler()

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

history = defaultdict(list)
best_acc = 0.0

for epoch in range(1, PHASE2_EPOCHS + 1):
    student.train()
    scheduler.step()

    metrics = defaultdict(float)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for r_t, h_t, r_s, lbl in pbar:
        r_t, h_t, r_s, lbl = r_t.to(DEVICE), h_t.to(DEVICE), r_s.to(DEVICE), lbl.unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            s_proj, s_logit = student.get_features_and_logits(r_s)
            with torch.no_grad():
                t_logit, t_feat = teacher.get_logits(r_t, h_t), teacher.get_fused_features(r_t, h_t)
            loss, parts = criterion(s_logit, s_proj, t_logit, t_feat, lbl)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        scaler.step(optimizer);
        scaler.update()

        metrics['loss'] += loss.item()
        for k, v in parts.items(): metrics[k] += v

    # Validation
    student.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for _, _, r_s, lbl in val_loader:
            r_s, lbl = r_s.to(DEVICE), lbl.to(DEVICE)
            out = torch.sigmoid(student(r_s)).squeeze()
            val_correct += ((out >= 0.5).float() == lbl).sum().item()
            val_total += lbl.size(0)

    val_acc = val_correct / val_total
    if epoch >= SWA_START_EPOCH: swa_model.update_parameters(student)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(student.state_dict(), OUT_DIR / "best_kd_p2_student.pth")

    print(f"Epoch {epoch} | Val Acc: {val_acc:.4f} | Loss: {metrics['loss'] / len(train_loader):.4f}")

# Finalize SWA
update_bn(train_loader, swa_model, device=DEVICE)
torch.save(swa_model.module.state_dict(), OUT_DIR / "swa_kd_p2_student.pth")


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE & TTA
# ──────────────────────────────────────────────────────────────────────────────

def tta_inference(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for _, _, r_s, lbl in tqdm(loader, desc="TTA"):
            r_s = r_s.to(DEVICE)
            # TTA: Original, HFlip, VFlip, Brighter
            aug1 = r_s
            aug2 = torch.flip(r_s, [3])
            aug3 = torch.flip(r_s, [2])

            p = (torch.sigmoid(model(aug1)) + torch.sigmoid(model(aug2)) + torch.sigmoid(model(aug3))) / 3
            all_probs.extend(p.cpu().numpy())
            all_labels.extend(lbl.numpy())
    return np.array(all_labels), np.array(all_probs).flatten()


y_true, y_prob = tta_inference(student, test_loader)
print("\nFinal Test Results (TTA):")
print(classification_report(y_true, (y_prob >= 0.5).astype(int), digits=4))