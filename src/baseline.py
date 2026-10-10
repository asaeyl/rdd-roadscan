"""BASELINE: image-level multi-label classifier (ResNet-18, 4 sigmoid outputs, BCE loss).
Question it answers: "which damage types are in this image?" - no locations."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src import config as cfg
from src.metrics import mean_average_precision


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size=cfg.IMAGE_SIZE):
    """Train: horizontal flip + colour jitter only (a vertical flip would turn the road upside down)."""
    normalize = transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD)
    train = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(0.5),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                transforms.ToTensor(), normalize])
    evaluate = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])
    return train, evaluate


class RoadDamageDataset(Dataset):
    """Gives (image tensor, 4 numbers 0/1 = which damage types are present)."""

    def __init__(self, table, transform):
        self.image_paths = table.image_path.tolist()
        self.labels = table[cfg.HAS_COLUMNS].values.astype("float32")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i]).convert("RGB")
        return self.transform(image), torch.from_numpy(self.labels[i])


def make_loader(table, transform, shuffle, batch_size=cfg.BATCH_SIZE):
    generator = torch.Generator().manual_seed(cfg.SEED)
    return DataLoader(RoadDamageDataset(table, transform), batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, generator=generator)


def build_model(pretrained=True):
    """ResNet-18 with the last layer replaced by 4 outputs."""
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(cfg.DAMAGE_CLASSES))
    return model


@torch.no_grad()
def predict_scores(model, table, image_size=cfg.IMAGE_SIZE):
    """Returns (scores, true labels): probabilities between 0 and 1, shape (images, 4)."""
    device = next(model.parameters()).device
    _, eval_transform = build_transforms(image_size)
    loader = make_loader(table, eval_transform, shuffle=False)
    model.eval()
    all_scores, all_labels = [], []
    for images, labels in loader:
        all_scores.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


def train_baseline(train_df, val_df, out_dir, epochs=cfg.EPOCHS, image_size=cfg.IMAGE_SIZE):
    """Train, keep best.pt (best validation mean AP) and last.pt (final epoch). Returns the training history."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    train_transform, _ = build_transforms(image_size)
    train_loader = make_loader(train_df, train_transform, shuffle=True)

    model = build_model().to(device)
    share_positive = train_df[cfg.HAS_COLUMNS].values.mean(axis=0)               # how common each damage type is
    pos_weight = torch.tensor((1 - share_positive) / share_positive, dtype=torch.float32)
    pos_weight = pos_weight.clamp(max=cfg.BASELINE_MAX_POS_WEIGHT).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": cfg.BASELINE_LR_BACKBONE},
                                   {"params": model.fc.parameters(), "lr": cfg.BASELINE_LR_HEAD}],
                                  weight_decay=cfg.BASELINE_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history, best_val_map = [], -1
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_function(model(images), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(labels)
        scheduler.step()

        val_scores, val_labels = predict_scores(model, val_df, image_size)
        val_map = mean_average_precision(val_labels, val_scores)
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), out_dir / "best.pt")
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_df), "val_mean_AP": val_map})
        print(f"epoch {epoch:02d} | train loss {history[-1]['train_loss']:.3f} | validation mean AP {val_map:.3f}")

    if cfg.SAVE_LAST_WEIGHTS:
        torch.save(model.state_dict(), out_dir / "last.pt")
    return pd.DataFrame(history)


def load_model(weights_path):
    """Load saved baseline weights (no retraining)."""
    device = get_device()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()


def model_cost(weights_path, image_size=cfg.IMAGE_SIZE, n_runs=30):
    """Number of parameters (millions) and CPU time per image (ms)."""
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    params_millions = sum(p.numel() for p in model.parameters()) / 1e6
    one_image = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        for _ in range(5):                                                       # warm-up
            model(one_image)
        start = time.time()
        for _ in range(n_runs):
            model(one_image)
    return params_millions, (time.time() - start) / n_runs * 1000
