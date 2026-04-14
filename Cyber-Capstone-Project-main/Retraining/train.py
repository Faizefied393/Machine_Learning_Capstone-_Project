# ──────────────────────────────────────────────────────────────────────────────
#  train.py  – fine-tune crane CNN with 3× data augmentation per image
# ──────────────────────────────────────────────────────────────────────────────
import os, argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ───────────────  hyper-parameters (override via CLI)  ───────────────
DEF_EPOCHS       =  5
DEF_BATCH_SIZE   = 32
DEF_LR           =  1e-4
DEF_WEIGHT_DECAY =  1e-5
DEF_IMG_SIZE     = (128, 128)

# ───────────────  model (identical to prod)  ───────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.20),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.30),
        )
        with torch.no_grad():
            f_size = self.features(torch.zeros(1, 3, *DEF_IMG_SIZE)).view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(f_size, 128), nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(128, 64),    nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(64,   4),    nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(4,     1),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class CraneDataset(Dataset):
    """
    original + 3 augments per real image
    """
    _EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, img_dir: Path, xls_path: Path):
        df = pd.read_excel(xls_path)

        # build a lookup: STEM (upper-case)  ->  full Path
        img_lookup = {p.stem.upper(): p
                      for p in img_dir.iterdir()
                      if p.suffix.lower() in self._EXTS}

        samples, missing = [], []
        for pict, cnt in zip(df["PICT Num"], df["WHCR Cnt"]):
            raw  = str(pict).strip()          # whatever Excel gives us
            stem = Path(raw).stem.upper()     # removes extension if present

            # 1️⃣ exact file name (including ext) exists?
            direct = img_dir / raw
            if direct.exists():
                samples.append((direct, 1 if cnt > 0 else 0))
                continue

            # 2️⃣ match by stem (case-insensitive)
            if stem in img_lookup:
                samples.append((img_lookup[stem], 1 if cnt > 0 else 0))
            else:
                missing.append(raw)

        if not samples:
            raise RuntimeError(
                f"No images matched. Checked {len(df)} rows; "
                f"your --data folder is: {img_dir}"
            )

        if missing:
            print(f"⚠️  {len(missing)} rows had no matching image "
                  f"(showing first 10): {missing[:10]}")

        self.samples = samples
        # ---------- transforms / augments stay unchanged ----------
        self.base = transforms.Compose([
            transforms.Resize(DEF_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        self.augments = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3),
            transforms.RandomRotation(degrees=25),
        ]

    def __len__(self):
        return len(self.samples) * 4

    def __getitem__(self, idx):
        real_idx, aug_id = divmod(idx, 4)
        img_path, label  = self.samples[real_idx]

        img = Image.open(img_path).convert("RGB")
        if aug_id > 0:
            img = self.augments[aug_id-1](img)
        img = self.base(img)
        return img, torch.tensor([label], dtype=torch.float32)


# ───────────────  small progress helper  ───────────────
def progress(epoch, step, total, loss):
    bar_len = 30
    filled  = int(bar_len * step / total)
    bar = "█"*filled + "·"*(bar_len-filled)
    print(f"\r[E{epoch+1}] |{bar}| {step}/{total}  loss={loss:.4f}", end="")

# ───────────────  training loop  ───────────────
def train(model, loader, epochs, device, lr, wd):
    model.to(device)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        running = 0.0
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            progress(ep, i, len(loader), running/i)
        print()

# ───────────────  entry-point  ───────────────
if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--data",   default="data/images")
    a.add_argument("--labels", default="data/labels.xlsx")
    a.add_argument("--epochs", type=int,   default=DEF_EPOCHS)
    a.add_argument("--bs",     type=int,   default=DEF_BATCH_SIZE)
    a.add_argument("--lr",     type=float, default=DEF_LR)
    a.add_argument("--wd",     type=float, default=DEF_WEIGHT_DECAY)
    a.add_argument("--weights",default="../lambda/cnn_model.pt")
    a.add_argument("--out",    default="cnn_model_finetuned.pt")
    args = a.parse_args()

    # mac-friendly device choice: Apple Silicon (mps) or CPU
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device →", device)

    ds = CraneDataset(Path(args.data), Path(args.labels))
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0)

    model = CNN()
    if Path(args.weights).exists():
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print("Loaded pre-trained weights:", args.weights)

    train(model, dl, args.epochs, device, args.lr, args.wd)

    torch.save(model.state_dict(), args.out)
    print("✅ Saved fine-tuned weights →", args.out)
