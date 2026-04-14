# ──────────────────────────────────────────────────────────────────────────────
#  train-long.py  – 200-epoch fine-tuning with 3× positive class weight
# ──────────────────────────────────────────────────────────────────────────────
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ───────────────  hyper-parameters  ───────────────
DEF_EPOCHS       = 200          # ← train “a lot longer”
DEF_BATCH_SIZE   = 32
DEF_LR           = 1e-4
DEF_WEIGHT_DECAY = 1e-5
DEF_IMG_SIZE     = (128, 128)
POS_WEIGHT       = 3.0          # ← positives count 3×

# ───────────────  model (unchanged)  ───────────────
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
            f_sz = self.features(torch.zeros(1, 3, *DEF_IMG_SIZE)).view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(f_sz, 128), nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(64,   4),   nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(4,     1),
        )
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)

# ───────────────  dataset (unchanged)  ───────────────
class CraneDataset(Dataset):
    _EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, img_dir: Path, xls_path: Path):
        df = pd.read_excel(xls_path)

        # build lookup STEM→Path (case-insensitive)
        imgs = {p.stem.upper(): p for p in img_dir.iterdir()
                if p.suffix.lower() in self._EXTS}

        self.samples = []
        for pict, cnt in zip(df["PICT Num"], df["WHCR Cnt"]):
            raw  = str(pict).strip()
            stem = Path(raw).stem.upper()

            if (direct := img_dir / raw).exists():
                self.samples.append((direct, int(cnt > 0)))
            elif stem in imgs:
                self.samples.append((imgs[stem], int(cnt > 0)))

        if not self.samples:
            raise RuntimeError("No matching images – check --data/--labels paths.")

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

    def __len__(self):             return len(self.samples) * 4   # 1 original + 3 augments
    def __getitem__(self, idx):
        real_idx, aug_id = divmod(idx, 4)
        path, label      = self.samples[real_idx]

        img = Image.open(path).convert("RGB")
        if aug_id:  img = self.augments[aug_id-1](img)
        img = self.base(img)
        return img, torch.tensor([label], dtype=torch.float32)

# ───────────────  training helpers  ───────────────
def progress(ep, step, total, loss):
    bar_len = 30
    filled  = int(bar_len * step / total)
    bar     = "█"*filled + "·"*(bar_len-filled)
    print(f"\r[E{ep+1:03}] |{bar}| {step}/{total}  loss={loss:.4f}", end="")

def train(model, loader, epochs, device, lr, wd, pos_wt):
    model.to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_wt], device=device))
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

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

# ───────────────  entry point  ───────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="data/images")
    p.add_argument("--labels", default="data/labels.xlsx")
    p.add_argument("--epochs", type=int,   default=DEF_EPOCHS)
    p.add_argument("--bs",     type=int,   default=DEF_BATCH_SIZE)
    p.add_argument("--lr",     type=float, default=DEF_LR)
    p.add_argument("--wd",     type=float, default=DEF_WEIGHT_DECAY)
    p.add_argument("--poswt",  type=float, default=POS_WEIGHT, help="weight for positive class")
    p.add_argument("--weights",default="../lambda/cnn_model.pt")
    p.add_argument("--out",    default="cnn_model_finetuned.pt")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device →", device)

    ds = CraneDataset(Path(args.data), Path(args.labels))
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0)

    net = CNN()
    if Path(args.weights).exists():
        net.load_state_dict(torch.load(args.weights, map_location=device))
        print("Loaded pre-trained weights:", args.weights)

    train(net, dl, args.epochs, device, args.lr, args.wd, args.poswt)

    torch.save(net.state_dict(), args.out)
    print("\n✅ Saved fine-tuned weights →", args.out)
