# ──────────────────────────────────────────────────────────────────────────────
#  evaluate.py – run the fine-tuned CNN on the labelled dataset
# ──────────────────────────────────────────────────────────────────────────────
import argparse, csv
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from train import CNN, DEF_IMG_SIZE     # reuse model + img size from train.py

# ───────────────  dataset (no augmentations)  ───────────────
class OrigDataset(Dataset):
    _EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff",
             ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"}

    def __init__(self, img_dir: Path, xls_path: Path):
        df = pd.read_excel(xls_path)
        img_lookup = {p.stem.upper(): p for p in img_dir.iterdir()
                      if p.suffix in self._EXTS}

        self.samples = []
        missing     = []
        for pict, cnt in zip(df["PICT Num"], df["WHCR Cnt"]):
            raw  = str(pict).strip()
            stem = Path(raw).stem.upper()

            # exact file (with extension) ?
            direct = img_dir / raw
            if direct.exists():
                self.samples.append((direct, 1 if cnt > 0 else 0))
            elif stem in img_lookup:
                self.samples.append((img_lookup[stem], 1 if cnt > 0 else 0))
            else:
                missing.append(raw)

        if missing:
            print(f"⚠️  {len(missing)} rows had no matching image.")

        self.tfm = transforms.Compose([
            transforms.Resize(DEF_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        fp, label = self.samples[idx]
        img = self.tfm(Image.open(fp).convert("RGB"))
        return img, torch.tensor(label, dtype=torch.float32), fp.name

# ───────────────  evaluation  ───────────────
def evaluate(net, loader, device, thresh=0.5):
    net.eval().to(device)
    TP=TN=FP=FN=0
    rows = []

    with torch.no_grad():
        for x, y, names in loader:
            x, y = x.to(device), y.to(device)
            logits = net(x).squeeze(1)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= thresh).float()

            for name, gt, pr, pb in zip(names, y.cpu(), preds.cpu(), probs.cpu()):
                gt, pr = int(gt.item()), int(pr.item())
                rows.append([name, gt, round(pb.item(),3), pr])
                if   gt==1 and pr==1: TP+=1
                elif gt==0 and pr==0: TN+=1
                elif gt==0 and pr==1: FP+=1
                else:                 FN+=1

    acc = (TP+TN)/(TP+TN+FP+FN)
    return acc, (TP,FP,FN,TN), rows

# ───────────────  main  ───────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="data/images")
    p.add_argument("--labels", default="data/labels.xlsx")
    p.add_argument("--weights",default="cnn_model_finetuned.pt")
    p.add_argument("--bs",     type=int, default=64)
    p.add_argument("--thresh", type=float, default=0.50)
    p.add_argument("--out",    default="eval_results.csv")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device :", device)

    ds = OrigDataset(Path(args.data), Path(args.labels))
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0)

    net = CNN()
    net.load_state_dict(torch.load(args.weights, map_location=device))
    print("Loaded weights:", args.weights)

    acc, (TP,FP,FN,TN), rows = evaluate(net, dl, device, args.thresh)

    # save CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "ground_truth", "probability", "prediction"])
        writer.writerows(rows)
    print(f"Results written → {args.out}")

    # summary
    print("\nConfusion matrix (threshold =", args.thresh, ")")
    print(f"        Pred 1   Pred 0")
    print(f"Actual 1   {TP:4d}      {FN:4d}")
    print(f"Actual 0   {FP:4d}      {TN:4d}")
    print(f"\nAccuracy : {acc*100:.2f}%  on {len(ds)} images")
