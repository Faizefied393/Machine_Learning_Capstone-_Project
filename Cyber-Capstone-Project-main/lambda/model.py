import os, torch, boto3
from torchvision import transforms
from PIL import Image
from io import BytesIO
import timm                                               # NEW
WEIGHTS = "/opt/weights/effnet_b0_imagenet.pt"

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()
# ─── env + S3 client ─────────────────────────────────────
THRESH = float(os.getenv("PRED_THRESHOLD", "0.50"))
BUCKET = os.getenv("S3_BUCKET")
REGION = os.getenv("AWS_REGION", "us-east-2")
s3     = boto3.client("s3", region_name=REGION)





# ─── transforms must match EffNet-B0 training ────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

# ─── prediction helper ───────────────────────────────────
@torch.no_grad()
def predict(s3_key: str) -> str | None:
    if s3_key.endswith(".DS_Store"):
        return None
    try:
        body = s3.get_object(Bucket=BUCKET, Key=s3_key)["Body"]
        img  = Image.open(BytesIO(body.read())).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        prob   = torch.sigmoid(model(tensor)).item()
        return "has cranes" if prob >= THRESH else "does not have cranes"
    except Exception as e:
        print("Error on", s3_key, e)
        return None



