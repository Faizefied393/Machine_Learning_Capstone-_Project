FROM public.ecr.aws/lambda/python:3.10

RUN pip install --no-cache-dir numpy==1.26.4

RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 \
        torchvision==0.17.2 \
        timm==0.9.12 \
        pandas==2.2.3 \
        openpyxl \
        boto3 \
        XlsxWriter==3.2.0

RUN python - <<'PY'
import timm, torch, os
m = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
os.makedirs('/opt/weights', exist_ok=True)
torch.save(m.state_dict(), '/opt/weights/effnet_b0_imagenet.pt')
PY

COPY app.py model.py ./

CMD ["app.handler"]


