import boto3, io, zipfile, os, json, pandas as pd, tempfile
from model import predict

s3     = boto3.client("s3")
BUCKET = os.environ["S3_BUCKET"]

def handler(event, context):
    job_id = event["job_id"]
    manifest_key = event["manifest"]

    # ----- read manifest -----
    keys = json.loads(
        s3.get_object(Bucket=BUCKET, Key=manifest_key)["Body"].read()
    )

    preds = [predict(k) for k in keys]

    # ----- write Excel to an in-memory buffer (tiny) -----
    xls_buf = io.BytesIO()
    pd.DataFrame({"file": keys, "prediction": preds}).to_excel(xls_buf, index=False)
    xls_buf.seek(0)

    # ----- stream a ZIP directly to S3 -----
    out_key = f"results/{job_id}-preds.zip"
    with tempfile.NamedTemporaryFile() as tmp:
        # ➊ create a ZipFile that writes to the tempfile
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("predictions.xlsx", xls_buf.getvalue())

            # add only positives, one at a time (keeps /tmp small)
            for k, p in zip(keys, preds):
                if p == "has cranes":
                    obj = s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()
                    z.writestr(f"positives/{os.path.basename(k)}", obj)

        # ➋ rewind and upload the finished file
        tmp.seek(0)
        s3.upload_fileobj(tmp, BUCKET, out_key)

    return {"job_id": job_id, "result_key": out_key}



