# ── app.py  (deployed on Render) ──────────────────────────────────────────
import os, json, uuid
from io import BytesIO
from zipfile import ZipFile
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, Response, stream_with_context
)
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

# ── env & clients ────────────────────────────────────────────────────────
load_dotenv()

AWS_REGION  = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET   = os.getenv("S3_BUCKET")                    # icf-whooping-crane-uploads
LAMBDA_ARN  = os.getenv("LAMBDA_ARN")                   # arn:aws:lambda:…:crane-inference

s3  = boto3.client("s3",  region_name=AWS_REGION)
lam = boto3.client("lambda", region_name=AWS_REGION)

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ── Auth page (kept as-is) ───────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if (request.form.get("username"), request.form.get("password")) == (
            "RDKirkwood", "WHCR25@Projects"
        ):
            return redirect(url_for("upload_files"))
        flash("Invalid username or password. Please try again.")
        return redirect(request.url)
    return render_template("login.html")


# ── Landing page ─────────────────────────────────────────────────────────
@app.route("/upload")
def upload_files():
    return render_template("LandingPage.html")


# ── 1 · browser asks for pre-signed PUT URLs ─────────────────────────────
@app.route("/generate-presigned-urls", methods=["POST"])
def generate_presigned_urls():
    filenames = request.get_json().get("filenames", [])

    def gen():
        yield "["
        first = True
        for key in filenames:
            url = s3.generate_presigned_url(
                "put_object",
                Params={"Bucket": S3_BUCKET, "Key": key},
                ExpiresIn=3600,
                HttpMethod="PUT",
            )
            if not first:
                yield ","
            first = False
            yield json.dumps({"filename": key, "url": url})
        yield "]"

    return Response(stream_with_context(gen()),
                    mimetype="application/json")


# ── 2 · browser tells us to start the Lambda job ─────────────────────────
@app.route("/start-job", methods=["POST"])
def start_job():
    data = request.get_json()

    # BEFORE ─ would crash because the key no longer exists
    # keys = data["filenames"]

    # AFTER ─ matches what the new JS sends
    keys = data["keys"]

    job_id = str(uuid.uuid4())
    manifest_key = f"manifests/{job_id}.json"

    # upload manifest & invoke Lambda exactly as before …
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=manifest_key,
        Body=json.dumps(keys).encode("utf-8")
    )

    lam.invoke(
        FunctionName=LAMBDA_ARN,      # keep whatever constant you defined
        InvocationType="Event",
        Payload=json.dumps({"manifest": manifest_key,
                            "job_id"  : job_id})
    )

    return jsonify({"job_id": job_id}), 202



# ── 3 · browser polls until ZIP appears ──────────────────────────────────
@app.route("/job-status/<job_id>")
def job_status(job_id):
    key = f"results/{job_id}-preds.zip"
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=3600
        )
        return jsonify({"status": "done", "download_url": url})
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "403", "NoSuchKey"):
            return jsonify({"status": "pending"}), 202
        raise


# ── gunicorn entrypoint ──────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
