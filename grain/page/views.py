import io
import os
import mimetypes
import uuid
from importlib.metadata import version
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    Response,
    request,
    url_for,
)
from PIL import Image, ImageOps
from config import settings, storage

page = Blueprint("page", __name__, template_folder="templates")

MAX_WIDTH = 600
MAX_HEIGHT = 600


@page.get("/image/<path:filename>")
def serve_image(filename):
    client = storage.get_minio_client()
    response = client.get_object(settings.MINIO_BUCKET, filename)
    data = response.read()

    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)

    output = io.BytesIO()
    img.save(output, format=img.format or "JPEG")
    output.seek(0)

    return Response(
        output.read(), mimetype=img.format and f"image/{img.format.lower()}"
    )


@page.post("/upload")
def upload():
    if "image" not in request.files:
        flash("No image provided")
        return redirect(url_for("page.home"))

    file = request.files["image"]
    text = request.form.get("text", "")

    if file.filename == "":
        flash("No image selected")
        return redirect(url_for("page.home"))

    if file:
        ext = os.path.splitext(file.filename)[1].lower()
        base_name = f"dataset/{uuid.uuid4()}"
        filename = f"{base_name}{ext}"

        client = storage.get_minio_client()

        client.put_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=filename,
            data=file,
            length=-1,
            part_size=5 * 1024 * 1024,
        )

        client.put_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=f"{base_name}.txt",
            data=io.BytesIO(text.encode("utf-8")),
            length=len(text.encode("utf-8")),
        )

        flash("Upload successful")

    return redirect(url_for("page.home"))


@page.get("/upload")
def upload_image():
    return render_template("page/upload.html", active_page="upload")


@page.get("/")
def home():
    client = storage.get_minio_client()

    items = []
    objects = client.list_objects(bucket_name=settings.MINIO_BUCKET, prefix="dataset/")
    seen_bases = set()
    for obj in objects:
        if obj.object_name.endswith(".txt"):
            continue
        base_name = obj.object_name.rsplit(".", 1)[0]
        if base_name in seen_bases:
            continue
        seen_bases.add(base_name)

        try:
            txt_response = client.get_object(settings.MINIO_BUCKET, f"{base_name}.txt")
            text = txt_response.read().decode("utf-8")
        except:
            text = ""

        items.append(
            {
                "name": obj.object_name,
                "text": text,
                "last_modified": obj.last_modified,
            }
        )

    items.sort(key=lambda x: x["last_modified"], reverse=True)

    context = {"items": items, "active_page": "dashboard"}
    return render_template("page/home.html", **context)


@page.get("/predict")
def predict():
    return render_template("page/predict.html", active_page="predict", results=None)


@page.post("/predict")
def predict_post():
    if "image" not in request.files:
        flash("No image provided")
        return redirect(url_for("page.predict"))

    file = request.files["image"]
    if file.filename == "":
        flash("No image selected")
        return redirect(url_for("page.predict"))

    try:
        from grain.medical import Medical
    except ImportError as e:
        flash(f"Failed to import Medical module: {e}")
        return redirect(url_for("page.predict"))

    file.seek(0)
    file_content = file.read()
    query_img = Image.open(io.BytesIO(file_content))
    query_img = query_img.convert("RGB")
    query_path = f"/tmp/{uuid.uuid4()}_query.jpg"
    query_img.save(query_path)

    results = settings.get_medical().query_diagnosis(query_path, top_k=3)

    os.remove(query_path)

    client = storage.get_minio_client()
    query_filename = f"predict/{uuid.uuid4()}.jpg"
    client.put_object(
        bucket_name=settings.MINIO_BUCKET,
        object_name=query_filename,
        data=io.BytesIO(file_content),
        length=len(file_content),
    )

    return render_template(
        "page/predict.html",
        active_page="predict",
        results=results,
        query_image=query_filename,
    )
