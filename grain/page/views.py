import io
import os
import mimetypes
from importlib.metadata import version
from flask import Blueprint, render_template, Response
from PIL import Image, ImageOps
from config import settings, storage

page = Blueprint("page", __name__, template_folder="templates")

MAX_WIDTH = 200
MAX_HEIGHT = 200


@page.get("/image/<path:filename>")
def serve_image(filename):
    client = storage.get_minio_client()
    response = client.get_object(settings.MINIO_BUCKET, filename)
    data = response.read()

    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
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
        filename = f"dataset/{uuid.uuid4()}{ext}"

        client = storage.get_minio_client()

        client.put_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=filename,
            data=file,
            length=-1,
            part_size=5 * 1024 * 1024,
            metadata={"text": text},
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
    for obj in objects:
        stat = client.stat_object(settings.MINIO_BUCKET, obj.object_name)
        items.append(
            {
                "name": obj.object_name,
                "last_modified": obj.last_modified,
            }
        )

    items.sort(key=lambda x: x["last_modified"], reverse=True)

    context = {"items": items, "active_page": "dashboard"}
    return render_template("page/home.html", **context)
