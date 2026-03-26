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


@page.get("/upload")
def upload_image():
    return render_template("page/upload.html")


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

    context = {"items": items}
    return render_template("page/home.html", **context)
