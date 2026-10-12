from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

detectRoute = APIRouter(prefix="/api/v1", tags=["api_v1"])

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "best.pt"
)
model = YOLO(MODEL_PATH)

ALLOWED = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024


# endpoint عادي (مش async) عشان الـ inference ما يقفلش الـ event loop
@detectRoute.post("/detect")
def detect(file: UploadFile = File(...), conf: float = 0.25):
    if file.content_type not in ALLOWED:
        raise HTTPException(415, "Please upload a JPG, PNG or WEBP image.")

    data = file.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is too large (max 10 MB).")

    try:
        import io
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the image.")

    # result = model.predict(img, conf=conf, verbose=False)[0]
    result = model.predict(img, conf=conf, device="cpu", verbose=False)[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxyn[0].tolist()  # قيم من 0 لـ 1
        cls_id = int(box.cls[0])
        detections.append(
            {
                "class_id": cls_id,
                "class_name": result.names[cls_id],
                "confidence": round(float(box.conf[0]), 4),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )
    detections.sort(key=lambda d: -d["confidence"])

    return {"image_width": img.width, "image_height": img.height, "detections": detections}