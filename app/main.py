import io
import os
import uuid
import hmac
import zipfile
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from PIL import Image

from . import face_engine
from .storage import get_storage
from .index_store import FaceIndex

# ---------------------------------------------------------------------------
# Configuration (variables d'environnement)
# ---------------------------------------------------------------------------
EVENT_NAME = os.environ.get("EVENT_NAME", "Galerie de l'événement")
EVENT_SUBTITLE = os.environ.get("EVENT_SUBTITLE", "")
EVENT_LOCATION = os.environ.get("EVENT_LOCATION", "")
EVENT_PASSWORD = os.environ.get("EVENT_PASSWORD", "changeme")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme-admin")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-secret-change-me")
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.38"))
THUMB_MAX_SIDE = int(os.environ.get("THUMB_MAX_SIDE", "480"))
MAX_PHOTOS = int(os.environ.get("MAX_PHOTOS", "2000"))
PHOTOS_PAGE_SIZE = int(os.environ.get("PHOTOS_PAGE_SIZE", "24"))

BASE_DIR = os.path.dirname(__file__)

_FR_MONTHS = {
    1: "janv.", 2: "févr.", 3: "mars", 4: "avr.", 5: "mai", 6: "juin",
    7: "juil.", 8: "août", 9: "sept.", 10: "oct.", 11: "nov.", 12: "déc.",
}


def _fr_datetime(dt: datetime) -> str:
    return f"{dt.day:02d} {_FR_MONTHS[dt.month]} {dt.year} · {dt.hour:02d}h{dt.minute:02d}"


def _fr_date(dt: datetime) -> str:
    return f"{dt.day:02d} {_FR_MONTHS[dt.month]} {dt.year}"


app = FastAPI(title=EVENT_NAME)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

storage = get_storage()
face_index = FaceIndex()


@app.on_event("startup")
def on_startup():
    storage.sync_from_remote()
    data = storage.load_index_bytes()
    global face_index
    face_index = FaceIndex.from_bytes(data) if data else FaceIndex()
    # Réchauffe le modèle de reconnaissance faciale pour que la première
    # requête d'un invité ne soit pas ralentie par le chargement du modèle.
    try:
        face_engine.get_app()
    except Exception:
        pass


def _safe_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a or "", b or "")


def require_guest(request: Request):
    if not request.session.get("guest_ok"):
        raise HTTPException(status_code=303, headers={"Location": "/"})
    return True


def require_admin(request: Request):
    if not request.session.get("admin_ok"):
        raise HTTPException(status_code=303, headers={"Location": "/admin"})
    return True


@app.exception_handler(HTTPException)
async def redirect_handler(request: Request, exc: HTTPException):
    if exc.status_code == 303 and "Location" in (exc.headers or {}):
        return RedirectResponse(url=exc.headers["Location"], status_code=303)
    return await http_exception_handler(request, exc)


def _photo_list():
    """Liste des photos, plus récentes en premier."""
    return list(reversed(list(face_index.photos.values())))


def _photo_dict(entry) -> dict:
    return {"photo_id": entry.photo_id, "thumb_id": entry.thumb_id, "filename": entry.original_filename}


# ---------------------------------------------------------------------------
# Espace invité
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if request.session.get("guest_ok"):
        return RedirectResponse(url="/invite")
    return templates.TemplateResponse(
        request,
        "password.html",
        {
            "event_name": EVENT_NAME,
            "event_subtitle": EVENT_SUBTITLE,
            "event_location": EVENT_LOCATION,
            "error": None,
        },
    )


@app.post("/auth")
def auth(request: Request, password: str = Form(...)):
    if _safe_eq(password.strip(), EVENT_PASSWORD):
        request.session["guest_ok"] = True
        return RedirectResponse(url="/invite", status_code=303)
    return templates.TemplateResponse(
        request,
        "password.html",
        {
            "event_name": EVENT_NAME,
            "event_subtitle": EVENT_SUBTITLE,
            "event_location": EVENT_LOCATION,
            "error": "Mot de passe incorrect.",
        },
        status_code=401,
    )


@app.get("/invite", response_class=HTMLResponse)
def invite(request: Request, _=Depends(require_guest)):
    all_photos = _photo_list()
    page = [_photo_dict(p) for p in all_photos[:PHOTOS_PAGE_SIZE]]
    return templates.TemplateResponse(
        request,
        "invite.html",
        {
            "event_name": EVENT_NAME,
            "event_date": _fr_date(datetime.now()),
            "photos": page,
            "total_photos": len(all_photos),
            "page_size": PHOTOS_PAGE_SIZE,
            "has_more": len(all_photos) > PHOTOS_PAGE_SIZE,
        },
    )


@app.get("/api/photos")
def api_photos(request: Request, offset: int = 0, limit: int = PHOTOS_PAGE_SIZE, _=Depends(require_guest)):
    all_photos = _photo_list()
    offset = max(0, offset)
    limit = max(1, min(limit, 100))
    page = all_photos[offset : offset + limit]
    return JSONResponse(
        {
            "photos": [_photo_dict(p) for p in page],
            "has_more": offset + limit < len(all_photos),
            "total": len(all_photos),
        }
    )


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, selfie: UploadFile = File(...), _=Depends(require_guest)):
    data = await selfie.read()
    try:
        img = face_engine.read_image(data)
    except Exception:
        return templates.TemplateResponse(
            request,
            "results.html",
            {
                "event_name": EVENT_NAME,
                "error": "Image illisible, réessaie avec une autre photo.",
                "results": [],
                "total_photos": len(face_index.photos),
            },
        )

    faces = face_engine.detect_faces(img)
    face = face_engine.largest_face(faces)
    if face is None:
        return templates.TemplateResponse(
            request,
            "results.html",
            {
                "event_name": EVENT_NAME,
                "error": "Aucun visage détecté sur cette photo. Essaie avec un selfie net, bien "
                "éclairé, visage face à la caméra.",
                "results": [],
                "total_photos": len(face_index.photos),
            },
        )

    matches = face_index.search(face.normed_embedding, MATCH_THRESHOLD)
    results = []
    for photo_id, score in matches:
        entry = face_index.photos.get(photo_id)
        if entry:
            results.append(
                {
                    "photo_id": photo_id,
                    "thumb_id": entry.thumb_id,
                    "filename": entry.original_filename,
                    "score": round(score * 100),
                }
            )

    request.session["search_photo_ids"] = [r["photo_id"] for r in results]

    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "event_name": EVENT_NAME,
            "error": None,
            "results": results,
            "total_photos": len(face_index.photos),
        },
    )


@app.get("/download/zip")
def download_zip(request: Request, _=Depends(require_guest)):
    photo_ids = request.session.get("search_photo_ids") or []
    if not photo_ids:
        raise HTTPException(status_code=404, detail="Aucun résultat de recherche à télécharger.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for photo_id in photo_ids:
            path = storage.get_photo_path(photo_id)
            if path.exists():
                entry = face_index.photos.get(photo_id)
                arcname = entry.original_filename if entry else photo_id
                zf.write(path, arcname=arcname)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=mes-photos.zip"},
    )


@app.get("/photo/{photo_id}")
def get_photo(photo_id: str, request: Request, _=Depends(require_guest)):
    path = storage.get_photo_path(photo_id)
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(path))


@app.get("/thumb/{thumb_id}")
def get_thumb(thumb_id: str, request: Request, _=Depends(require_guest)):
    path = storage.get_thumb_path(thumb_id)
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(path))


# ---------------------------------------------------------------------------
# Espace organisateur (admin)
# ---------------------------------------------------------------------------
@app.get("/admin", response_class=HTMLResponse)
def admin_home(request: Request):
    if request.session.get("admin_ok"):
        return RedirectResponse(url="/admin/upload")
    return templates.TemplateResponse(request, "admin_password.html", {"error": None})


@app.post("/admin/auth")
def admin_auth(request: Request, password: str = Form(...)):
    if _safe_eq(password.strip(), ADMIN_PASSWORD):
        request.session["admin_ok"] = True
        return RedirectResponse(url="/admin/upload", status_code=303)
    return templates.TemplateResponse(
        request, "admin_password.html", {"error": "Mot de passe incorrect."}, status_code=401
    )


@app.get("/admin/upload", response_class=HTMLResponse)
def admin_upload_page(request: Request, _=Depends(require_admin)):
    stats = face_index.stats()
    return templates.TemplateResponse(
        request,
        "admin_upload.html",
        {
            "event_name": EVENT_NAME,
            "stats": stats,
            "max_photos": MAX_PHOTOS,
            "batches": face_index.recent_batches(5),
        },
    )


def _make_thumb_bytes(img_bgr) -> bytes:
    h, w = img_bgr.shape[:2]
    scale = THUMB_MAX_SIDE / max(h, w)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


@app.post("/admin/upload")
async def admin_upload(request: Request, files: list[UploadFile] = File(...), _=Depends(require_admin)):
    if len(face_index.photos) + len(files) > MAX_PHOTOS:
        raise HTTPException(status_code=400, detail=f"Limite de {MAX_PHOTOS} photos dépassée.")

    processed, skipped, faces_in_batch = 0, 0, 0
    for f in files:
        raw = await f.read()
        try:
            img = face_engine.read_image(raw)
        except Exception:
            skipped += 1
            continue

        ext = os.path.splitext(f.filename or "")[1].lower() or ".jpg"
        photo_id = f"{uuid.uuid4().hex}{ext}"
        thumb_id = f"{uuid.uuid4().hex}.jpg"

        storage.save_photo(photo_id, raw)
        storage.save_thumb(thumb_id, _make_thumb_bytes(img.copy()))

        faces = face_engine.detect_faces(img)
        embeddings = [fc.normed_embedding for fc in faces]
        bboxes = [face_engine.face_box_int(fc) for fc in faces]

        face_index.add_photo(photo_id, f.filename or photo_id, thumb_id, embeddings, bboxes)
        processed += 1
        faces_in_batch += len(embeddings)

    if processed:
        face_index.add_batch(
            batch_id=uuid.uuid4().hex,
            label=f"Lot du {_fr_datetime(datetime.now())}",
            n_photos=processed,
            n_faces=faces_in_batch,
        )

    storage.save_index_bytes(face_index.to_bytes())

    stats = face_index.stats()
    return JSONResponse(
        {
            "processed": processed,
            "skipped": skipped,
            "stats": stats,
            "batches": [b.__dict__ for b in face_index.recent_batches(5)],
        }
    )


@app.get("/admin/stats")
def admin_stats(_=Depends(require_admin)):
    return face_index.stats()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "storage": storage.backend_name, **face_index.stats()}
