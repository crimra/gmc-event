"""
Moteur de reconnaissance faciale.

Utilise insightface (modèle "buffalo_sc", léger, tourne bien en CPU) pour :
  - détecter les visages dans une image
  - calculer un "embedding" (vecteur) par visage, normalisé
  - comparer deux embeddings par similarité cosinus

Testé : deux photos différentes de la même personne donnent un score ~0.6-0.8,
deux personnes différentes donnent un score proche de 0 (voire négatif).
Seuil par défaut : 0.38 (ajustable via la variable d'env MATCH_THRESHOLD).
"""
import os
import threading

import cv2
import numpy as np

_MODEL_NAME = os.environ.get("FACE_MODEL", "buffalo_sc")
_DET_SIZE = int(os.environ.get("FACE_DET_SIZE", "640"))

_app = None
_lock = threading.Lock()


def get_app():
    """Charge le modèle une seule fois (paresseux, thread-safe)."""
    global _app
    if _app is None:
        with _lock:
            if _app is None:
                from insightface.app import FaceAnalysis

                fa = FaceAnalysis(name=_MODEL_NAME)
                fa.prepare(ctx_id=-1, det_size=(_DET_SIZE, _DET_SIZE))
                _app = fa
    return _app


def read_image(path_or_bytes):
    """Charge une image depuis un chemin ou des bytes en tableau BGR (format OpenCV)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes))
    if img is None:
        raise ValueError("Image illisible ou format non supporté")
    return img


def detect_faces(image_bgr):
    """
    Retourne la liste des visages détectés dans l'image.
    Chaque élément a .bbox (x1,y1,x2,y2), .det_score, .normed_embedding (vecteur 512-d normalisé).
    """
    app = get_app()
    return app.get(image_bgr)


def largest_face(faces):
    """Utile pour un selfie : on garde le plus grand visage (le plus proche de la caméra)."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def face_box_int(face):
    x1, y1, x2, y2 = face.bbox
    return int(max(0, x1)), int(max(0, y1)), int(x2), int(y2)
