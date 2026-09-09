"""
Index des visages : associe à chaque visage détecté (dans les photos
uploadées par l'organisateur) un embedding + l'identifiant de la photo dont
il provient. C'est cet index qui est comparé au selfie de l'invité.

Stocké comme un simple pickle (liste de dicts) — largement suffisant pour
< 2000 photos (quelques milliers de visages, ~quelques dizaines de Mo max).
"""
import pickle
import threading
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class FaceEntry:
    photo_id: str          # nom de fichier de la photo dans le storage
    embedding: np.ndarray   # vecteur 512-d normalisé
    bbox: tuple = None      # position du visage dans la photo (debug/UI)


@dataclass
class PhotoEntry:
    photo_id: str
    original_filename: str
    thumb_id: str
    n_faces: int = 0


@dataclass
class BatchEntry:
    batch_id: str
    label: str              # ex: "Lot du 06 sept. 2026 · 14h12"
    n_photos: int
    n_faces: int
    status: str = "Indexé"


class FaceIndex:
    def __init__(self):
        self.faces: List[FaceEntry] = []
        self.photos: dict[str, PhotoEntry] = {}
        self.batches: List[BatchEntry] = []
        self._lock = threading.Lock()

    def add_photo(self, photo_id, original_filename, thumb_id, face_embeddings, bboxes):
        with self._lock:
            self.photos[photo_id] = PhotoEntry(
                photo_id=photo_id,
                original_filename=original_filename,
                thumb_id=thumb_id,
                n_faces=len(face_embeddings),
            )
            for emb, bbox in zip(face_embeddings, bboxes):
                self.faces.append(FaceEntry(photo_id=photo_id, embedding=emb, bbox=bbox))

    def add_batch(self, batch_id: str, label: str, n_photos: int, n_faces: int):
        with self._lock:
            self.batches.append(
                BatchEntry(batch_id=batch_id, label=label, n_photos=n_photos, n_faces=n_faces)
            )

    def recent_batches(self, limit: int = 5) -> list:
        return list(reversed(self.batches[-limit:]))

    def photo_exists(self, photo_id) -> bool:
        return photo_id in self.photos

    def search(self, query_embedding: np.ndarray, threshold: float, top_k: int = 200):
        """Retourne les photos dont au moins un visage dépasse le seuil de similarité,
        triées par meilleur score décroissant (une entrée par photo, pas par visage)."""
        if not self.faces:
            return []
        best_per_photo = {}
        for entry in self.faces:
            score = float(np.dot(query_embedding, entry.embedding))
            if score < threshold:
                continue
            if entry.photo_id not in best_per_photo or score > best_per_photo[entry.photo_id]:
                best_per_photo[entry.photo_id] = score
        results = [
            (photo_id, score) for photo_id, score in best_per_photo.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def stats(self):
        return {"n_photos": len(self.photos), "n_faces": len(self.faces)}

    def to_bytes(self) -> bytes:
        with self._lock:
            return pickle.dumps({"faces": self.faces, "photos": self.photos, "batches": self.batches})

    @classmethod
    def from_bytes(cls, data: bytes) -> "FaceIndex":
        idx = cls()
        if data:
            obj = pickle.loads(data)
            idx.faces = obj.get("faces", [])
            idx.photos = obj.get("photos", {})
            idx.batches = obj.get("batches", [])
        return idx
