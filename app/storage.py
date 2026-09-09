"""
Couche de stockage : abstrait où vivent les photos et l'index des visages.

- LocalStorage : tout sur disque local (./data). Pratique en dev, mais NE
  SURVIT PAS à un redémarrage sur un hébergement gratuit type Hugging Face
  Space (disque éphémère).

- HFDatasetStorage : les photos originales + l'index sont poussés dans un
  dépôt "Dataset" Hugging Face PRIVÉ (gratuit, persistant). Au démarrage,
  l'index est téléchargé une fois ; les photos sont mises en cache localement
  à la demande. Chaque upload/màj de l'index est repoussé immédiatement vers
  le Hub -> les données survivent aux redémarrages / mises en veille du Space.

Le choix se fait via la variable d'env STORAGE_BACKEND=local|hf_dataset.
"""
import os
import shutil
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()
PHOTOS_DIR = DATA_DIR / "photos"
THUMBS_DIR = DATA_DIR / "thumbs"
INDEX_PATH = DATA_DIR / "index.pkl"

PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
THUMBS_DIR.mkdir(parents=True, exist_ok=True)


class LocalStorage:
    """Stockage 100% local (dev / auto-hébergement sur une machine qui reste allumée)."""

    backend_name = "local"

    def save_photo(self, filename: str, data: bytes) -> str:
        path = PHOTOS_DIR / filename
        path.write_bytes(data)
        return str(path)

    def save_thumb(self, filename: str, data: bytes) -> str:
        path = THUMBS_DIR / filename
        path.write_bytes(data)
        return str(path)

    def get_photo_path(self, filename: str) -> Path:
        return PHOTOS_DIR / filename

    def get_thumb_path(self, filename: str) -> Path:
        return THUMBS_DIR / filename

    def load_index_bytes(self):
        if INDEX_PATH.exists():
            return INDEX_PATH.read_bytes()
        return None

    def save_index_bytes(self, data: bytes):
        INDEX_PATH.write_bytes(data)

    def sync_from_remote(self):
        """No-op en local : rien à télécharger."""
        pass


class HFDatasetStorage:
    """
    Stockage persistant gratuit via un dépôt Hugging Face 'dataset' privé.

    Variables d'environnement requises :
      HF_TOKEN            token Hugging Face avec droit d'écriture (secret du Space)
      HF_DATASET_REPO      ex: "monpseudo/mon-evenement-galerie"

    Les photos originales et l'index (index.pkl) sont stockés dans ce dépôt.
    Le cache local (./data) sert de tampon rapide ; à chaque écriture, le
    fichier est aussi repoussé vers le Hub pour rester durable.
    """

    backend_name = "hf_dataset"

    def __init__(self):
        from huggingface_hub import HfApi

        self.token = os.environ["HF_TOKEN"]
        self.repo_id = os.environ["HF_DATASET_REPO"]
        self.api = HfApi(token=self.token)
        self.api.create_repo(
            repo_id=self.repo_id, repo_type="dataset", private=True, exist_ok=True
        )

    def _upload(self, local_path: Path, repo_path: str):
        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            repo_type="dataset",
        )

    def save_photo(self, filename: str, data: bytes) -> str:
        path = PHOTOS_DIR / filename
        path.write_bytes(data)
        self._upload(path, f"photos/{filename}")
        return str(path)

    def save_thumb(self, filename: str, data: bytes) -> str:
        path = THUMBS_DIR / filename
        path.write_bytes(data)
        self._upload(path, f"thumbs/{filename}")
        return str(path)

    def get_photo_path(self, filename: str) -> Path:
        local = PHOTOS_DIR / filename
        if not local.exists():
            self._download_to(f"photos/{filename}", local)
        return local

    def get_thumb_path(self, filename: str) -> Path:
        local = THUMBS_DIR / filename
        if not local.exists():
            self._download_to(f"thumbs/{filename}", local)
        return local

    def _download_to(self, repo_path: str, local_path: Path):
        from huggingface_hub import hf_hub_download

        try:
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=repo_path,
                token=self.token,
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(downloaded, local_path)
        except Exception:
            pass

    def load_index_bytes(self):
        self._download_to("index.pkl", INDEX_PATH)
        if INDEX_PATH.exists():
            return INDEX_PATH.read_bytes()
        return None

    def save_index_bytes(self, data: bytes):
        INDEX_PATH.write_bytes(data)
        self._upload(INDEX_PATH, "index.pkl")

    def sync_from_remote(self):
        """Au démarrage : rapatrie l'index (léger) depuis le Hub. Les photos
        sont récupérées à la demande (evite de tout retélécharger)."""
        self.load_index_bytes()


def get_storage():
    backend = os.environ.get("STORAGE_BACKEND", "local")
    if backend == "hf_dataset":
        return HFDatasetStorage()
    return LocalStorage()
