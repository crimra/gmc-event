"""
Couche de stockage : abstrait où vivent les photos et l'index des visages.

- LocalStorage : tout sur disque local (./data). Pratique en dev, mais NE
  SURVIT PAS à un redémarrage sur un hébergement gratuit type Hugging Face
  Space (disque éphémère).

- HFDatasetStorage : les photos originales + l'index sont poussés dans un
  dépôt "Dataset" Hugging Face PRIVÉ (gratuit, persistant). Au démarrage,
  l'index est téléchargé une fois ; les photos sont mises en cache localement
  à la demande.

  Important : les dépôts Hugging Face limitent les commits à 128/heure sur le
  plan gratuit. Un commit par fichier (photo, miniature, index...) épuise vite
  cette limite dès qu'on envoie plusieurs photos. `commit_uploads` et
  `commit_delete` regroupent donc tous les fichiers d'un même lot dans UN SEUL
  commit, quel que soit le nombre de photos.

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

    def commit_uploads(self, photo_filenames: list, thumb_filenames: list, index_bytes: bytes):
        """Les fichiers sont déjà sur disque (save_photo/save_thumb) ; il ne
        reste qu'à sauvegarder l'index à jour."""
        INDEX_PATH.write_bytes(index_bytes)

    def commit_delete(self, photo_filename: str, thumb_filename: str, index_bytes: bytes):
        for path in (PHOTOS_DIR / photo_filename, THUMBS_DIR / thumb_filename):
            if path.exists():
                path.unlink()
        INDEX_PATH.write_bytes(index_bytes)

    def load_index_bytes(self):
        if INDEX_PATH.exists():
            return INDEX_PATH.read_bytes()
        return None

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
    Le cache local (./data) sert de tampon rapide ; save_photo/save_thumb
    n'écrivent que localement, et c'est commit_uploads/commit_delete qui
    poussent tout le lot vers le Hub en un seul commit.
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

    def save_photo(self, filename: str, data: bytes) -> str:
        path = PHOTOS_DIR / filename
        path.write_bytes(data)
        return str(path)

    def save_thumb(self, filename: str, data: bytes) -> str:
        path = THUMBS_DIR / filename
        path.write_bytes(data)
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

    def _commit(self, adds: dict, deletes: list, message: str):
        """adds: {repo_path: local_path}. deletes: [repo_path, ...].
        Un seul commit pour tout, pour rester sous la limite de commits/heure."""
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path))
            for repo_path, local_path in adds.items()
        ]
        operations += [CommitOperationDelete(path_in_repo=repo_path) for repo_path in deletes]
        if not operations:
            return
        try:
            self.api.create_commit(
                repo_id=self.repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=message,
            )
        except Exception as e:
            # Best-effort : les fichiers restent disponibles localement sur
            # cette instance même si la synchro distante échoue (ex: quota de
            # commits temporairement dépassé).
            print(f"[storage] échec du commit Hugging Face ({message}): {e}")

    def commit_uploads(self, photo_filenames: list, thumb_filenames: list, index_bytes: bytes):
        INDEX_PATH.write_bytes(index_bytes)
        adds = {f"photos/{fn}": PHOTOS_DIR / fn for fn in photo_filenames}
        adds.update({f"thumbs/{fn}": THUMBS_DIR / fn for fn in thumb_filenames})
        adds["index.pkl"] = INDEX_PATH
        self._commit(adds, [], f"Ajout de {len(photo_filenames)} photo(s)")

    def commit_delete(self, photo_filename: str, thumb_filename: str, index_bytes: bytes):
        INDEX_PATH.write_bytes(index_bytes)
        for path in (PHOTOS_DIR / photo_filename, THUMBS_DIR / thumb_filename):
            if path.exists():
                path.unlink()
        self._commit(
            {"index.pkl": INDEX_PATH},
            [f"photos/{photo_filename}", f"thumbs/{thumb_filename}"],
            "Suppression d'une photo",
        )

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

    def sync_from_remote(self):
        """Au démarrage : rapatrie l'index (léger) depuis le Hub. Les photos
        sont récupérées à la demande (evite de tout retélécharger)."""
        self.load_index_bytes()


def get_storage():
    backend = os.environ.get("STORAGE_BACKEND", "local")
    if backend == "hf_dataset":
        return HFDatasetStorage()
    return LocalStorage()
