---
title: Galerie Événement
emoji: 📸
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# Galerie photo par reconnaissance faciale — 0€

Une petite application qui permet à chaque invité d'un événement de retrouver
**automatiquement toutes les photos où il apparaît**, en envoyant simplement
un selfie — comme Google Photos, mais pour un événement précis.

- Accès invités protégé par **un mot de passe général** (celui de l'événement)
- L'invité envoie un **selfie** → l'appli retrouve toutes les photos où son
  visage apparaît, par similarité faciale
- Espace organisateur séparé (mot de passe admin) pour **uploader les photos
  en masse**
- Jusqu'à ~2000 photos, **100% gratuit** en hébergement

Testé de bout en bout (upload → détection de visages → recherche par selfie
→ résultats) avant livraison.

---

## Comment ça marche (en bref)

1. L'organisateur se connecte sur `/admin` avec le mot de passe admin et
   uploade toutes les photos de l'événement. Pour chaque photo, l'appli
   détecte les visages et calcule un "embedding" (une empreinte numérique du
   visage) pour chacun.
2. Un invité arrive sur la page d'accueil, entre le mot de passe de
   l'événement, puis envoie un selfie.
3. L'appli calcule l'empreinte du visage du selfie et la compare à toutes les
   empreintes déjà enregistrées → elle renvoie les photos où la similarité
   dépasse un certain seuil.

Aucune photo n'est publique : il faut le mot de passe de l'événement pour
tout voir, et le mot de passe admin pour en ajouter.

---

## 1. Tester en local (optionnel)

```bash
pip install -r requirements.txt
cp .env.example .env   # puis modifie les mots de passe dedans
export $(cat .env | grep -v '^#' | xargs)   # charge les variables
uvicorn app.main:app --reload
```

Ouvre `http://localhost:8000`. Avec `STORAGE_BACKEND=local`, tout reste sur
ton disque dans `./data`.

## 2. Déployer gratuitement en ligne (Hugging Face Spaces)

C'est l'option recommandée : hébergement gratuit, l'appli reste accessible
24h/24, et les photos sont sauvegardées durablement dans un dépôt privé.

### Étape 1 — Créer un compte Hugging Face (gratuit)

Sur https://huggingface.co/join — juste un email.

### Étape 2 — Créer un token d'accès

Dans ton profil → **Settings → Access Tokens → New token**, rôle **Write**.
Copie ce token, tu en auras besoin (à ne jamais partager publiquement).

### Étape 3 — Créer un "Space"

Sur https://huggingface.co/new-space :
- **SDK** : Docker
- **Visibilité** : Private (recommandé — l'appli elle-même gère déjà l'accès
  par mot de passe, mais autant limiter qui peut voir le Space lui-même)
- **Hardware** : gratuit (CPU basic) — largement suffisant pour ce volume de
  photos

### Étape 4 — Envoyer le code dans le Space

Le Space te donne une URL de dépôt git (`https://huggingface.co/spaces/ton-pseudo/nom-du-space`).

```bash
cd event-gallery
git init
git remote add space https://huggingface.co/spaces/TON-PSEUDO/NOM-DU-SPACE
git add .
git commit -m "Galerie événement"
git push space main
```

(Identifiants demandés : ton nom d'utilisateur Hugging Face + le token créé à
l'étape 2 comme mot de passe.)

### Étape 5 — Configurer les secrets du Space

Dans le Space → **Settings → Variables and secrets**, ajoute (comme
**secrets**, pas variables publiques) :

| Nom | Valeur |
|---|---|
| `EVENT_NAME` | Le nom de ton événement |
| `EVENT_SUBTITLE` | Sous-titre affiché sous le nom (facultatif) |
| `EVENT_LOCATION` | Lieu affiché en bas de l'écran d'accueil (facultatif) |
| `EVENT_PASSWORD` | Mot de passe que tu donneras aux invités |
| `ADMIN_PASSWORD` | Ton mot de passe organisateur (garde-le pour toi) |
| `SESSION_SECRET` | Une longue chaîne aléatoire (ex. générée sur https://randomkeygen.com) |
| `STORAGE_BACKEND` | `hf_dataset` |
| `HF_TOKEN` | Le token créé à l'étape 2 |
| `HF_DATASET_REPO` | ex. `TON-PSEUDO/mon-evenement-photos` (créé automatiquement) |

Le Space redémarre automatiquement après ajout des secrets. Après quelques
minutes de build (installation des dépendances), l'appli est en ligne à
l'adresse `https://TON-PSEUDO-NOM-DU-SPACE.hf.space`.

### Étape 6 — Ajouter les photos

Va sur `https://.../admin`, connecte-toi avec le mot de passe admin, et
sélectionne toutes les photos de l'événement (plusieurs centaines à la fois,
c'est géré par lots automatiquement). Compte grosso modo **quelques minutes
pour 2000 photos** (le calcul se fait pendant l'upload).

### Étape 7 — Partager avec les invités

Donne-leur simplement l'adresse du Space + le mot de passe de l'événement.
Chacun envoie son selfie et retrouve ses photos.

---

## Limites à connaître (hébergement gratuit)

- **Mise en veille** : un Space gratuit inactif se met en pause et redémarre
  au premier visiteur (30-60 secondes de délai la toute première fois après
  une pause, ensuite c'est instantané). Rien à faire, c'est automatique.
- **Capacité** : pensé et testé pour un usage jusqu'à ~2000 photos. Au-delà,
  il faudrait passer à du matériel payant (quelques euros/mois).
- **Précision** : la reconnaissance fonctionne mieux avec des visages nets,
  de face et bien éclairés (aussi bien sur les photos de l'événement que sur
  le selfie). Le seuil de sensibilité (`MATCH_THRESHOLD`) est ajustable dans
  les secrets du Space si les résultats sont trop stricts ou trop larges.
- **Vie privée** : le dépôt Hugging Face qui stocke les photos doit rester
  **privé** (c'est le réglage par défaut ici) — seul toi (avec ton token) et
  l'appli y ont accès.

---

## Structure du projet

```
app/
  main.py          routes web (auth, upload, recherche)
  face_engine.py    détection de visages + calcul des empreintes (insightface)
  storage.py        où sont stockées les photos (local ou Hugging Face Dataset)
  index_store.py     index des empreintes de visages
  templates/         pages HTML (mot de passe, upload selfie, résultats, admin)
  static/            style
Dockerfile           pour le déploiement sur Hugging Face Spaces
requirements.txt
.env.example         variables de configuration à renseigner
```
