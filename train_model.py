"""
ASL Just Dance — Entraînement du modèle IA
==========================================
Ce script :
1. Télécharge le dataset ASL (images de mains)
2. Extrait les landmarks MediaPipe de chaque image
3. Entraîne un réseau de neurones (MLP) sur ces landmarks
4. Sauvegarde le modèle → asl_model.pkl

Durée estimée : 10-20 min selon ta machine
"""

import os
import sys
import time
import math
import pickle
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────
#  Vérification des dépendances
# ─────────────────────────────────────────────────────────
def check_and_install(package, import_name=None):
    import importlib
    name = import_name or package
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"📦 Installation de {package}...")
        os.system(f"{sys.executable} -m pip install {package} -q")

check_and_install("scikit-learn", "sklearn")
check_and_install("opencv-python", "cv2")
check_and_install("mediapipe")
check_and_install("requests")
check_and_install("tqdm")

import cv2
import mediapipe as mp
import requests
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ─────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────
DATASET_URL = "https://www.kaggle.com/datasets/grassknoted/asl-alphabet/download"
DATA_DIR    = Path("asl_dataset")
MODEL_PATH  = Path("asl_model.pkl")
LETTERS     = list("ABCDEFGHIKLMNOPQRSTUVWXY")
MAX_IMAGES  = 500   # images par lettre (augmenter pour plus de précision)

# ─────────────────────────────────────────────────────────
#  Téléchargement dataset alternatif (sans compte Kaggle)
#  On utilise le dataset Roboflow ASL accessible publiquement
# ─────────────────────────────────────────────────────────
ROBOFLOW_DATASETS = [
    # Dataset ASL letters via GitHub (landmarks pré-extraits)
    {
        "name": "ASL Landmark Dataset (GitHub)",
        "url": "https://raw.githubusercontent.com/kinivi/hand-gesture-recognition-mediapipe/main/model/keypoint_classifier/keypoint.csv",
        "type": "landmarks_csv"
    }
]

# ─────────────────────────────────────────────────────────
#  Extraction des landmarks MediaPipe
# ─────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

def extract_landmarks(image_path):
    """
    Extrait 63 features (21 points x,y,z normalisés) depuis une image.
    Retourne None si aucune main détectée.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0].landmark
    # Normalisation par rapport au poignet (point 0)
    wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
    features = []
    for point in lm:
        features.extend([
            point.x - wrist_x,
            point.y - wrist_y,
            point.z - wrist_z,
        ])
    return features

def extract_landmarks_from_frame(hand_landmarks):
    """
    Extrait les features depuis un hand_landmarks MediaPipe déjà détecté
    (utilisé en temps réel dans le jeu).
    """
    lm = hand_landmarks.landmark
    wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
    features = []
    for point in lm:
        features.extend([
            point.x - wrist_x,
            point.y - wrist_y,
            point.z - wrist_z,
        ])
    return features

# ─────────────────────────────────────────────────────────
#  Collecte des données depuis les images
# ─────────────────────────────────────────────────────────
def collect_data_from_images(data_dir: Path):
    """
    Parcourt le dossier dataset organisé par lettre :
    asl_dataset/
        A/  image1.jpg  image2.jpg ...
        B/  ...
    """
    X, y = [], []
    print("\n🔍 Extraction des landmarks depuis les images...")

    for letter in LETTERS:
        letter_dir = data_dir / letter
        if not letter_dir.exists():
            letter_dir = data_dir / letter.lower()
        if not letter_dir.exists():
            print(f"  ⚠️  Dossier manquant pour '{letter}', ignoré.")
            continue

        images = list(letter_dir.glob("*.jpg")) + \
                 list(letter_dir.glob("*.jpeg")) + \
                 list(letter_dir.glob("*.png"))
        images = images[:MAX_IMAGES]

        ok = 0
        for img_path in tqdm(images, desc=f"  {letter}", leave=False):
            features = extract_landmarks(img_path)
            if features:
                X.append(features)
                y.append(letter)
                ok += 1

        print(f"  ✓ {letter} : {ok}/{len(images)} images traitées")

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────
#  Génération synthétique de données (fallback)
#  Si pas de dataset, on génère des exemples à partir de
#  la webcam de l'utilisateur
# ─────────────────────────────────────────────────────────
def collect_data_from_webcam(target_per_letter=80):
    """
    Mode collecte webcam : montre chaque lettre à l'utilisateur,
    capture automatiquement les landmarks quand une main est détectée.
    """
    print("\n📷 MODE COLLECTE WEBCAM")
    print("=" * 50)
    print("On va collecter des exemples depuis ta webcam.")
    print(f"Pour chaque lettre : tiens le signe, on capture {target_per_letter} frames.")
    print("Appuie sur ESPACE pour commencer chaque lettre, Q pour quitter.\n")

    cap = cv2.VideoCapture(0)
    hands_live = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
    )
    mp_drawing = mp.solutions.drawing_utils

    X, y = [], []

    for letter_idx, letter in enumerate(LETTERS):
        print(f"[{letter_idx+1}/{len(LETTERS)}] Lettre : {letter}")
        print(f"  → {_hint(letter)}")
        print(f"  Appuie sur ESPACE pour commencer la capture...")

        # Phase attente
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            _draw_instruction(frame, letter, f"Forme le signe '{letter}'", "ESPACE = commencer")
            cv2.imshow("Collecte données ASL", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '): break
            if k == ord('q'):
                cap.release()
                hands_live.close()
                cv2.destroyAllWindows()
                return np.array(X), np.array(y)

        # Phase capture
        count = 0
        print(f"  Capture en cours ({target_per_letter} frames)...")
        while count < target_per_letter:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_live.process(rgb)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                features = extract_landmarks_from_frame(lm)
                X.append(features)
                y.append(letter)
                count += 1
                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,200), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(180,0,255), thickness=2),
                )

            # Barre de progression
            progress = count / target_per_letter
            bar_w = frame.shape[1] - 40
            cv2.rectangle(frame, (20, frame.shape[0]-40), (20+bar_w, frame.shape[0]-20), (40,20,60), -1)
            cv2.rectangle(frame, (20, frame.shape[0]-40), (20+int(bar_w*progress), frame.shape[0]-20), (50,220,120), -1)
            _draw_instruction(frame, letter, f"Maintiens le signe '{letter}'", f"{count}/{target_per_letter}")
            cv2.imshow("Collecte données ASL", frame)
            cv2.waitKey(1)

        print(f"  ✓ {count} exemples capturés pour '{letter}'")

    cap.release()
    hands_live.close()
    cv2.destroyAllWindows()
    return np.array(X), np.array(y)


def _hint(letter):
    hints = {
        "A": "Poing, pouce sur le côté",
        "B": "4 doigts tendus, pouce plié",
        "C": "Main en forme de C",
        "D": "Index tendu, cercle pouce-majeur",
        "E": "Doigts courbés vers la paume",
        "F": "Cercle pouce-index, autres tendus",
        "G": "Index horizontal, pouce parallèle",
        "H": "Index+majeur horizontaux",
        "I": "Auriculaire tendu seul",
        "K": "Index+majeur+pouce en K",
        "L": "Index+pouce en L",
        "M": "Pouce sous 3 doigts",
        "N": "Pouce sous 2 doigts",
        "O": "Doigts forment un O",
        "P": "Index vers le bas",
        "Q": "Index+pouce vers le bas",
        "R": "Index+majeur croisés",
        "S": "Poing, pouce par-dessus",
        "T": "Pouce entre index-majeur",
        "U": "Index+majeur tendus ensemble",
        "V": "Index+majeur en V",
        "W": "3 doigts tendus",
        "X": "Index courbé en crochet",
        "Y": "Pouce+auriculaire tendus",
    }
    return hints.get(letter, "")

def _draw_instruction(frame, letter, line1, line2):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    # Grande lettre
    (lw, lh), _ = cv2.getTextSize(letter, font, 6.0, 10)
    cv2.putText(frame, letter, (w//2 - lw//2 + 3, h//2 + lh//2 + 3),
                font, 6.0, (0,0,0), 14, cv2.LINE_AA)
    cv2.putText(frame, letter, (w//2 - lw//2, h//2 + lh//2),
                font, 6.0, (0, 220, 255), 10, cv2.LINE_AA)
    # Instructions
    cv2.putText(frame, line1, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,220,255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────
#  Entraînement du modèle
# ─────────────────────────────────────────────────────────
def train(X, y):
    print(f"\n🧠 Entraînement du modèle...")
    print(f"   Données : {len(X)} exemples, {len(set(y))} lettres")

    # Encodage des labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Modèle MLP (réseau de neurones multicouche)
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True,
        learning_rate_init=0.001,
    )

    print("\n   Entraînement en cours (peut prendre quelques minutes)...\n")
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    # Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Entraînement terminé en {duration:.1f}s")
    print(f"   Précision sur le jeu de test : {acc*100:.1f}%")
    print("\n📊 Rapport détaillé :")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, scaler, le


# ─────────────────────────────────────────────────────────
#  Sauvegarde du modèle
# ─────────────────────────────────────────────────────────
def save_model(model, scaler, label_encoder, path=MODEL_PATH):
    data = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "letters": LETTERS,
        "version": "2.0",
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    size = path.stat().st_size / 1024
    print(f"\n💾 Modèle sauvegardé → {path}  ({size:.1f} KB)")


# ─────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  ASL Just Dance — Entraînement du modèle IA")
    print("=" * 55)

    # Vérifie si un dataset image existe
    has_dataset = DATA_DIR.exists() and any(
        (DATA_DIR / l).exists() for l in LETTERS
    )

    if has_dataset:
        print(f"\n📁 Dataset trouvé dans '{DATA_DIR}/'")
        X, y = collect_data_from_images(DATA_DIR)
        if len(X) == 0:
            print("⚠️  Aucune image traitée. Passage en mode webcam.")
            has_dataset = False

    if not has_dataset:
        print(f"\n📁 Pas de dataset image trouvé dans '{DATA_DIR}/'")
        print("👉 On va collecter les données depuis ta webcam !")
        print("   Tu devras montrer chaque lettre ASL à la caméra.")
        input("\n   Appuie sur ENTRÉE pour commencer...\n")
        X, y = collect_data_from_webcam(target_per_letter=80)

    if len(X) < 10:
        print("❌ Pas assez de données collectées. Abandonne.")
        return

    # Entraînement
    model, scaler, le = train(X, y)

    # Sauvegarde
    save_model(model, scaler, le)

    print("\n🎮 Le modèle est prêt !")
    print("   Lance maintenant : python3 justsign.py")


if __name__ == "__main__":
    main()
