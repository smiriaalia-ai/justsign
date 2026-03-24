"""
ASL Just Dance — Entraînement MOTS (mains + visage)
===================================================
Ce script entraîne un modèle pour reconnaître des mots ASL
qui nécessitent la détection des mains ET du visage.

Mots inclus : HELLO, THANK YOU, PLEASE, SORRY, YES, NO, HELP, etc.
"""

import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path

# Vérification des dépendances
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
check_and_install("tqdm")

import cv2
import mediapipe as mp
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ─────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────
MODEL_PATH = Path("asl_words_model.pkl")

# Mots ASL à reconnaître (niveau intermédiaire)
WORDS = [
    "HELLO",      # Main ouverte près du front, mouvement vers l'extérieur
    "THANK_YOU",  # Main plate du menton vers l'avant
    "PLEASE",     # Main circulaire sur la poitrine
    "SORRY",      # Poing en cercle sur la poitrine
    "YES",        # Poing qui hoche (comme la tête)
    "NO",         # Index+majeur qui se ferment sur le pouce
    "HELP",       # Poing sur paume ouverte, lever
    "LOVE",       # Bras croisés sur la poitrine
    "FRIEND",     # Index crochus qui s'accrochent
    "FAMILY",     # Deux mains en F, cercle
    "EAT",        # Main vers la bouche
    "DRINK",      # Main en C vers la bouche
    "WATER",      # W sur le menton
    "NAME",       # Deux doigts H qui se tapent
    "GOOD",       # Main plate du menton vers le bas
    "BAD",        # Main de la bouche vers le bas, retournée
]

WORD_HINTS = {
    "HELLO": "Main ouverte près du front → vers l'extérieur",
    "THANK_YOU": "Main plate du menton → vers l'avant",
    "PLEASE": "Main circulaire sur la poitrine",
    "SORRY": "Poing en A, cercle sur la poitrine",
    "YES": "Poing qui hoche comme un oui",
    "NO": "Index+majeur se ferment sur le pouce",
    "HELP": "Poing sur paume ouverte, lever",
    "LOVE": "Bras croisés sur la poitrine",
    "FRIEND": "Index crochus qui s'accrochent",
    "FAMILY": "Deux mains en F, mouvement circulaire",
    "EAT": "Main groupée vers la bouche",
    "DRINK": "Main en C vers la bouche (boire)",
    "WATER": "W (3 doigts) tapote le menton",
    "NAME": "Deux doigts H qui se croisent",
    "GOOD": "Main plate du menton vers le bas",
    "BAD": "Main de la bouche, retourne vers le bas",
}

# Nombre de frames par séquence (pour capturer le mouvement)
SEQUENCE_LENGTH = 30
TARGET_SEQUENCES = 30  # séquences par mot

# ─────────────────────────────────────────────────────────
#  Détecteurs MediaPipe
# ─────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

def init_holistic():
    """Initialise le détecteur holistique (mains + visage + pose)"""
    return mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

# ─────────────────────────────────────────────────────────
#  Extraction des features
# ─────────────────────────────────────────────────────────
def extract_holistic_features(results):
    """
    Extrait les features depuis les résultats MediaPipe Holistic.
    
    Features extraites :
    - Main gauche : 21 points × 3 coords = 63
    - Main droite : 21 points × 3 coords = 63  
    - Visage (points clés) : 20 points × 3 coords = 60
    - Pose (épaules, coudes, poignets) : 8 points × 3 coords = 24
    
    Total : 210 features par frame
    """
    features = []
    
    # === MAIN GAUCHE ===
    if results.left_hand_landmarks:
        lm = results.left_hand_landmarks.landmark
        # Normaliser par rapport au poignet
        wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
        for p in lm:
            features.extend([p.x - wx, p.y - wy, p.z - wz])
    else:
        features.extend([0.0] * 63)  # padding si pas de main
    
    # === MAIN DROITE ===
    if results.right_hand_landmarks:
        lm = results.right_hand_landmarks.landmark
        wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
        for p in lm:
            features.extend([p.x - wx, p.y - wy, p.z - wz])
    else:
        features.extend([0.0] * 63)
    
    # === VISAGE (points clés sélectionnés) ===
    # On prend 20 points stratégiques du visage
    FACE_KEYPOINTS = [
        10,   # front haut
        152,  # menton
        234,  # joue gauche
        454,  # joue droite
        1,    # nez bout
        4,    # nez milieu
        5,    # nez haut
        195,  # entre les yeux
        61,   # coin bouche gauche
        291,  # coin bouche droite
        13,   # lèvre sup milieu
        14,   # lèvre inf milieu
        33,   # coin œil gauche ext
        133,  # coin œil gauche int
        362,  # coin œil droit int
        263,  # coin œil droit ext
        70,   # sourcil gauche
        300,  # sourcil droit
        94,   # nez côté gauche
        323,  # nez côté droit
    ]
    
    if results.face_landmarks:
        lm = results.face_landmarks.landmark
        # Normaliser par rapport au nez (point 1)
        nx, ny, nz = lm[1].x, lm[1].y, lm[1].z
        for idx in FACE_KEYPOINTS:
            if idx < len(lm):
                p = lm[idx]
                features.extend([p.x - nx, p.y - ny, p.z - nz])
            else:
                features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0] * 60)
    
    # === POSE (épaules, coudes, poignets) ===
    POSE_KEYPOINTS = [11, 12, 13, 14, 15, 16, 23, 24]  # épaules, coudes, poignets, hanches
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # Normaliser par rapport au milieu des épaules
        mid_x = (lm[11].x + lm[12].x) / 2
        mid_y = (lm[11].y + lm[12].y) / 2
        mid_z = (lm[11].z + lm[12].z) / 2
        for idx in POSE_KEYPOINTS:
            p = lm[idx]
            features.extend([p.x - mid_x, p.y - mid_y, p.z - mid_z])
    else:
        features.extend([0.0] * 24)
    
    return features  # 210 features


def extract_sequence_features(sequence):
    """
    Combine les features de plusieurs frames en une séquence.
    Ajoute des features de mouvement (vélocité).
    
    Input: liste de frames features [f1, f2, ..., fn]
    Output: features combinées pour la séquence
    """
    if len(sequence) < 2:
        return None
    
    sequence = np.array(sequence)
    
    # Features statistiques sur la séquence
    combined = []
    
    # Moyenne de chaque feature
    combined.extend(np.mean(sequence, axis=0))
    
    # Écart-type (variabilité = mouvement)
    combined.extend(np.std(sequence, axis=0))
    
    # Différence début-fin (direction du mouvement)
    combined.extend(sequence[-1] - sequence[0])
    
    # Vélocité moyenne (différence entre frames consécutives)
    velocities = np.diff(sequence, axis=0)
    combined.extend(np.mean(np.abs(velocities), axis=0))
    
    return combined  # 210 × 4 = 840 features


# ─────────────────────────────────────────────────────────
#  Collecte de données webcam
# ─────────────────────────────────────────────────────────
def collect_word_data(target_sequences=TARGET_SEQUENCES):
    """
    Collecte des séquences de mots ASL depuis la webcam.
    """
    print("\n📷 COLLECTE DE DONNÉES — MOTS ASL")
    print("=" * 55)
    print(f"On va collecter {target_sequences} séquences par mot.")
    print("Chaque séquence = le mouvement complet du signe.")
    print("\nInstructions :")
    print("  • ESPACE : commencer l'enregistrement d'une séquence")
    print("  • S : passer au mot suivant")
    print("  • Q : quitter et sauvegarder\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    holistic = init_holistic()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    all_sequences = []
    all_labels = []
    
    for word_idx, word in enumerate(WORDS):
        print(f"\n[{word_idx+1}/{len(WORDS)}] Mot : {word}")
        print(f"  → {WORD_HINTS.get(word, '')}")
        
        word_sequences = []
        
        while len(word_sequences) < target_sequences:
            recording = False
            current_sequence = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Détection
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                
                # Dessiner les landmarks
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1),
                    )
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2),
                    )
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2),
                    )
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2),
                    )
                
                # Enregistrement
                if recording:
                    features = extract_holistic_features(results)
                    current_sequence.append(features)
                    
                    # Indicateur visuel
                    progress = len(current_sequence) / SEQUENCE_LENGTH
                    bar_w = w - 100
                    cv2.rectangle(frame, (50, h-60), (50+bar_w, h-30), (40,20,60), -1)
                    cv2.rectangle(frame, (50, h-60), (50+int(bar_w*progress), h-30), (0,0,255), -1)
                    cv2.putText(frame, "ENREGISTREMENT...", (50, h-70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    
                    # Fin de séquence
                    if len(current_sequence) >= SEQUENCE_LENGTH:
                        seq_features = extract_sequence_features(current_sequence)
                        if seq_features:
                            word_sequences.append(seq_features)
                            print(f"    ✓ Séquence {len(word_sequences)}/{target_sequences}")
                        recording = False
                        current_sequence = []
                
                # Interface
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 120), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                cv2.putText(frame, f"MOT: {word}", (20, 45),
                           cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,220,255), 2)
                cv2.putText(frame, WORD_HINTS.get(word, ""), (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                cv2.putText(frame, f"Sequences: {len(word_sequences)}/{target_sequences}", (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 1)
                
                status = "ESPACE=enregistrer | S=suivant | Q=quitter"
                cv2.putText(frame, status, (w-500, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                
                cv2.imshow("Collecte ASL - Mots", frame)
                
                k = cv2.waitKey(1) & 0xFF
                if k == ord(' ') and not recording:
                    recording = True
                    current_sequence = []
                    print(f"    ● Enregistrement séquence {len(word_sequences)+1}...")
                elif k == ord('s'):
                    break
                elif k == ord('q'):
                    # Sauvegarder ce qu'on a
                    for seq in word_sequences:
                        all_sequences.append(seq)
                        all_labels.append(word)
                    cap.release()
                    holistic.close()
                    cv2.destroyAllWindows()
                    return np.array(all_sequences), np.array(all_labels)
            
            if len(word_sequences) >= target_sequences:
                break
        
        # Ajouter les séquences de ce mot
        for seq in word_sequences:
            all_sequences.append(seq)
            all_labels.append(word)
        print(f"  ✓ {len(word_sequences)} séquences collectées pour '{word}'")
    
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
    
    return np.array(all_sequences), np.array(all_labels)


# ─────────────────────────────────────────────────────────
#  Entraînement
# ─────────────────────────────────────────────────────────
def train_word_model(X, y):
    print(f"\n🧠 Entraînement du modèle MOTS...")
    print(f"   Données : {len(X)} séquences, {len(set(y))} mots")
    
    # Encodage
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # Modèle plus grand pour les séquences
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        verbose=True,
        learning_rate_init=0.0005,
        batch_size=32,
    )
    
    print("\n   Entraînement en cours...\n")
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0
    
    # Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Entraînement terminé en {duration:.1f}s")
    print(f"   Précision : {acc*100:.1f}%")
    print("\n📊 Rapport :")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return model, scaler, le


def save_word_model(model, scaler, label_encoder, path=MODEL_PATH):
    data = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "words": WORDS,
        "sequence_length": SEQUENCE_LENGTH,
        "version": "1.0-words",
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    size = path.stat().st_size / 1024
    print(f"\n💾 Modèle sauvegardé → {path}  ({size:.1f} KB)")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  ASL Just Dance — Entraînement MOTS")
    print("=" * 55)
    print(f"\nMots à apprendre : {', '.join(WORDS)}")
    
    input("\nAppuie sur ENTRÉE pour commencer la collecte...\n")
    
    X, y = collect_word_data(target_sequences=TARGET_SEQUENCES)
    
    if len(X) < 10:
        print("❌ Pas assez de données. Abandonne.")
        return
    
    model, scaler, le = train_word_model(X, y)
    save_word_model(model, scaler, le)
    
    print("\n🎮 Modèle MOTS prêt !")
    print("   Lance : python3 justsign_v2.py")


if __name__ == "__main__":
    main()