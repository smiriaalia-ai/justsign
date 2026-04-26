"""
ASL Just Dance — Entraînement MOTS (mains + visage)
===================================================
Ce script entraîne un modèle pour reconnaître les 40 signes
utilisés dans le jeu Bohemian Rhapsody.

SYSTÈME DE REPRISE :
  - Les données sont sauvegardées après chaque mot complété.
  - Si tu quittes (Q ou P), tu reprends automatiquement où tu t'es arrêté.
  - Fichier de progression : asl_progress.pkl
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
MODEL_PATH    = Path("asl_words_model.pkl")
PROGRESS_PATH = Path("asl_progress.pkl")   # fichier de progression automatique

# Les 40 signes utilisés dans le jeu Bohemian Rhapsody
WORDS = [
    "AFRAID", "BORN", "CONTINUE", "CRY", "DANCE",
    "DEATH", "DEVIL", "DIE", "DON'T-MATTER", "DREAM",
    "ESCAPE", "FEAR", "FORBID", "GOD", "GOODBYE",
    "HAPPEN", "KILL-HIM", "LATE", "LEAVE", "LIFE",
    "LOOK", "LOVE-ME", "ME", "MOTHER", "NO",
    "NONE", "NOT-WANT", "PAIN", "PITY", "POOR",
    "PRESERVE", "REAL", "RUIN", "RUN-AWAY", "SAD",
    "SOMETIMES", "STUCK", "THUNDER", "TRUTH", "WHATEVER",
]

WORD_HINTS = {
    "ABUSE-ME":     "Signe ABUSE + pointage vers soi",
    "ACCEPT":       "Deux mains ouvertes ramenées vers la poitrine",
    "AFRAID":       "Mains croisées devant la poitrine, écartement rapide",
    "ALL":          "Main droite fait un arc au-dessus de la main gauche",
    "ALLOW":        "Mains plates, mouvement vers l'avant et vers le bas",
    "APPROACH":     "Index droit se rapproche de l'index gauche",
    "AWESOME":      "Mains en A, mouvement d'enthousiasme vers l'avant",
    "BECOME":       "Mains jointes, rotation des poignets",
    "BEG":          "Mains en coupe tendues vers l'avant, regard suppliant",
    "BEGIN":        "Index tourne entre le majeur et l'index de l'autre main",
    "BORN":         "Main droite plate glisse depuis le ventre vers l'avant",
    "BOY":          "Main au front, fermeture des doigts (casquette)",
    "BUT":          "Index crochus qui s'écartent",
    "CAN":          "Deux poings descendent simultanément",
    "CONTINUE":     "Deux pouces joints glissent vers l'avant",
    "CRY":          "Index tracent des larmes sur les joues",
    "DANCE":        "Index et majeur 'dansent' sur la paume opposée",
    "DEATH":        "Main plate se retourne (de face à dos)",
    "DEVIL":        "Mains en Y sur les tempes (cornes)",
    "DIE":          "Main plate se retourne, paume vers le bas",
    "DON'T-CARE":   "Index du front, balayé vers l'extérieur",
    "DON'T-MATTER": "Main plate agitée devant soi (indifférence)",
    "DREAM":        "Index plié glisse depuis le front vers l'extérieur",
    "ESCAPE":       "Index droit sort rapidement d'un poing fermé",
    "EYES-OPEN":    "Doigts pointent les yeux, puis s'ouvrent",
    "FAIL":         "Index glisse du pouce vers le bas (toboggan)",
    "FAMILY":       "Deux mains en F, cercle vers l'avant",
    "FEAR":         "Doigts écartés tremblent devant la poitrine",
    "FINISH":       "Mains ouvertes, rotation rapide vers l'extérieur",
    "FOR":          "Index part du front et tourne vers l'avant",
    "FOR-ME":       "Index front + pointage vers soi",
    "FOR-US":       "Index front + cercle entre les deux personnes",
    "FORBID":       "Poing droit frappe la paume gauche ouverte",
    "FROM":         "X droit recule depuis l'index gauche",
    "GOD":          "Main ouverte du front vers le bas (révérence)",
    "GOODBYE":      "Main ouverte, salut de la main",
    "HAPPEN":       "Deux index pointent vers le haut, pivotent vers le bas",
    "HAPPY":        "Main plate fait des cercles ascendants sur la poitrine",
    "HE":           "Pointage index vers une personne",
    "HERE":         "Deux mains plates, cercles horizontaux devant soi",
    "HIS":          "Paume ouverte pointée vers une personne",
    "HURT-ME":      "Index se touchent douloureusement + pointage vers soi",
    "I":            "Pointage index vers sa propre poitrine",
    "IF":           "Deux petits doigts s'alternent en montant",
    "KILL-HIM":     "Index tranche sous la paume + pointage vers lui",
    "LATE":         "Main derrière l'épaule, signe 'pas encore'",
    "LEAVE":        "Main ouverte, balayage vers l'extérieur",
    "LEAVE?":       "Main ouverte, balayage + expression interrogative",
    "LIFE":         "Deux mains L remontent le long du torse",
    "LIGHTNING":    "Index zigzague vers le bas (éclair)",
    "LOOK":         "V pointé vers les yeux puis vers l'objet",
    "LOVE-ME":      "Bras croisés sur la poitrine + pointage vers soi",
    "LOVE-ME?":     "Bras croisés + pointage soi + expression interrogative",
    "MAN":          "Pouce du front descend vers la poitrine",
    "ME":           "Pointage index vers sa propre poitrine",
    "MOTHER":       "Pouce de la main ouverte tapote le menton",
    "MUST":         "X droit tire vers le bas avec force",
    "MY":           "Paume à plat sur la poitrine",
    "NO":           "Index + majeur se ferment sur le pouce",
    "NONE":         "Deux O devant soi, écartement vers les côtés",
    "NOT":          "Pouce sous le menton, balayage vers l'avant",
    "NOT-MATTER":   "Main plate agitée + signe NOT",
    "NOT-WANT":     "Mains en griffes retournées vers l'extérieur",
    "NOW":          "Mains courbées descendent ensemble",
    "OVERWHELM":    "Deux mains remontent au-dessus de la tête",
    "PAIN":         "Index se touchent par le bout douloureusement",
    "PAST":         "Main plate glisse par-dessus l'épaule",
    "PITY":         "Majeur fait des cercles sur la poitrine (compassion)",
    "PLEASE":       "Main circulaire sur la poitrine",
    "POOR":         "Main droite glisse sous le coude gauche (usure)",
    "PRESERVE":     "Deux mains P bougent vers le bas simultanément",
    "PRETEND":      "Index part du nez et part vers le côté",
    "REAL":         "Index droit part des lèvres vers l'avant",
    "RETURN":       "Index décrit un arc et revient vers soi",
    "RUIN":         "Deux R frottent l'un contre l'autre vers le bas",
    "RUN-AWAY":     "Index et pouce courent, puis balayage vers l'extérieur",
    "SAD":          "Deux mains ouvertes descendent devant le visage",
    "SEE-MYSELF":   "V pointé vers les yeux + pointage vers soi",
    "SOMETIMES":    "Index droit fait un cercle puis S",
    "SORRY":        "Poing en A, cercle sur la poitrine",
    "STUCK":        "V droit bloqué sous la main gauche",
    "SWEETHEART":   "Deux A se touchent les jointures puis ouvrent",
    "THINK":        "Index tapote la tempe",
    "THIS":         "Index pointe vers le bas (ici/ceci)",
    "THUNDER":      "T puis index zigzague (tonnerre)",
    "TOMORROW":     "Pouce de la joue avance vers l'avant",
    "TRUTH":        "Index part des lèvres vers l'avant (vérité)",
    "UNDERSTAND":   "Index plié depuis le front s'ouvre d'un coup",
    "WAIT":         "Doigts ondulent face à face (attente)",
    "WEAK":         "Doigts courbés glissent vers le bas sur la paume",
    "WEAK-MAN":     "WEAK + signe MAN enchaînés",
    "WHATEVER":     "W des deux mains s'écartent (peu importe)",
    "WHICH":        "Deux A alternent de haut en bas",
    "WHY":          "Doigts du front glissent vers le bas en Y",
    "WISH":         "Main en C sur la poitrine (désir)",
    "YES":          "Poing qui hoche comme un oui",
    "YOU":          "Pointage index vers l'interlocuteur",
}

# Nombre de frames par séquence (pour capturer le mouvement)
SEQUENCE_LENGTH  = 30
TARGET_SEQUENCES = 30  # séquences par mot  ← réduis à 15 pour aller plus vite


# ─────────────────────────────────────────────────────────
#  Sauvegarde / Reprise de progression
# ─────────────────────────────────────────────────────────
def save_progress(all_sequences, all_labels, next_word_idx):
    """Sauvegarde les données collectées + l'index du prochain mot."""
    data = {
        "sequences":        all_sequences,
        "labels":           all_labels,
        "next_word_idx":    next_word_idx,
        "target_sequences": TARGET_SEQUENCES,
    }
    with open(PROGRESS_PATH, "wb") as f:
        pickle.dump(data, f)
    size = PROGRESS_PATH.stat().st_size / 1024
    label = WORDS[next_word_idx] if next_word_idx < len(WORDS) else "FIN"
    print(f"  💾 Progression sauvegardée ({size:.1f} KB) — reprendra à : {label}")


def load_progress():
    """
    Charge la progression existante.
    Retourne (sequences, labels, next_word_idx) ou None si aucune sauvegarde.
    """
    if not PROGRESS_PATH.exists():
        return None
    with open(PROGRESS_PATH, "rb") as f:
        data = pickle.load(f)

    seqs   = data.get("sequences", [])
    labels = data.get("labels", [])
    idx    = data.get("next_word_idx", 0)

    print(f"\n📂 Sauvegarde trouvée !")
    print(f"   Mots terminés     : {idx} / {len(WORDS)}")
    print(f"   Séquences stockées: {len(seqs)}")
    if idx < len(WORDS):
        print(f"   Prochain mot      : {WORDS[idx]}")
    return seqs, labels, idx


def delete_progress():
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
        print(f"🗑️  Fichier de progression supprimé.")


# ─────────────────────────────────────────────────────────
#  Détecteurs MediaPipe
# ─────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_face     = mp.solutions.face_mesh
mp_pose     = mp.solutions.pose
mp_holistic = mp.solutions.holistic


def init_holistic():
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
    features = []

    # Main gauche
    if results.left_hand_landmarks:
        lm = results.left_hand_landmarks.landmark
        wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
        for p in lm:
            features.extend([p.x - wx, p.y - wy, p.z - wz])
    else:
        features.extend([0.0] * 63)

    # Main droite
    if results.right_hand_landmarks:
        lm = results.right_hand_landmarks.landmark
        wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
        for p in lm:
            features.extend([p.x - wx, p.y - wy, p.z - wz])
    else:
        features.extend([0.0] * 63)

    # Visage (20 points clés)
    FACE_KEYPOINTS = [10,152,234,454,1,4,5,195,61,291,13,14,33,133,362,263,70,300,94,323]
    if results.face_landmarks:
        lm = results.face_landmarks.landmark
        nx, ny, nz = lm[1].x, lm[1].y, lm[1].z
        for idx in FACE_KEYPOINTS:
            if idx < len(lm):
                p = lm[idx]
                features.extend([p.x - nx, p.y - ny, p.z - nz])
            else:
                features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0] * 60)

    # Pose (épaules, coudes, poignets, hanches)
    POSE_KEYPOINTS = [11, 12, 13, 14, 15, 16, 23, 24]
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
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
    if len(sequence) < 2:
        return None
    sequence = np.array(sequence)
    combined = []
    combined.extend(np.mean(sequence, axis=0))
    combined.extend(np.std(sequence, axis=0))
    combined.extend(sequence[-1] - sequence[0])
    velocities = np.diff(sequence, axis=0)
    combined.extend(np.mean(np.abs(velocities), axis=0))
    return combined  # 840 features


# ─────────────────────────────────────────────────────────
#  Collecte de données (avec reprise)
# ─────────────────────────────────────────────────────────
def collect_word_data(target_sequences=TARGET_SEQUENCES,
                      resume_sequences=None,
                      resume_labels=None,
                      start_word_idx=0):
    """
    Collecte des séquences depuis la webcam.
    Supporte la reprise depuis un mot précis avec des données déjà collectées.
    Retourne (X, y, paused).
    """
    all_sequences = list(resume_sequences) if resume_sequences else []
    all_labels    = list(resume_labels)    if resume_labels    else []
    total_words   = len(WORDS)
    paused        = False

    print("\n📷 COLLECTE DE DONNÉES — MOTS ASL")
    print("=" * 55)
    print(f"Séquences par mot : {target_sequences}")
    print(f"Mots restants     : {total_words - start_word_idx} / {total_words}")
    print("\nTouches :")
    print("  ESPACE  → enregistrer une séquence")
    print("  S       → passer au mot suivant")
    print("  P       → PAUSE  (sauvegarde et quitte)")
    print("  Q       → quitter (sauvegarde et quitte)\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    holistic   = init_holistic()
    mp_drawing = mp.solutions.drawing_utils

    for word_offset, word in enumerate(WORDS[start_word_idx:]):
        word_idx       = start_word_idx + word_offset
        word_sequences = []

        print(f"\n[{word_idx+1}/{total_words}] Mot : {word}")
        print(f"  → {WORD_HINTS.get(word, '')}")

        while len(word_sequences) < target_sequences:
            recording        = False
            current_sequence = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame   = cv2.flip(frame, 1)
                h, w    = frame.shape[:2]
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                # Landmarks
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1))
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2))
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2))
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2))

                # Enregistrement
                if recording:
                    features = extract_holistic_features(results)
                    current_sequence.append(features)
                    progress = len(current_sequence) / SEQUENCE_LENGTH
                    bar_w    = w - 100
                    cv2.rectangle(frame, (50, h-60), (50+bar_w, h-30), (40,20,60), -1)
                    cv2.rectangle(frame, (50, h-60), (50+int(bar_w*progress), h-30), (0,0,255), -1)
                    cv2.putText(frame, "ENREGISTREMENT...", (50, h-70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    if len(current_sequence) >= SEQUENCE_LENGTH:
                        seq_features = extract_sequence_features(current_sequence)
                        if seq_features:
                            word_sequences.append(seq_features)
                            print(f"    ✓ Séquence {len(word_sequences)}/{target_sequences}")
                        recording        = False
                        current_sequence = []

                # Interface
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 130), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, f"MOT: {word}", (20, 45),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,220,255), 2)
                cv2.putText(frame, WORD_HINTS.get(word, ""), (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                cv2.putText(frame,
                            f"Sequences: {len(word_sequences)}/{target_sequences}   "
                            f"Mot {word_idx+1}/{total_words}",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,255,100), 1)
                cv2.putText(frame, "ESPACE=enregistrer | S=suivant | P=pause | Q=quitter",
                            (w-610, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                cv2.imshow("Collecte ASL - Mots", frame)

                k = cv2.waitKey(1) & 0xFF

                if k == ord(' ') and not recording:
                    recording        = True
                    current_sequence = []
                    print(f"    ● Séquence {len(word_sequences)+1}...")

                elif k == ord('s'):
                    break  # passe au mot suivant

                elif k in (ord('p'), ord('q')):
                    # Sauvegarde les séquences partielles du mot en cours
                    for seq in word_sequences:
                        all_sequences.append(seq)
                        all_labels.append(word)
                    # Reprendre SUR ce mot (pas complété)
                    save_progress(all_sequences, all_labels, word_idx)
                    if k == ord('p'):
                        print(f"\n⏸️  PAUSE — reprendra au mot '{word}' au prochain lancement.")
                    else:
                        print(f"\n💾 Sauvegardé — reprendra au mot '{word}' au prochain lancement.")
                    cap.release()
                    holistic.close()
                    cv2.destroyAllWindows()
                    paused = True
                    return np.array(all_sequences), np.array(all_labels), paused

            if len(word_sequences) >= target_sequences:
                break

        # Mot entièrement collecté
        for seq in word_sequences:
            all_sequences.append(seq)
            all_labels.append(word)
        print(f"  ✅ {len(word_sequences)} séquences pour '{word}'")

        # Sauvegarde automatique après chaque mot
        save_progress(all_sequences, all_labels, word_idx + 1)

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
    return np.array(all_sequences), np.array(all_labels), paused


# ─────────────────────────────────────────────────────────
#  Entraînement
# ─────────────────────────────────────────────────────────
def train_word_model(X, y):
    print(f"\n🧠 Entraînement du modèle...")
    print(f"   {len(X)} séquences, {len(set(y))} mots")

    le       = LabelEncoder()
    y_enc    = le.fit_transform(y)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

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

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n✅ Terminé en {duration:.1f}s  —  Précision : {acc*100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return model, scaler, le


def save_word_model(model, scaler, label_encoder, path=MODEL_PATH):
    data = {
        "model":           model,
        "scaler":          scaler,
        "label_encoder":   label_encoder,
        "words":           WORDS,
        "sequence_length": SEQUENCE_LENGTH,
        "version":         "2.0-bohemian-rhapsody",
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n💾 Modèle → {path}  ({path.stat().st_size/1024:.1f} KB)")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ASL Just Dance — Bohemian Rhapsody (Queen)")
    print(f"  {len(WORDS)} mots uniques")
    print("=" * 60)

    # Vérifier s'il existe une progression sauvegardée
    progress = load_progress()

    if progress is not None:
        resume_seqs, resume_labels, start_idx = progress

        if start_idx >= len(WORDS):
            # Tous les mots sont déjà collectés → entraînement direct
            print("\n✅ Collecte terminée ! Lancement de l'entraînement...")
            X = np.array(resume_seqs)
            y = np.array(resume_labels)
            model, scaler, le = train_word_model(X, y)
            save_word_model(model, scaler, le)
            delete_progress()
            print("\n🎸 Modèle Bohemian Rhapsody prêt !")
            return

        rep = input(f"\nReprendre depuis '{WORDS[start_idx]}' (mot {start_idx+1}/{len(WORDS)}) ? [O/n] : ").strip().lower()
        if rep == 'n':
            print("Nouveau départ — la sauvegarde précédente sera écrasée.")
            resume_seqs, resume_labels, start_idx = [], [], 0
    else:
        print(f"\n{len(WORDS)} mots : {', '.join(WORDS[:8])} ...")
        resume_seqs, resume_labels, start_idx = [], [], 0

    input("\nAppuie sur ENTRÉE pour commencer...\n")

    X, y, paused = collect_word_data(
        target_sequences=TARGET_SEQUENCES,
        resume_sequences=resume_seqs,
        resume_labels=resume_labels,
        start_word_idx=start_idx,
    )

    if paused:
        print("\n⏸️  Session suspendue. Relance le script pour continuer.")
        return

    if len(X) < 10:
        print("❌ Pas assez de données. Abandonne.")
        return

    model, scaler, le = train_word_model(X, y)
    save_word_model(model, scaler, le)
    delete_progress()

    print("\n🎸 Modèle Bohemian Rhapsody prêt !")
    print("   Lance : python3 justsign_v2.py")


if __name__ == "__main__":
    main()
