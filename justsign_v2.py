"""
ASL Just Dance v5 — Lettres + Mots
==================================
- Débutant : lettres simples (main seule)
- Intermédiaire : MOTS (main + visage + mouvement)
- Expert : lettres rapides

  ┌──────────────┬──────────────┐
  │  CAMÉRA LIVE │ GESTE A FAIRE│
  ├──────────────┴──────────────┤
  │  ◄◄◄  éléments défilent ◄◄◄ │
  └─────────────────────────────┘
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pickle
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

# ═══════════════════════════════════════
#  DIMENSIONS
# ═══════════════════════════════════════
SW, SH = 1080, 680
HW = SW // 2
CAM_H = 480
SCR_H = SH - CAM_H
WIN = "ASL Just Dance"

# ═══════════════════════════════════════
#  NIVEAUX
# ═══════════════════════════════════════
LEVELS = {
    "debutant": {
        "label": "DÉBUTANT",
        "mode": "letters",
        "items": list("ABCIL"),
        "t": 14.0,
        "hold": 15,
        "color": (40, 230, 110),
        "description": "Lettres simples • Main seule",
    },
    "intermediaire": {
        "label": "INTERMÉDIAIRE", 
        "mode": "words",
        "items": ["HELLO", "THANK_YOU", "PLEASE", "YES", "NO", "HELP", "GOOD", "EAT"],
        "t": 8.0,
        "hold": 25,
        "color": (0, 130, 255),
        "description": "Mots ASL • Main + Visage",
    },
    "expert": {
        "label": "EXPERT",
        "mode": "letters", 
        "items": list("ABCDEFGHIKLMNOPQRSTUVWXY"),
        "t": 6.0,
        "hold": 25,
        "color": (30, 30, 220),
        "description": "Alphabet complet • Rapide",
    },
}

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
WORDS = ["HELLO", "THANK_YOU", "PLEASE", "SORRY", "YES", "NO", "HELP", 
         "LOVE", "FRIEND", "FAMILY", "EAT", "DRINK", "WATER", "NAME", "GOOD", "BAD"]

SPD = 2.0
GAP = 180
TX = 160

# ═══════════════════════════════════════
#  COULEURS
# ═══════════════════════════════════════
BG = (14, 10, 26)
PINK = (160, 30, 240)
CYAN = (230, 210, 0)
GRN = (40, 230, 110)
ORG = (0, 130, 255)
RED = (30, 30, 220)
GOLD = (20, 185, 255)
WHT = (235, 235, 255)
DRK = (25, 16, 46)
MID = (50, 35, 80)
GRY = (90, 80, 110)
FD, FS = cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX

# ═══════════════════════════════════════
#  HINTS
# ═══════════════════════════════════════
LETTER_HINTS = {
    "A": "Poing / pouce côté", "B": "4 doigts tendus", "C": "Main en C",
    "D": "Index + cercle", "E": "Doigts courbés", "F": "Cercle pouce-index",
    "G": "Index horizontal", "H": "2 doigts horiz.", "I": "Auriculaire seul",
    "K": "Index+maj+pouce", "L": "L avec les doigts", "M": "Pouce sous 3 doigts",
    "N": "Pouce sous 2 doigts", "O": "Doigts en O", "P": "Index vers le bas",
    "Q": "Index+pouce bas", "R": "Doigts croisés", "S": "Poing fermé",
    "T": "Pouce entre doigts", "U": "2 doigts serrés", "V": "Doigts en V",
    "W": "3 doigts", "X": "Index courbé", "Y": "Pouce+auriculaire",
}

WORD_HINTS = {
    "HELLO": "Main ouverte, front → extérieur",
    "THANK_YOU": "Main plate, menton → avant",
    "PLEASE": "Main en cercle sur poitrine",
    "SORRY": "Poing A en cercle sur poitrine",
    "YES": "Poing qui hoche",
    "NO": "Index+majeur → pouce",
    "HELP": "Poing sur paume, lever",
    "LOVE": "Bras croisés poitrine",
    "FRIEND": "Index crochus liés",
    "FAMILY": "Mains F en cercle",
    "EAT": "Main groupée → bouche",
    "DRINK": "Main C → bouche",
    "WATER": "W tapote menton",
    "NAME": "Doigts H croisés",
    "GOOD": "Main menton → bas",
    "BAD": "Main bouche → retourner bas",
}

# ═══════════════════════════════════════
#  MODÈLES IA
# ═══════════════════════════════════════
_letter_model = None
_word_model = None
SEQUENCE_LENGTH = 30  # frames pour les mots


def load_models():
    global _letter_model, _word_model
    
    # Modèle lettres
    p = Path("asl_model.pkl")
    if p.exists():
        with open(p, "rb") as f:
            _letter_model = pickle.load(f)
        print("✓ Modèle LETTRES chargé")
    else:
        print("⚠ Pas de modèle lettres")
    
    # Modèle mots
    p = Path("asl_words_model.pkl")
    if p.exists():
        with open(p, "rb") as f:
            _word_model = pickle.load(f)
        print("✓ Modèle MOTS chargé")
    else:
        print("⚠ Pas de modèle mots (niveau intermédiaire désactivé)")


def predict_letter(hand_landmarks):
    """Prédit une lettre depuis les landmarks de main."""
    if not _letter_model:
        return None, 0.0
    
    lm = hand_landmarks.landmark
    wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
    f = []
    for p in lm:
        f.extend([p.x - wx, p.y - wy, p.z - wz])
    
    X = _letter_model["scaler"].transform([f])
    pr = _letter_model["model"].predict_proba(X)[0]
    i = int(np.argmax(pr))
    c = float(pr[i])
    
    if c < 0.45:
        return None, c
    return _letter_model["label_encoder"].inverse_transform([i])[0], c


def extract_holistic_features(results):
    """Extrait les features pour la reconnaissance de mots."""
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
    FACE_KEYPOINTS = [10, 152, 234, 454, 1, 4, 5, 195, 61, 291,
                      13, 14, 33, 133, 362, 263, 70, 300, 94, 323]
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
    
    # Pose (8 points)
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
    
    return features


def extract_sequence_features(sequence):
    """Combine les features de plusieurs frames."""
    if len(sequence) < 2:
        return None
    
    sequence = np.array(sequence)
    combined = []
    combined.extend(np.mean(sequence, axis=0))
    combined.extend(np.std(sequence, axis=0))
    combined.extend(sequence[-1] - sequence[0])
    velocities = np.diff(sequence, axis=0)
    combined.extend(np.mean(np.abs(velocities), axis=0))
    
    return combined


def predict_word(sequence_features):
    """Prédit un mot depuis les features de séquence."""
    if not _word_model or sequence_features is None:
        return None, 0.0
    
    X = _word_model["scaler"].transform([sequence_features])
    pr = _word_model["model"].predict_proba(X)[0]
    i = int(np.argmax(pr))
    c = float(pr[i])
    
    if c < 0.5:
        return None, c
    return _word_model["label_encoder"].inverse_transform([i])[0], c


# ═══════════════════════════════════════
#  DÉTECTION GÉOMÉTRIQUE (fallback lettres)
# ═══════════════════════════════════════
def _fe(lm, h="Right"):
    e = [0] * 5
    e[0] = 1 if (h == "Right" and lm[4].x < lm[3].x) or (h == "Left" and lm[4].x > lm[3].x) else 0
    for i, (a, b) in enumerate([(8, 6), (12, 10), (16, 14), (20, 18)]):
        e[i + 1] = 1 if lm[a].y < lm[b].y else 0
    return e


def _d(lm, i, j):
    return math.hypot(lm[i].x - lm[j].x, lm[i].y - lm[j].y)


def detect_letter_rules(hl, hand="Right"):
    """Détection par règles géométriques (fallback)."""
    lm = hl.landmark
    t, i, m, r, p = _fe(lm, hand)
    d48 = _d(lm, 4, 8)
    d412 = _d(lm, 4, 12)
    d812 = _d(lm, 8, 12)
    
    R = None
    if not i and not m and not r and not p:
        if lm[4].y < lm[3].y and lm[4].x > lm[8].x:
            R = "A"
        elif d48 > 0.06 and d48 < 0.18:
            R = "C"
        elif _d(lm, 4, 6) < 0.07:
            R = "T"
        elif lm[4].y > lm[8].y and lm[4].y > lm[12].y:
            R = "M"
        elif lm[4].y > lm[8].y:
            R = "N"
        elif d48 < 0.08:
            R = "O"
        elif _d(lm, 8, 6) < 0.06:
            R = "X"
        else:
            R = "S"
    elif i and m and r and p and not t:
        R = "B"
    elif i and not m and not r and not p:
        if d412 < 0.06:
            R = "D"
        elif abs(lm[8].y - lm[5].y) < 0.06:
            R = "G"
        elif lm[8].y > lm[6].y and t:
            R = "P"
        elif lm[8].y > lm[5].y and lm[4].y > lm[3].y:
            R = "Q"
        elif t:
            R = "L"
    elif not i and m and r and p and d48 < 0.05:
        R = "F"
    elif i and m and not r and not p:
        if abs(lm[8].y - lm[12].y) < 0.05:
            R = "H"
        elif d812 < 0.04 and not t:
            R = "U"
        elif d812 > 0.05 and not t:
            R = "V"
        elif t:
            R = "K"
        else:
            R = "R"
    elif not i and not m and not r and p and not t:
        R = "I"
    elif not i and not m and not r and p and t:
        R = "Y"
    elif i and m and r and not p and not t:
        R = "W"
    
    return R, 0.0


def detect_letter(hl, hand="Right"):
    """Détecte une lettre (IA + fallback règles)."""
    L, c = predict_letter(hl)
    if L:
        return L, c, "ML"
    L, c = detect_letter_rules(hl, hand)
    return L, c, "règles"


# ═══════════════════════════════════════
#  SKELETONS DE RÉFÉRENCE
# ═══════════════════════════════════════
CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
BASE = {0: (.50, .95), 1: (.45, .80), 5: (.35, .65), 9: (.48, .62), 13: (.60, .64), 17: (.70, .68)}
TIPS = {4, 8, 12, 16, 20}

SK = {
    "A": {**BASE, 2: (.38, .72), 3: (.32, .65), 4: (.35, .60), 6: (.28, .52), 7: (.25, .47), 8: (.25, .44),
          10: (.42, .52), 11: (.40, .45), 12: (.40, .42), 14: (.54, .53), 15: (.52, .47), 16: (.52, .44),
          18: (.65, .57), 19: (.63, .52), 20: (.63, .50)},
    "B": {**BASE, 2: (.38, .72), 3: (.32, .65), 4: (.30, .60), 6: (.28, .44), 7: (.26, .30), 8: (.26, .20),
          10: (.42, .42), 11: (.40, .27), 12: (.40, .17), 14: (.56, .43), 15: (.54, .28), 16: (.54, .18),
          18: (.68, .47), 19: (.66, .32), 20: (.66, .22)},
    # ... (autres lettres comme avant)
}
for _l in LETTERS:
    if _l not in SK:
        SK[_l] = SK["B"]


def draw_skel(img, letter, x, y, w, h, color=CYAN):
    s = SK.get(letter, SK["B"])
    mg = 0.10
    pts = {k: (int(x + mg * w + rx * w * (1 - 2 * mg)), int(y + mg * h + ry * h * (1 - 2 * mg)))
           for k, (rx, ry) in s.items()}
    for a, b in CONN:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], tuple(int(c * .5) for c in color), 2, cv2.LINE_AA)
    for k, pt in pts.items():
        sz = 6 if k == 0 else (5 if k in TIPS else 3)
        cv2.circle(img, pt, sz, GOLD if k in TIPS else color, -1, cv2.LINE_AA)


# ═══════════════════════════════════════
#  HELPERS DESSIN
# ═══════════════════════════════════════
def tx(img, s, x, y, f=FS, sc=0.6, c=WHT, th=1):
    cv2.putText(img, s, (x, y), f, sc, c, th, cv2.LINE_AA)


def txc(img, s, cx, y, f=FD, sc=1.0, c=WHT, th=2):
    w, _ = cv2.getTextSize(s, f, sc, th)[0]
    cv2.putText(img, s, (cx - w // 2, y), f, sc, c, th, cv2.LINE_AA)


def glow(img, s, cx, y, f, sc, c, th=2, r=4):
    w, _ = cv2.getTextSize(s, f, sc, th)[0]
    x = cx - w // 2
    sh = tuple(min(255, int(v * .22)) for v in c)
    cv2.putText(img, s, (x + r, y + r), f, sc, sh, th + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), f, sc, c, th, cv2.LINE_AA)


def bar(img, x, y, w, h, val, mx, c):
    cv2.rectangle(img, (x, y), (x + w, y + h), MID, -1)
    f = int(w * min(val, mx) / mx) if mx else 0
    if f:
        cv2.rectangle(img, (x, y), (x + f, y + h), c, -1)


# ═══════════════════════════════════════
#  CACHE PANNEAU RÉFÉRENCE
# ═══════════════════════════════════════
_ref_cache = {}


def ref_panel_letter(letter):
    """Panneau de référence pour une lettre."""
    key = f"letter_{letter}"
    if key in _ref_cache:
        return _ref_cache[key].copy()
    
    p = np.zeros((CAM_H, HW, 3), dtype=np.uint8)
    p[:] = (18, 12, 32)
    cx = HW // 2
    
    txc(p, "LETTRE A FAIRE", cx, 30, FS, .54, PINK, 1)
    cv2.line(p, (10, 42), (HW - 10, 42), MID, 1)
    glow(p, letter, cx, 148, FD, 5.5, CYAN, 9, 5)
    draw_skel(p, letter, 18, 152, HW - 36, CAM_H - 230, CYAN)
    
    cv2.rectangle(p, (8, CAM_H - 68), (HW - 8, CAM_H - 8), DRK, -1)
    cv2.rectangle(p, (8, CAM_H - 68), (HW - 8, CAM_H - 8), PINK, 1)
    tx(p, "Indice: " + LETTER_HINTS.get(letter, ""), 16, CAM_H - 44, FS, .43, GOLD)
    
    _ref_cache[key] = p.copy()
    return p


def ref_panel_word(word):
    """Panneau de référence pour un mot."""
    key = f"word_{word}"
    if key in _ref_cache:
        return _ref_cache[key].copy()
    
    p = np.zeros((CAM_H, HW, 3), dtype=np.uint8)
    p[:] = (18, 12, 32)
    cx = HW // 2
    
    txc(p, "MOT A SIGNER", cx, 30, FS, .54, ORG, 1)
    cv2.line(p, (10, 42), (HW - 10, 42), MID, 1)
    
    # Afficher le mot
    display_word = word.replace("_", " ")
    glow(p, display_word, cx, 140, FD, 2.8, ORG, 6, 4)
    
    # Icône main + visage
    cv2.circle(p, (cx - 80, 220), 35, CYAN, 2, cv2.LINE_AA)  # Main
    cv2.circle(p, (cx + 80, 220), 35, PINK, 2, cv2.LINE_AA)  # Visage
    txc(p, "MAIN", cx - 80, 270, FS, .4, CYAN, 1)
    txc(p, "VISAGE", cx + 80, 270, FS, .4, PINK, 1)
    txc(p, "+", cx, 225, FD, 1.5, WHT, 2)
    
    # Instruction de mouvement
    txc(p, "Fais le MOUVEMENT complet", cx, 320, FS, .5, WHT, 1)
    
    cv2.rectangle(p, (8, CAM_H - 68), (HW - 8, CAM_H - 8), DRK, -1)
    cv2.rectangle(p, (8, CAM_H - 68), (HW - 8, CAM_H - 8), ORG, 1)
    hint = WORD_HINTS.get(word, "")
    tx(p, hint[:40], 16, CAM_H - 44, FS, .40, GOLD)
    
    _ref_cache[key] = p.copy()
    return p


# ═══════════════════════════════════════
#  ÉTAT DU JEU
# ═══════════════════════════════════════
@dataclass
class ScrollItem:
    text: str
    x: float
    is_current: bool = False
    done: bool = False


@dataclass
class GameState:
    mode: str = "menu"
    difficulty: str = "debutant"
    game_mode: str = "letters"  # "letters" ou "words"
    
    t_item: float = 10.0
    hold_frames: int = 20
    
    queue: List[str] = field(default_factory=list)
    idx: int = 0
    hold: int = 0
    t_start: float = 0.0
    
    cd: int = 3
    cd_start: float = 0.0
    
    correct: int = 0
    wrong: int = 0
    skipped: int = 0
    score: int = 0
    streak: int = 0
    best: int = 0
    rate: float = 0.0
    
    detected: str = ""
    conf: float = 0.0
    det_mode: str = ""
    
    # Buffer pour les mots (séquence de frames)
    frame_buffer: deque = field(default_factory=lambda: deque(maxlen=SEQUENCE_LENGTH))
    word_prediction: str = ""
    word_conf: float = 0.0
    
    scroll: List[ScrollItem] = field(default_factory=list)
    flash_col: tuple = (0, 0, 0)
    flash_t: float = 0.0
    popup: str = ""
    popup_t: float = 0.0
    
    @property
    def current(self):
        return self.queue[self.idx] if self.idx < len(self.queue) else ""
    
    def update_rate(self):
        tot = self.correct + self.wrong + self.skipped
        self.rate = self.correct / tot * 100 if tot else 0.0


def init_scroll(gs):
    gs.scroll = [ScrollItem(item, float(SW + 80 + i * GAP), i == 0) 
                 for i, item in enumerate(gs.queue)]


def start_game(gs, difficulty=None):
    if difficulty:
        gs.difficulty = difficulty
    
    lvl = LEVELS[gs.difficulty]
    gs.queue = list(lvl["items"])
    gs.t_item = lvl["t"]
    gs.hold_frames = lvl["hold"]
    gs.game_mode = lvl["mode"]
    
    gs.idx = gs.hold = gs.correct = gs.wrong = gs.skipped = gs.score = gs.streak = gs.best = 0
    gs.rate = 0.0
    gs.mode = "countdown"
    gs.cd = 3
    gs.cd_start = time.time()
    gs.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    gs.word_prediction = ""
    gs.word_conf = 0.0
    
    init_scroll(gs)


def advance(gs, result):
    elapsed = time.time() - gs.t_start
    
    for sl in gs.scroll:
        if sl.text == gs.current and sl.is_current:
            sl.done = True
            sl.is_current = False
            break
    
    if result == "correct":
        gs.correct += 1
        gs.score += 100 + max(0, int((gs.t_item - elapsed) * 15)) + gs.streak * 10
        gs.streak += 1
        gs.best = max(gs.best, gs.streak)
        gs.flash_col = GRN
        gs.flash_t = time.time()
        gs.popup = "PARFAIT !"
        gs.popup_t = time.time()
    elif result == "wrong":
        gs.wrong += 1
        gs.streak = 0
        gs.flash_col = RED
        gs.flash_t = time.time()
        gs.popup = "RATÉ !"
        gs.popup_t = time.time()
    elif result == "skip":
        gs.skipped += 1
        gs.score = max(0, gs.score - 25)
        gs.streak = 0
    
    gs.update_rate()
    gs.hold = 0
    gs.idx += 1
    gs.frame_buffer.clear()
    
    if gs.idx >= len(gs.queue):
        gs.mode = "result"
        return
    
    for sl in gs.scroll:
        if sl.text == gs.queue[gs.idx] and not sl.done:
            sl.is_current = True
            break
    
    gs.t_start = time.time()


# ═══════════════════════════════════════
#  RENDU — MENU
# ═══════════════════════════════════════
def render_menu(cam, gs):
    s = np.zeros((SH, SW, 3), dtype=np.uint8)
    s[:] = BG
    
    if cam is not None:
        full = cv2.resize(cam, (SW, SH))
        cv2.addWeighted(full, .10, s, .90, 0, s)
    
    cx = SW // 2
    
    glow(s, "ASL", cx, 120, FD, 4.5, CYAN, 9, 16)
    glow(s, "JUST DANCE", cx, 190, FD, 1.8, PINK, 3, 7)
    cv2.line(s, (100, 208), (SW - 100, 208), MID, 1)
    txc(s, "Apprends la langue des signes americaine", cx, 240, FS, .55, WHT, 1)
    
    txc(s, "CHOISIS TON NIVEAU", cx, 280, FD, .75, GOLD, 1)
    
    # Boutons niveaux
    btn_data = [
        ("debutant", "1", cx - 330, 305),
        ("intermediaire", "2", cx - 110, 305),
        ("expert", "3", cx + 110, 305),
    ]
    
    for key, num, bx, by in btn_data:
        lvl = LEVELS[key]
        col = lvl["color"]
        selected = (gs.difficulty == key)
        
        # Vérifier si le modèle mots existe pour intermédiaire
        disabled = (key == "intermediaire" and _word_model is None)
        
        bg_col = tuple(int(c * .3) for c in col) if selected else DRK
        if disabled:
            bg_col = (30, 30, 30)
            col = GRY
        
        border = 3 if selected else 1
        cv2.rectangle(s, (bx, by), (bx + 210, by + 140), bg_col, -1)
        cv2.rectangle(s, (bx, by), (bx + 210, by + 140), col, border)
        
        if selected and not disabled:
            cv2.rectangle(s, (bx, by), (bx + 210, by + 5), col, -1)
        
        glow(s, num, bx + 105, by + 55, FD, 2.2, col, 4, 4)
        txc(s, lvl["label"], bx + 105, by + 85, FD, .55, col, 1)
        
        # Description
        desc = lvl["description"]
        txc(s, desc, bx + 105, by + 108, FS, .32, WHT if not disabled else GRY, 1)
        
        if disabled:
            txc(s, "(entrainer d'abord)", bx + 105, by + 128, FS, .28, RED, 1)
        else:
            txc(s, f"[{num}]", bx + 105, by + 128, FS, .35, GRY, 1)
    
    # Bouton JOUER
    bx_play = cx - 100
    cv2.rectangle(s, (bx_play, 465), (bx_play + 200, 525), DRK, -1)
    cv2.rectangle(s, (bx_play, 465), (bx_play + 200, 525), GRN, 2)
    glow(s, "JOUER", cx, 502, FD, .9, GRN, 2, 3)
    txc(s, "[ESPACE]", cx, 520, FS, .38, WHT, 1)
    
    txc(s, "[Q] Quitter", cx, 555, FS, .45, GRY, 1)
    
    # Info niveau
    lvl_sel = LEVELS[gs.difficulty]
    mode_text = "MOTS" if lvl_sel["mode"] == "words" else "LETTRES"
    txc(s, f"{lvl_sel['label']} • {len(lvl_sel['items'])} {mode_text} • {lvl_sel['t']:.0f}s",
        cx, 590, FS, .44, lvl_sel["color"], 1)
    
    return s


# ═══════════════════════════════════════
#  RENDU — COUNTDOWN
# ═══════════════════════════════════════
def render_countdown(cam, gs):
    s = np.zeros((SH, SW, 3), dtype=np.uint8)
    
    if cam is not None:
        full = cv2.resize(cam, (SW, SH))
        cv2.addWeighted(full, .25, s, .75, 0, s)
    
    cx = SW // 2
    
    mode_text = "MOTS" if gs.game_mode == "words" else "LETTRES"
    glow(s, f"MODE {mode_text}", cx, SH // 2 - 150, FD, 1.2, ORG if gs.game_mode == "words" else CYAN, 2, 5)
    glow(s, "PRÉPAREZ-VOUS !", cx, SH // 2 - 100, FD, 1.5, CYAN, 3, 7)
    
    n = gs.cd
    col = GRN if n == 1 else (ORG if n == 2 else PINK)
    glow(s, str(n), cx, SH // 2 + 70, FD, 8.0, col, 14, 24)
    
    if gs.queue:
        first = gs.queue[0].replace("_", " ")
        txc(s, f"Premier : {first}", cx, SH // 2 + 148, FS, .68, WHT, 1)
    
    return s


# ═══════════════════════════════════════
#  RENDU — JEU
# ═══════════════════════════════════════
def render_playing(cam, gs, holistic_results=None):
    s = np.zeros((SH, SW, 3), dtype=np.uint8)
    s[:] = BG
    
    item = gs.current
    elapsed = time.time() - gs.t_start
    rem = max(0, gs.t_item - elapsed)
    ratio = rem / gs.t_item
    fc = GRN if ratio > .5 else (ORG if ratio > .25 else RED)
    
    # ── Gauche : caméra ──────────────────────────────────
    if cam is not None:
        s[0:CAM_H, 0:HW] = cam
    else:
        cv2.rectangle(s, (0, 0), (HW, CAM_H), (18, 12, 36), -1)
        txc(s, "Camera...", HW // 2, CAM_H // 2, FS, .7, GRY, 1)
    
    cv2.rectangle(s, (2, 2), (HW - 2, CAM_H - 2), fc, 3)
    
    # HUD détection
    cv2.rectangle(s, (6, 6), (338, 68), (0, 0, 0), -1)
    cv2.rectangle(s, (6, 6), (338, 68), fc, 1)
    
    if gs.game_mode == "letters":
        det = gs.detected
        dc = GRN if det == item else (RED if det else GRY)
        badge = "IA" if gs.det_mode == "ML" else "reg"
        tx(s, f"{badge} LETTRE:", 16, 27, FS, .46, WHT)
        tx(s, det or "---", 160, 33, FD, .95, dc, 2)
    else:
        det = gs.word_prediction
        dc = GRN if det == item else (ORG if det else GRY)
        tx(s, "MOT:", 16, 27, FS, .46, WHT)
        display = det.replace("_", " ") if det else "---"
        tx(s, display[:12], 80, 33, FD, .75, dc, 2)
    
    if gs.conf > 0:
        cp = int(gs.conf * 100)
        cc = GRN if cp > 70 else (ORG if cp > 45 else RED)
        tx(s, f"{cp}%", 283, 27, FS, .38, cc)
    
    tx(s, f"Score:{gs.score}  x{gs.streak}  {gs.rate:.0f}%", 16, 58, FS, .42, GOLD)
    
    # Indicateur buffer pour mots
    if gs.game_mode == "words":
        buf_fill = len(gs.frame_buffer) / SEQUENCE_LENGTH
        bar(s, 6, 70, 150, 8, buf_fill, 1.0, PINK)
        tx(s, "Buffer", 160, 77, FS, .3, GRY)
    
    # Barre maintien
    hr = gs.hold / gs.hold_frames
    bc = GRN if hr > .6 else (ORG if hr > .3 else GRY)
    cv2.rectangle(s, (4, CAM_H - 22), (HW - 4, CAM_H - 4), DRK, -1)
    fw = int((HW - 8) * hr)
    if fw:
        cv2.rectangle(s, (4, CAM_H - 22), (4 + fw, CAM_H - 4), bc, -1)
    txc(s, f"MAINTENEZ {int(hr * 100)}%", HW // 2, CAM_H - 7, FS, .4, WHT, 1)
    
    # ── Droite : panneau référence ───────────────────────
    if gs.game_mode == "letters":
        ref = ref_panel_letter(item)
    else:
        ref = ref_panel_word(item)
    
    tcy = CAM_H - 108
    cv2.rectangle(ref, (HW // 2 - 82, tcy - 26), (HW // 2 + 82, tcy + 12), (18, 12, 32), -1)
    txc(ref, f"{rem:.1f}s", HW // 2, tcy, FD, 1.1, fc, 2)
    bar(ref, 52, tcy + 4, HW - 104, 7, rem, gs.t_item, fc)
    tx(ref, f"{gs.idx + 1}/{len(gs.queue)} | {gs.correct}ok {gs.wrong}x {gs.skipped}->",
       10, CAM_H - 20, FS, .37, GRY)
    
    s[0:CAM_H, HW:SW] = ref
    cv2.line(s, (HW, 0), (HW, CAM_H), MID, 2)
    
    # ── Bande défilement ─────────────────────────────────
    sy = CAM_H
    cv2.rectangle(s, (0, sy), (SW, SH), (8, 4, 16), -1)
    cv2.line(s, (0, sy), (SW, sy), PINK, 2)
    
    cv2.rectangle(s, (TX - 55, sy + 3), (TX + 55, SH - 3), (28, 12, 50), -1)
    cv2.rectangle(s, (TX - 55, sy + 3), (TX + 55, SH - 3), PINK, 2)
    txc(s, "ICI", TX, sy + 20, FS, .44, PINK, 1)
    
    bar(s, TX + 64, sy + 5, SW - TX - 72, 9, gs.idx, len(gs.queue), PINK)
    tx(s, f"{gs.idx + 1}/{len(gs.queue)}", TX + 64, sy + 23, FS, .34, GRY)
    tx(s, "[S]Passer [M]Menu", SW - 188, sy + 18, FS, .37, GRY)
    
    scy = sy + SCR_H // 2 + 12
    for sl in gs.scroll:
        sx = int(sl.x)
        if sx < -80 or sx > SW + 50:
            continue
        
        display_text = sl.text.replace("_", " ")
        
        if sl.is_current:
            cv2.rectangle(s, (sx - 60, scy - 65), (sx + 60, scy + 8), (0, 25, 45), -1)
            cv2.rectangle(s, (sx - 60, scy - 65), (sx + 60, scy + 8), 
                         ORG if gs.game_mode == "words" else CYAN, 1)
            glow(s, display_text[:8], sx, scy, FD, 1.8 if gs.game_mode == "words" else 2.8,
                 ORG if gs.game_mode == "words" else CYAN, 3 if gs.game_mode == "words" else 5, 4)
        elif sl.done:
            tx(s, display_text[:8], sx - 20, scy + 8, FD, .6, GRY, 1)
        else:
            dr = sl.x - TX
            fade = max(.15, 1.0 - dr / (SW * .6))
            sc = max(.45, 1.4 * fade)
            sh = int(195 * fade)
            txc(s, display_text[:8], sx, scy, FD, sc, (sh // 4, sh // 3, sh), max(1, int(sc)))
    
    # Flash
    if time.time() - gs.flash_t < .3:
        ov = s[0:CAM_H].copy()
        cv2.rectangle(ov, (0, 0), (SW, CAM_H), gs.flash_col, -1)
        cv2.addWeighted(ov, .18, s[0:CAM_H], .82, 0, s[0:CAM_H])
    
    # Popup
    if time.time() - gs.popup_t < .6 and gs.popup:
        pc = GRN if "PARFAIT" in gs.popup else RED
        glow(s, gs.popup, HW // 2, CAM_H // 2, FD, 3.2, pc, 7, 6)
    
    return s


# ═══════════════════════════════════════
#  RENDU — RÉSULTATS
# ═══════════════════════════════════════
def render_result(gs):
    s = np.zeros((SH, SW, 3), dtype=np.uint8)
    s[:] = BG
    cx = SW // 2
    
    mode_text = "MOTS" if gs.game_mode == "words" else "LETTRES"
    glow(s, f"RÉSULTATS - {mode_text}", cx, 75, FD, 1.8, GOLD, 5, 8)
    cv2.line(s, (80, 95), (SW - 80, 95), MID, 1)
    
    rt = gs.rate
    rc = GRN if rt >= 70 else (ORG if rt >= 40 else RED)
    
    glow(s, str(gs.score), cx - 195, 205, FD, 3.3, GOLD, 6, 7)
    txc(s, "SCORE", cx - 195, 242, FS, .5, GOLD, 1)
    glow(s, f"{rt:.0f}%", cx + 195, 205, FD, 3.3, rc, 6, 7)
    txc(s, "RÉUSSITE", cx + 195, 242, FS, .5, rc, 1)
    
    stats = [
        (f"Corrects : {gs.correct}", GRN),
        (f"Ratés    : {gs.wrong}", RED),
        (f"Passés   : {gs.skipped}", ORG),
        (f"Meilleure série : {gs.best}", CYAN),
    ]
    
    for i, (st, c) in enumerate(stats):
        cv2.rectangle(s, (cx - 255, 268 + i * 58), (cx + 255, 318 + i * 58), DRK, -1)
        cv2.rectangle(s, (cx - 255, 268 + i * 58), (cx + 255, 318 + i * 58), c, 1)
        txc(s, st, cx, 299 + i * 58, FS, .62, c, 1)
    
    if rt >= 80:
        gr, gc = "EXCELLENT !", GOLD
    elif rt >= 60:
        gr, gc = "BIEN JOUÉ !", GRN
    elif rt >= 40:
        gr, gc = "CONTINUE !", CYAN
    else:
        gr, gc = "ENTRAÎNE-TOI !", ORG
    
    glow(s, gr, cx, 510, FD, 1.2, gc, 2, 5)
    
    cv2.rectangle(s, (cx - 270, 540), (cx - 30, 600), DRK, -1)
    cv2.rectangle(s, (cx - 270, 540), (cx - 30, 600), GRN, 2)
    txc(s, "REJOUER [ESPACE]", cx - 150, 576, FS, .58, GRN, 1)
    
    cv2.rectangle(s, (cx + 30, 540), (cx + 270, 600), DRK, -1)
    cv2.rectangle(s, (cx + 30, 540), (cx + 270, 600), WHT, 1)
    txc(s, "MENU [M]", cx + 150, 576, FS, .58, WHT, 1)
    
    return s


# ═══════════════════════════════════════
#  THREAD CAMÉRA
# ═══════════════════════════════════════
class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, HW * 2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
    
    def _run(self):
        while self.running:
            ret, f = self.cap.read()
            if ret:
                f = cv2.flip(f, 1)
                f = cv2.resize(f, (HW, CAM_H))
                with self.lock:
                    self.frame = f
    
    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
    
    def stop(self):
        self.running = False
        self.cap.release()


# ═══════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════
def main():
    cam = Camera()
    
    # Détecteur pour lettres (main seule)
    mp_hands_mod = mp.solutions.hands
    hands_detector = mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=.7,
        min_tracking_confidence=.5,
    )
    
    # Détecteur holistique pour mots (main + visage)
    mp_holistic = mp.solutions.holistic
    holistic_detector = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    mp_drawing = mp.solutions.drawing_utils
    stl = mp_drawing.DrawingSpec(color=CYAN, thickness=2, circle_radius=3)
    stc = mp_drawing.DrawingSpec(color=PINK, thickness=1)
    
    gs = GameState()
    load_models()
    
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, SW, SH)
    
    # Pré-calcul des panneaux
    for l in LETTERS:
        ref_panel_letter(l)
    for w in WORDS:
        ref_panel_word(w)
    
    current_frame = None
    holistic_results = None
    
    while True:
        raw = cam.get()
        
        if raw is not None:
            current_frame = raw.copy()
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            
            gs.detected = ""
            gs.conf = 0.0
            gs.det_mode = ""
            
            if gs.mode == "playing":
                if gs.game_mode == "letters":
                    # Détection lettres (main seule)
                    res = hands_detector.process(rgb)
                    if res.multi_hand_landmarks:
                        hl = res.multi_hand_landmarks[0]
                        hn = "Right"
                        if res.multi_handedness:
                            hn = res.multi_handedness[0].classification[0].label
                        L, c, m = detect_letter(hl, hn)
                        gs.detected = L or ""
                        gs.conf = c
                        gs.det_mode = m
                        mp_drawing.draw_landmarks(current_frame, hl, mp_hands_mod.HAND_CONNECTIONS, stl, stc)
                
                else:
                    # Détection mots (holistique)
                    holistic_results = holistic_detector.process(rgb)
                    
                    # Dessiner les landmarks
                    if holistic_results.face_landmarks:
                        mp_drawing.draw_landmarks(
                            current_frame, holistic_results.face_landmarks,
                            mp_holistic.FACEMESH_CONTOURS,
                            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(80, 200, 120), thickness=1),
                        )
                    if holistic_results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            current_frame, holistic_results.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS, stl, stc,
                        )
                    if holistic_results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            current_frame, holistic_results.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS, stl, stc,
                        )
                    
                    # Extraire et accumuler les features
                    features = extract_holistic_features(holistic_results)
                    gs.frame_buffer.append(features)
                    
                    # Prédire quand le buffer est plein
                    if len(gs.frame_buffer) >= SEQUENCE_LENGTH:
                        seq_features = extract_sequence_features(list(gs.frame_buffer))
                        word, conf = predict_word(seq_features)
                        gs.word_prediction = word or ""
                        gs.word_conf = conf
                        gs.conf = conf
        
        # Logique de jeu
        if gs.mode == "countdown":
            elapsed = time.time() - gs.cd_start
            gs.cd = max(1, math.ceil(3 - elapsed))
            if elapsed >= 3.0:
                gs.mode = "playing"
                gs.t_start = time.time()
        
        elif gs.mode == "playing" and gs.current:
            # Défilement
            for sl in gs.scroll:
                if not sl.done:
                    sl.x -= SPD
            
            elapsed = time.time() - gs.t_start
            
            if elapsed >= gs.t_item:
                advance(gs, "wrong")
            else:
                # Vérification selon le mode
                if gs.game_mode == "letters":
                    if gs.detected == gs.current:
                        gs.hold = min(gs.hold + 1, gs.hold_frames)
                        if gs.hold >= gs.hold_frames:
                            advance(gs, "correct")
                    else:
                        gs.hold = max(0, gs.hold - 1)
                else:
                    # Mode mots
                    if gs.word_prediction == gs.current:
                        gs.hold = min(gs.hold + 1, gs.hold_frames)
                        if gs.hold >= gs.hold_frames:
                            advance(gs, "correct")
                    else:
                        gs.hold = max(0, gs.hold - 1)
        
        # Rendu
        if gs.mode == "menu":
            screen = render_menu(current_frame, gs)
        elif gs.mode == "countdown":
            screen = render_countdown(current_frame, gs)
        elif gs.mode == "playing":
            screen = render_playing(current_frame, gs, holistic_results)
        elif gs.mode == "result":
            screen = render_result(gs)
        else:
            screen = np.zeros((SH, SW, 3), dtype=np.uint8)
        
        cv2.imshow(WIN, screen)
        k = cv2.waitKey(1) & 0xFF
        
        if k == ord('q'):
            break
        elif k == ord('m'):
            gs.mode = "menu"
            gs.hold = 0
        elif k == ord('1') and gs.mode == "menu":
            gs.difficulty = "debutant"
        elif k == ord('2') and gs.mode == "menu" and _word_model is not None:
            gs.difficulty = "intermediaire"
        elif k == ord('3') and gs.mode == "menu":
            gs.difficulty = "expert"
        elif k in (ord(' '), 13):
            if gs.mode in ("menu", "result"):
                start_game(gs)
        elif k == ord('s') and gs.mode == "playing":
            advance(gs, "skip")
    
    cam.stop()
    hands_detector.close()
    holistic_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
