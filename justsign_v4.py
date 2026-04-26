"""
JUST SIGN v4 — Modern Edition
=====================================
MODES:
  1  BEGINNER     — Letters A B C I L
  2  INTERMEDIATE — ASL words (requires asl_words_model.pkl)
  3  EXPERT       — Full alphabet, fast
  4  SONG MODE    — Bohemian Rhapsody (optional: --mp3 file.mp3)
  5  LEARN LETTERS — Step-by-step alphabet training
  6  LEARN WORDS   — Step-by-step ASL word training

LAUNCH:
  py -3.11 justsign_v4.py
  py -3.11 justsign_v4.py --mp3 bohemian.mp3

KEYS:
  SPACE / ENTER  → Start / Next
  S              → Skip (game modes)
  ← arrow        → Previous (training modes)
  V              → Watch video (training modes)
  P              → Pause
  M / ESC        → Back to menu
  Q              → Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import time, math, pickle, threading, argparse, webbrowser, random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from collections import deque

# ──────────────────────────────────────────────────────────
#  OPTIONAL IMPORTS
# ──────────────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    _HAS_PYGAME = True
    print("✓ pygame ready")
except Exception:
    _HAS_PYGAME = False
    print("⚠  pygame not available (pip install pygame)")

try:
    import pyttsx3
    _tts = pyttsx3.init()
    _tts.setProperty('rate', 155)
    _tts.setProperty('volume', 0.9)
    _HAS_TTS = True
    _tts_lock = threading.Lock()
    print("✓ TTS ready")
except Exception:
    _HAS_TTS = False

# ──────────────────────────────────────────────────────────
#  SCREEN CONSTANTS
# ──────────────────────────────────────────────────────────
SW, SH  = 1080, 720
HW      = SW // 2
CAM_H   = 490
BOT_H   = SH - CAM_H
WIN     = "JUST SIGN v4"

# ──────────────────────────────────────────────────────────
#  COLOUR PALETTE — matched to JUST SIGN logo (BGR)
#  Logo: hot pink / purple / gold / deep navy background
# ──────────────────────────────────────────────────────────
BG     = ( 10,  5,  20)      # deep navy (logo background)
BG2    = ( 18, 10,  36)
PANEL  = ( 26, 15,  48)
EDGE   = ( 80, 40, 130)
CYAN   = ( 20, 210, 255)     # hot pink/magenta → stays as accent
PINK   = ( 80,  30, 255)     # logo hot pink (BGR: low B, low G, high R)
GRN    = ( 30, 210,  90)
ORG    = (  0, 140, 255)
RED    = ( 40,  40, 230)
GOLD   = ( 20, 190, 255)     # gold from disco ball
WHT    = (240, 240, 255)
DRK    = ( 20, 12,  40)
MID    = ( 50, 28,  85)
GRY    = ( 90, 75, 115)
PURP   = (190,  50, 220)     # logo purple
TEAL   = (180, 200,  30)
LOGO_PINK = (80,  30, 255)   # exact logo pink
LOGO_PURP = (190, 50, 220)   # exact logo purple
DISCO  = ( 10, 190, 255)     # disco gold shimmer
FD     = cv2.FONT_HERSHEY_DUPLEX
FS     = cv2.FONT_HERSHEY_SIMPLEX

# ──────────────────────────────────────────────────────────
#  LOGO LOADER
# ──────────────────────────────────────────────────────────
_LOGO_PATH = Path(__file__).parent / "logo.png"
_LOGO: dict = {}   # cache of pre-resized logos

def _load_logo(size_key, w, h, circle=False):
    """Load and cache logo at a given size. Returns BGR ndarray or None."""
    if size_key in _LOGO:
        return _LOGO[size_key]
    src = _LOGO_PATH
    if not src.exists():
        # Try same folder as CWD
        src = Path("logo.png")
    if not src.exists():
        _LOGO[size_key] = None
        return None
    img = cv2.imread(str(src))
    if img is None:
        _LOGO[size_key] = None
        return None
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if circle:
        mask = np.zeros((h,w), np.uint8)
        cv2.circle(mask, (w//2, h//2), min(w,h)//2-2, 255, -1)
        img = cv2.bitwise_and(img, img, mask=mask)
    _LOGO[size_key] = img
    return img

def overlay_logo(canvas, logo_img, cx, cy, alpha=1.0):
    """Paste logo centred at (cx,cy). Handles boundaries."""
    if logo_img is None: return
    h, w = logo_img.shape[:2]
    x0, y0 = cx - w//2, cy - h//2
    x1, y1 = x0+w, y0+h
    cH, cW = canvas.shape[:2]
    # Clamp
    lx0 = max(0,-x0); ly0 = max(0,-y0)
    lx1 = w - max(0, x1-cW); ly1 = h - max(0, y1-cH)
    x0=max(0,x0); y0=max(0,y0); x1=min(cW,x1); y1=min(cH,y1)
    if x1<=x0 or y1<=y0: return
    patch = logo_img[ly0:ly1, lx0:lx1]
    if alpha >= 1.0:
        canvas[y0:y1, x0:x1] = patch
    else:
        roi = canvas[y0:y1, x0:x1].astype(np.float32)
        roi = roi*(1-alpha) + patch.astype(np.float32)*alpha
        canvas[y0:y1, x0:x1] = roi.astype(np.uint8)

def disco_shimmer(img, t_now, intensity=0.06):
    """Add subtle pulsing disco shimmer overlay."""
    h, w = img.shape[:2]
    shimmer = np.zeros((h,w,3), np.float32)
    # Horizontal scan line sweep
    sweep_y = int((t_now % 2.0) / 2.0 * h)
    for dy in range(-3,4):
        y = (sweep_y+dy) % h
        alpha2 = intensity * (1.0 - abs(dy)/4.0)
        shimmer[y,:] = [v*alpha2 for v in DISCO]
    img[:] = np.clip(img.astype(np.float32) + shimmer, 0, 255).astype(np.uint8)

# ──────────────────────────────────────────────────────────
#  GAME DATA
# ──────────────────────────────────────────────────────────
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
WORDS   = ["HELLO","THANK_YOU","PLEASE","SORRY","YES","NO",
           "HELP","LOVE","FRIEND","FAMILY","EAT","DRINK",
           "WATER","NAME","GOOD","BAD"]

LEVELS = {
    "debutant":      {"label":"BEGINNER",     "mode":"letters","items":list("ABCIL"), "t":14.0,"hold":15,"color":GRN,  "icon":"1","desc":"5 letters - 14s each"},
    "intermediaire": {"label":"INTERMEDIATE", "mode":"words",  "items":WORDS[:8],     "t":8.0, "hold":25,"color":ORG,  "icon":"2","desc":"8 words - 8s each"},
    "expert":        {"label":"EXPERT",       "mode":"letters","items":LETTERS,        "t":6.0, "hold":25,"color":RED,  "icon":"3","desc":"Full alphabet - 6s"},
}

SONG_DETECT_SEC = 4.0

SONG_LYRICS = [
    {"text":"Is this the real life?",         "sign":"REAL",         "t":1.0},
    {"text":"Is this just fantasy?",          "sign":"DREAM",        "t":6.0},
    {"text":"Caught in a landslide",          "sign":"STUCK",        "t":11.0},
    {"text":"No escape from reality",         "sign":"ESCAPE",       "t":16.0},
    {"text":"Open your eyes",                 "sign":"LOOK",         "t":21.0},
    {"text":"Look up to the skies",           "sign":"LOOK",         "t":26.0},
    {"text":"I'm just a poor boy",            "sign":"POOR",         "t":31.0},
    {"text":"I need no sympathy",             "sign":"PITY",         "t":36.0},
    {"text":"Easy come, easy go",             "sign":"HAPPEN",       "t":41.0},
    {"text":"Little high, little low",        "sign":"SOMETIMES",    "t":46.0},
    {"text":"Anyway the wind blows",          "sign":"WHATEVER",     "t":51.0},
    {"text":"Doesn't really matter to me",   "sign":"DON'T-MATTER","t":56.0},
    {"text":"Mama,",                          "sign":"MOTHER",       "t":62.0},
    {"text":"just killed a man",              "sign":"KILL-HIM",     "t":67.0},
    {"text":"Put a gun against his head",     "sign":"PAIN",         "t":72.0},
    {"text":"now he's dead",                  "sign":"DEATH",        "t":79.0},
    {"text":"Mama, life had just begun",      "sign":"LIFE",         "t":87.0},
    {"text":"But now I've gone",              "sign":"LEAVE",        "t":92.0},
    {"text":"and thrown it all away",         "sign":"RUIN",         "t":97.0},
    {"text":"Mama, ooh",                      "sign":"MOTHER",       "t":107.0},
    {"text":"Didn't mean to make you cry",    "sign":"CRY",          "t":112.0},
    {"text":"If I'm not back again tomorrow", "sign":"CONTINUE",     "t":117.0},
    {"text":"Carry on, carry on",             "sign":"CONTINUE",     "t":122.0},
    {"text":"as if nothing really matters",   "sign":"DON'T-MATTER","t":127.0},
    {"text":"Too late, my time has come",     "sign":"LATE",         "t":150.0},
    {"text":"Sends shivers down my spine",    "sign":"AFRAID",       "t":155.0},
    {"text":"Body's aching all the time",     "sign":"PAIN",         "t":160.0},
    {"text":"Goodbye, everybody",             "sign":"GOODBYE",      "t":165.0},
    {"text":"I've got to go",                 "sign":"LEAVE",        "t":170.0},
    {"text":"Gotta leave you all behind",     "sign":"RUN-AWAY",     "t":175.0},
    {"text":"and face the truth",             "sign":"TRUTH",        "t":180.0},
    {"text":"Mama, ooh",                      "sign":"MOTHER",       "t":188.0},
    {"text":"I don't want to die",            "sign":"NOT-WANT",     "t":193.0},
    {"text":"I sometimes wish I'd never",     "sign":"SOMETIMES",    "t":198.0},
    {"text":"been born at all",               "sign":"BORN",         "t":203.0},
    {"text":"I see a little silhouette",      "sign":"LOOK",         "t":213.0},
    {"text":"Scaramouche, will you do the",   "sign":"DANCE",        "t":217.0},
    {"text":"Thunderbolts and lightning",     "sign":"THUNDER",      "t":221.0},
    {"text":"Very, very frightening me",      "sign":"FEAR",         "t":225.0},
    {"text":"nobody loves me",                "sign":"LOVE-ME",      "t":233.0},
    {"text":"Bismillah! No!",                 "sign":"GOD",          "t":241.0},
    {"text":"We will not let you go",         "sign":"FORBID",       "t":244.0},
    {"text":"Let him go!",                    "sign":"LEAVE",        "t":247.0},
    {"text":"Beelzebub has a devil",          "sign":"DEVIL",        "t":251.0},
    {"text":"put aside for me",               "sign":"PRESERVE",     "t":255.0},
    {"text":"So you think you can stone me",  "sign":"PAIN",         "t":260.0},
    {"text":"and spit in my eye",             "sign":"ME",           "t":265.0},
    {"text":"So you think you can love me",   "sign":"LOVE-ME",      "t":269.0},
    {"text":"and leave me to die",            "sign":"DIE",          "t":274.0},
    {"text":"Oh baby, can't do this to me",   "sign":"NO",           "t":279.0},
    {"text":"Just gotta get out",             "sign":"ESCAPE",       "t":284.0},
    {"text":"Nothing really matters",         "sign":"NONE",         "t":305.0},
    {"text":"Anyone can see",                 "sign":"LOOK",         "t":311.0},
    {"text":"Nothing really matters",         "sign":"DON'T-MATTER","t":317.0},
    {"text":"Nothing really matters to me",   "sign":"SAD",          "t":323.0},
    {"text":"Anyway the wind blows...",       "sign":"WHATEVER",     "t":333.0},
]

LETTER_HINTS = {
    "A":"Fist, thumb on side",       "B":"4 fingers up, thumb folded",
    "C":"Curved C shape",             "D":"Index up, others in circle",
    "E":"Fingers bent like a claw",   "F":"Thumb-index circle, 3 fingers up",
    "G":"Index+thumb horizontal",     "H":"Index+middle horizontal",
    "I":"Pinky only",                 "K":"Index+middle V, thumb between",
    "L":"Index up + thumb out",       "M":"Thumb under 3 fingers",
    "N":"Thumb under 2 fingers",      "O":"All fingers in O shape",
    "P":"Index pointing down",        "Q":"Index+thumb pointing down",
    "R":"Crossed index+middle",       "S":"Fist, thumb over fingers",
    "T":"Thumb between index+middle", "U":"Index+middle together up",
    "V":"Index+middle V shape",       "W":"3 fingers spread",
    "X":"Hooked index finger",        "Y":"Thumb+pinky out",
}

WORD_HINTS = {
    "HELLO":"Open hand, forehead → out",   "THANK_YOU":"Flat hand, chin → forward",
    "PLEASE":"Hand circle on chest",       "SORRY":"A-fist circle on chest",
    "YES":"Fist nodding",                  "NO":"Index+middle snap to thumb",
    "HELP":"Fist on palm, lift up",        "LOVE":"Arms crossed on chest",
    "FRIEND":"Hooked index fingers link",  "FAMILY":"F-hands make circle",
    "EAT":"Pinched hand to mouth",         "DRINK":"C-hand to mouth, tilt",
    "WATER":"W taps chin twice",           "NAME":"H-hands cross each other",
    "GOOD":"Flat hand chin → forward",     "BAD":"Hand mouth → flip down",
}

SIGN_HINTS = {
    "REAL":"Index from lips forward",          "DREAM":"Bent index from temple out",
    "STUCK":"V-hand blocked under left hand",  "ESCAPE":"Index escapes from fist",
    "LOOK":"V at eyes then toward object",     "POOR":"Right hand slides under left elbow",
    "PITY":"Middle finger circles on chest",   "HAPPEN":"Two index fingers pivot down",
    "SOMETIMES":"Index circles then S",        "WHATEVER":"Two W-hands spread apart",
    "DON'T-MATTER":"Flat hand waves dismissively","MOTHER":"Thumb taps chin",
    "KILL-HIM":"Index slices under palm",      "DEATH":"Flat hand flips over",
    "LIFE":"Two L-hands rise up torso",        "LEAVE":"Open hand sweeps outward",
    "RUIN":"Two R-hands rub downward",         "CRY":"Index fingers trace tears on cheeks",
    "CONTINUE":"Two thumbs slide forward",     "LATE":"Hand behind shoulder",
    "AFRAID":"Crossed hands spring apart",     "PAIN":"Index fingertips touch painfully",
    "GOODBYE":"Open hand waves goodbye",       "TRUTH":"Index from lips forward",
    "NOT-WANT":"Claw hands turn outward",      "BORN":"Flat hand glides from belly forward",
    "DANCE":"Two fingers dance on palm",       "THUNDER":"T then index zigzags",
    "FEAR":"Spread fingers tremble at chest",  "LOVE-ME":"Arms crossed + point at self",
    "GOD":"Open hand arcs from head down",     "FORBID":"Fist strikes open palm",
    "DEVIL":"Y-hands at temples like horns",   "PRESERVE":"Two P-hands move down together",
    "ME":"Index points to own chest",          "NO":"Index+middle snap to thumb",
    "NONE":"Two O-hands spread apart",         "SAD":"Two open hands lower past face",
    "RUN-AWAY":"Fingers run then sweep out",   "DIE":"Flat hand flips, palm faces down",
}

LETTER_STEPS = {
    "A":["Close 4 fingers into a fist","Thumb rests on the side (not on top)","No finger pointing up"],
    "B":["4 fingers extended straight up","Pressed together, palm forward","Thumb folded across the palm"],
    "C":["Curve all fingers into an arc","Thumb curved facing fingers","Open C shape on the side"],
    "D":["Index finger up and straight","Middle+ring+pinky form a circle","Thumb joins the circle"],
    "E":["All fingers bent toward the palm","Thumb tucked under the fingers","Hand looks like a closed claw"],
    "F":["Thumb and index form a small circle","3 other fingers extended upward","Circle facing forward"],
    "G":["Index points horizontally","Thumb also horizontal, parallel","Like pointing but sideways"],
    "H":["Index and middle extended side by side","Both pointing horizontally","Other fingers and thumb folded"],
    "I":["Only the pinky is raised","All other fingers folded down","Thumb folded too"],
    "K":["Index and middle spread in a V","Thumb placed between the two","Like K shape, not a V"],
    "L":["Index pointing straight up","Thumb pointing out horizontally","Makes a perfect L shape"],
    "M":["Thumb under first 3 fingers","Index+middle+ring folded over","Pinky folded too"],
    "N":["Thumb under index and middle only","Only 2 fingers folded over","Fewer fingers than M"],
    "O":["All fingers curved inward","Thumb and fingers meet at tips","Forms an oval O"],
    "P":["Like K but rotated downward","Index points toward the floor","Thumb horizontal, middle points down"],
    "Q":["Index and thumb point downward","Pinching gesture toward the floor","Like G but pointing down"],
    "R":["Index and middle crossed over each other","Like crossing fingers for luck","Other fingers folded"],
    "S":["Closed fist","Thumb placed over the folded fingers","Different from A: thumb on top"],
    "T":["Thumb pushed between index and middle","Thumb visible sticking out","Half-closed fist"],
    "U":["Index and middle extended together","Side by side pointing up","Not spread apart like V"],
    "V":["Index and middle in a V shape","Spread like a peace sign","Other fingers folded"],
    "W":["Index+middle+ring extended","Spread out like a fan of 3","Makes a W shape"],
    "X":["Index finger bent into a hook","Other fingers folded","Like beckoning someone"],
    "Y":["Thumb out to the side","Pinky extended downward","Other 3 fingers folded - like a phone"],
}

WORD_STEPS = {
    "HELLO":    ["Open hand, palm facing out","Bring hand up to forehead","Sweep outward like a salute"],
    "THANK_YOU":["Flat hand, fingers together","Touch fingers to chin","Move hand forward and down"],
    "PLEASE":   ["Flat hand on chest","Make a large circle on chest","Circular rubbing motion"],
    "SORRY":    ["Make an A fist","Place it on your chest","Rotate it in circles"],
    "YES":      ["Make a fist","Nod the fist up and down","Like nodding your head with your hand"],
    "NO":       ["Index and middle extended","Snap them down onto the thumb","Quick snap of two fingers"],
    "HELP":     ["Place fist on open palm","Lift both hands together","Fist on palm = asking for help"],
    "LOVE":     ["Cross arms over your chest","Palms facing yourself","Like giving yourself a hug"],
    "FRIEND":   ["Right index finger hooked","Left index hooked from below","Both hooks link together"],
    "FAMILY":   ["Both hands make F shape","Palms facing each other","Make a big circle forward"],
    "EAT":      ["Pinch fingers together","Bring hand to mouth","Repeat the eating motion"],
    "DRINK":    ["Curve hand like holding a cup (C shape)","Bring to mouth","Tilt like drinking"],
    "WATER":    ["W shape with 3 fingers","Tap chin twice","W + two taps on chin"],
    "NAME":     ["Index+middle horizontal (H shape)","Cross with other hand same shape","Two H hands cross"],
    "GOOD":     ["Flat hand at chin","Move forward and downward","Like sending something forward"],
    "BAD":      ["Flat hand at mouth","Flip hand downward and away","End with palm facing up"],
}

LETTER_VIDEO = "https://www.youtube.com/watch?v=tkMg8g8vVUo"
WORD_VIDEO = {
    "HELLO":"https://www.lifeprint.com/asl101/pages-signs/h/hello.htm",
    "THANK_YOU":"https://www.lifeprint.com/asl101/pages-signs/t/thankyou.htm",
    "PLEASE":"https://www.lifeprint.com/asl101/pages-signs/p/please.htm",
    "SORRY":"https://www.lifeprint.com/asl101/pages-signs/s/sorry.htm",
    "YES":"https://www.lifeprint.com/asl101/pages-signs/y/yes.htm",
    "NO":"https://www.lifeprint.com/asl101/pages-signs/n/no.htm",
    "HELP":"https://www.lifeprint.com/asl101/pages-signs/h/help.htm",
    "LOVE":"https://www.lifeprint.com/asl101/pages-signs/l/love.htm",
    "FRIEND":"https://www.lifeprint.com/asl101/pages-signs/f/friend.htm",
    "FAMILY":"https://www.lifeprint.com/asl101/pages-signs/f/family.htm",
    "EAT":"https://www.lifeprint.com/asl101/pages-signs/e/eat.htm",
    "DRINK":"https://www.lifeprint.com/asl101/pages-signs/d/drink.htm",
    "WATER":"https://www.lifeprint.com/asl101/pages-signs/w/water.htm",
    "NAME":"https://www.lifeprint.com/asl101/pages-signs/n/name.htm",
    "GOOD":"https://www.lifeprint.com/asl101/pages-signs/g/good.htm",
    "BAD":"https://www.lifeprint.com/asl101/pages-signs/b/bad.htm",
}

# ──────────────────────────────────────────────────────────
#  SOUND
# ──────────────────────────────────────────────────────────
_mp3_path = None

def music_play(offset=0.0):
    if not _HAS_PYGAME or not _mp3_path: return
    try:
        pygame.mixer.music.load(_mp3_path)
        pygame.mixer.music.set_volume(0.55)
        pygame.mixer.music.play(start=float(offset))
    except Exception as e:
        print(f"⚠  Music: {e}")

def music_stop():
    if _HAS_PYGAME:
        try: pygame.mixer.music.stop()
        except: pass

def speak(text):
    if not _HAS_TTS: return
    def _run():
        with _tts_lock:
            try: _tts.say(text); _tts.runAndWait()
            except: pass
    threading.Thread(target=_run, daemon=True).start()

def _synth_beep(freqs, dur, vols):
    """Play a chord from a list of (freq, vol) tuples."""
    if not _HAS_PYGAME: return
    try:
        sr = 44100
        n  = int(sr * dur)
        t  = np.linspace(0, dur, n, endpoint=False)
        wave = np.zeros(n)
        for freq, vol in zip(freqs, vols):
            wave += np.sin(2*np.pi*freq*t) * vol
        wave = np.clip(wave, -1, 1)
        data = (wave * 26000).astype(np.int16)
        # apply short fade out
        fade = np.linspace(1, 0, n)**0.4
        data = (data * fade).astype(np.int16)
        stereo = np.column_stack([data, data])
        pygame.sndarray.make_sound(stereo).play()
    except: pass

def beep_ok():
    _synth_beep([523, 659, 784], 0.18, [0.5, 0.4, 0.35])   # C-E-G major chord

def beep_perfect():
    _synth_beep([523, 659, 784, 1047], 0.25, [0.4, 0.35, 0.3, 0.5])  # C-E-G-C

def beep_fail():
    _synth_beep([220, 185], 0.22, [0.5, 0.4])   # low dissonant

def beep_streak():
    _synth_beep([784, 988, 1175], 0.20, [0.4, 0.4, 0.5])  # G-B-D high

# ──────────────────────────────────────────────────────────
#  AI MODELS
# ──────────────────────────────────────────────────────────
SEQ_LEN = 30
_letter_model = None
_word_model   = None

def load_models():
    global _letter_model, _word_model
    p = Path("asl_model.pkl")
    if p.exists():
        with open(p,"rb") as f: _letter_model = pickle.load(f)
        print("✓ Letter model loaded")
    else:
        print("⚠  asl_model.pkl not found — using geometric detection")
    p = Path("asl_words_model.pkl")
    if p.exists():
        with open(p,"rb") as f: _word_model = pickle.load(f)
        print("✓ Word model loaded")
    else:
        print("⚠  asl_words_model.pkl not found")

def _geom_finger_ext(lm, hand="Right"):
    e = [0]*5
    e[0] = 1 if (hand=="Right" and lm[4].x < lm[3].x) or (hand=="Left" and lm[4].x > lm[3].x) else 0
    for i,(a,b) in enumerate([(8,6),(12,10),(16,14),(20,18)]):
        e[i+1] = 1 if lm[a].y < lm[b].y else 0
    return e

def _dist(lm,i,j): return math.hypot(lm[i].x-lm[j].x, lm[i].y-lm[j].y)

def detect_letter_geom(hl, hand="Right"):
    lm = hl.landmark
    t,i,m,r,p = _geom_finger_ext(lm, hand)
    d48=_dist(lm,4,8); d412=_dist(lm,4,12); d812=_dist(lm,8,12)
    R = None
    if not i and not m and not r and not p:
        if lm[4].y<lm[3].y and lm[4].x>lm[8].x: R="A"
        elif 0.06<d48<0.18: R="C"
        elif _dist(lm,4,6)<0.07: R="T"
        elif lm[4].y>lm[8].y and lm[4].y>lm[12].y: R="M"
        elif lm[4].y>lm[8].y: R="N"
        elif d48<0.08: R="O"
        elif _dist(lm,8,6)<0.06: R="X"
        else: R="S"
    elif i and m and r and p and not t: R="B"
    elif i and not m and not r and not p:
        if d412<0.06: R="D"
        elif abs(lm[8].y-lm[5].y)<0.06: R="G"
        elif lm[8].y>lm[6].y and t: R="P"
        elif lm[8].y>lm[5].y and lm[4].y>lm[3].y: R="Q"
        elif t: R="L"
    elif not i and m and r and p and d48<0.05: R="F"
    elif i and m and not r and not p:
        if abs(lm[8].y-lm[12].y)<0.05: R="H"
        elif d812<0.04 and not t: R="U"
        elif d812>0.05 and not t: R="V"
        elif t: R="K"
        else: R="R"
    elif not i and not m and not r and p and not t: R="I"
    elif not i and not m and not r and p and t: R="Y"
    elif i and m and r and not p and not t: R="W"
    return R, 0.6

def detect_letter(hl, hand="Right"):
    if _letter_model:
        lm = hl.landmark
        wx,wy,wz = lm[0].x, lm[0].y, lm[0].z
        f = []
        for p in lm: f.extend([p.x-wx, p.y-wy, p.z-wz])
        X = _letter_model["scaler"].transform([f])
        pr = _letter_model["model"].predict_proba(X)[0]
        idx = int(np.argmax(pr)); c = float(pr[idx])
        if c >= 0.45:
            return _letter_model["label_encoder"].inverse_transform([idx])[0], c, "AI"
    L, c = detect_letter_geom(hl, hand)
    return L, c, "geo"

def extract_holistic_feats(results):
    f = []
    for attr in ("left_hand_landmarks","right_hand_landmarks"):
        lm_obj = getattr(results, attr)
        if lm_obj:
            lm = lm_obj.landmark
            wx,wy,wz = lm[0].x,lm[0].y,lm[0].z
            for p in lm: f.extend([p.x-wx, p.y-wy, p.z-wz])
        else: f.extend([0.0]*63)
    FACE_KP = [10,152,234,454,1,4,5,195,61,291,13,14,33,133,362,263,70,300,94,323]
    if results.face_landmarks:
        lm = results.face_landmarks.landmark
        nx,ny,nz = lm[1].x,lm[1].y,lm[1].z
        for idx in FACE_KP:
            if idx < len(lm): p=lm[idx]; f.extend([p.x-nx,p.y-ny,p.z-nz])
            else: f.extend([0.0,0.0,0.0])
    else: f.extend([0.0]*60)
    POSE_KP = [11,12,13,14,15,16,23,24]
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        mx=(lm[11].x+lm[12].x)/2; my=(lm[11].y+lm[12].y)/2; mz=(lm[11].z+lm[12].z)/2
        for idx in POSE_KP: p=lm[idx]; f.extend([p.x-mx,p.y-my,p.z-mz])
    else: f.extend([0.0]*24)
    return f

def extract_seq_feats(seq):
    if len(seq) < 2: return None
    s = np.array(seq)
    c = []
    c.extend(np.mean(s,axis=0)); c.extend(np.std(s,axis=0))
    c.extend(s[-1]-s[0]); c.extend(np.mean(np.abs(np.diff(s,axis=0)),axis=0))
    return c

def predict_word(seq_feats):
    if not _word_model or seq_feats is None: return None, 0.0
    X = _word_model["scaler"].transform([seq_feats])
    pr = _word_model["model"].predict_proba(X)[0]
    idx = int(np.argmax(pr)); c = float(pr[idx])
    if c < 0.5: return None, c
    return _word_model["label_encoder"].inverse_transform([idx])[0], c

# ──────────────────────────────────────────────────────────
#  PARTICLE SYSTEM (for menu + celebrations)
# ──────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color, speed=None, size=None, life=None):
        self.x = float(x)
        self.y = float(y)
        self.color = color
        angle = random.uniform(0, 2*math.pi)
        spd = speed or random.uniform(0.4, 2.5)
        self.vx = math.cos(angle) * spd
        self.vy = math.sin(angle) * spd - random.uniform(0.5, 2.0)
        self.size = size or random.randint(2, 5)
        self.life = life or random.uniform(0.8, 2.5)
        self.born = time.time()
        self.gravity = 0.04

    def update(self):
        self.vy += self.gravity
        self.x += self.vx
        self.y += self.vy

    @property
    def alive(self):
        return time.time() - self.born < self.life

    @property
    def alpha(self):
        return max(0.0, 1.0 - (time.time()-self.born)/self.life)

    def draw(self, img):
        if not self.alive: return
        a = self.alpha
        c = tuple(int(v*a) for v in self.color)
        x, y = int(self.x), int(self.y)
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x,y), self.size, c, -1, cv2.LINE_AA)

# Global particle list
_particles: List[Particle] = []
_menu_particles: List[Particle] = []

def spawn_burst(x, y, color, n=20):
    for _ in range(n):
        _particles.append(Particle(x, y, color))

def spawn_menu_particle():
    """Slow ambient particles for menu."""
    x = random.randint(0, SW)
    colors = [CYAN, PINK, PURP, GOLD, GRN]
    c = random.choice(colors)
    p = Particle(x, SH+10, c, speed=random.uniform(0.3,1.0), size=random.randint(1,3), life=random.uniform(4,9))
    p.vy = -random.uniform(0.3, 1.2)
    p.vx = random.uniform(-0.3, 0.3)
    p.gravity = 0
    _menu_particles.append(p)

def update_particles(lst):
    lst[:] = [p for p in lst if p.alive]
    for p in lst: p.update()

def draw_particles(img, lst):
    for p in lst: p.draw(img)

# ──────────────────────────────────────────────────────────
#  DRAW HELPERS
# ──────────────────────────────────────────────────────────
def tx(img, s, x, y, f=FS, sc=0.6, c=WHT, th=1):
    cv2.putText(img, s, (x,y), f, sc, c, th, cv2.LINE_AA)

def txc(img, s, cx, y, f=FD, sc=1.0, c=WHT, th=2):
    w,_ = cv2.getTextSize(s, f, sc, th)[0]
    cv2.putText(img, s, (cx-w//2, y), f, sc, c, th, cv2.LINE_AA)

def glow(img, s, cx, y, f, sc, c, th=2, r=5):
    w,_ = cv2.getTextSize(s, f, sc, th)[0]
    x = cx-w//2
    # multi-layer glow
    for offset, alpha in [(r, 0.12), (r//2, 0.20)]:
        sh = tuple(min(255,int(v*alpha)) for v in c)
        cv2.putText(img, s, (x+offset, y+offset), f, sc, sh, th+3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), f, sc, c, th, cv2.LINE_AA)

def hbar(img, x, y, w, h, val, mx, c, bg=None):
    bg = bg or MID
    cv2.rectangle(img, (x,y), (x+w, y+h), bg, -1)
    fw = int(w * min(val,mx)/mx) if mx else 0
    if fw:
        cv2.rectangle(img, (x,y), (x+fw, y+h), c, -1)
        # highlight on top
        hl = tuple(min(255, int(v*1.4)) for v in c)
        cv2.rectangle(img, (x,y), (x+fw, y+max(1,h//3)), hl, -1)

def rounded_rect(img, x, y, w, h, r, fill, border=None, bw=1):
    cv2.rectangle(img, (x+r, y), (x+w-r, y+h), fill, -1)
    cv2.rectangle(img, (x, y+r), (x+w, y+h-r), fill, -1)
    for cx2, cy2 in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
        cv2.circle(img, (cx2,cy2), r, fill, -1, cv2.LINE_AA)
    if border:
        cv2.rectangle(img, (x+r, y), (x+w-r, y+h), border, bw)
        cv2.rectangle(img, (x, y+r), (x+w, y+h-r), border, bw)
        for cx2,cy2 in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
            cv2.circle(img,(cx2,cy2),r,border,bw,cv2.LINE_AA)

def draw_scanlines(img, alpha=0.03):
    """Subtle CRT scanlines for depth."""
    h = img.shape[0]
    overlay = img.copy()
    for y in range(0, h, 4):
        cv2.line(overlay, (0,y), (img.shape[1],y), (0,0,0), 1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_vignette(img):
    """Dark edges vignette."""
    h, w = img.shape[:2]
    Y, X = np.mgrid[0:h, 0:w]
    cx2, cy2 = w/2, h/2
    dist = np.sqrt(((X-cx2)/cx2)**2 + ((Y-cy2)/cy2)**2)
    mask = np.clip(dist - 0.4, 0, 1) * 0.6
    img[:,:,0] = np.clip(img[:,:,0] * (1-mask), 0, 255).astype(np.uint8)
    img[:,:,1] = np.clip(img[:,:,1] * (1-mask), 0, 255).astype(np.uint8)
    img[:,:,2] = np.clip(img[:,:,2] * (1-mask), 0, 255).astype(np.uint8)

def draw_hand_illustration(img, letter, x, y, w, h, color=CYAN):
    cx2, cy2 = x+w//2, y+h//2
    r = min(w,h)//2 - 10
    palm = np.array([[cx2-r//2,cy2+r//2],[cx2+r//2,cy2+r//2],[cx2+r//2+12,cy2-12],[cx2-r//2-12,cy2-12]],np.int32)
    palm_col = tuple(int(v*0.4) for v in color)
    cv2.fillPoly(img, [palm], palm_col)
    cv2.polylines(img, [palm], True, color, 2)
    bases = [(cx2-r//2-15,cy2-8),(cx2-r//3,cy2-12),(cx2,cy2-14),(cx2+r//3,cy2-12),(cx2+r//2+10,cy2-8)]
    configs = {
        "A":[(0,.4),(0,.5),(0,.5),(0,.5),(0,.5)], "B":[(0,.3),(1,.9),(1,.9),(1,.9),(1,.8)],
        "C":[(1,.5),(1,.6),(1,.6),(1,.5),(1,.4)], "D":[(0,.4),(1,.9),(0,.5),(0,.5),(0,.5)],
        "E":[(0,.2),(0,.4),(0,.4),(0,.4),(0,.4)], "F":[(0,.3),(0,.3),(1,.8),(1,.8),(1,.7)],
        "G":[(1,.5),(1,.8),(0,.5),(0,.5),(0,.5)], "H":[(0,.3),(1,.8),(1,.8),(0,.5),(0,.5)],
        "I":[(0,.3),(0,.5),(0,.5),(0,.5),(1,.8)], "K":[(1,.5),(1,.9),(1,.8),(0,.5),(0,.5)],
        "L":[(1,.6),(1,.9),(0,.5),(0,.5),(0,.5)], "M":[(0,.3),(0,.5),(0,.5),(0,.5),(0,.5)],
        "N":[(0,.3),(0,.5),(0,.5),(0,.5),(0,.5)], "O":[(1,.5),(1,.6),(1,.6),(1,.5),(1,.4)],
        "P":[(1,.5),(1,.7),(1,.7),(0,.5),(0,.5)], "Q":[(1,.5),(1,.7),(0,.5),(0,.5),(0,.5)],
        "R":[(0,.3),(1,.9),(1,.9),(0,.5),(0,.5)], "S":[(0,.3),(0,.5),(0,.5),(0,.5),(0,.5)],
        "T":[(1,.4),(0,.5),(0,.5),(0,.5),(0,.5)], "U":[(0,.3),(1,.9),(1,.9),(0,.5),(0,.5)],
        "V":[(0,.3),(1,.9),(1,.9),(0,.5),(0,.5)], "W":[(0,.3),(1,.9),(1,.9),(1,.9),(0,.5)],
        "X":[(0,.3),(1,.6),(0,.5),(0,.5),(0,.5)], "Y":[(1,.6),(0,.5),(0,.5),(0,.5),(1,.8)],
    }
    cfg = configs.get(letter, [(1,.8)]*5)
    fl = r * 0.85
    tip_color = tuple(min(255, int(v*1.5)) for v in color)
    for (bx,by),(ext,lng) in zip(bases,cfg):
        tl = int(fl*lng)
        if ext:
            tip = (bx, by-tl)
            cv2.line(img,(bx,by),tip,color,9,cv2.LINE_AA)
            cv2.circle(img,tip,6,tip_color,-1,cv2.LINE_AA)
        else:
            ctrl=(bx,by+tl//2); end=(bx,by+tl//3)
            pts=[(int((1-t2)**2*bx+2*(1-t2)*t2*ctrl[0]+t2**2*end[0]),
                  int((1-t2)**2*by+2*(1-t2)*t2*ctrl[1]+t2**2*end[1])) for t2 in np.linspace(0,1,8)]
            bone_col = tuple(int(v*0.6) for v in color)
            for j in range(len(pts)-1): cv2.line(img,pts[j],pts[j+1],bone_col,8,cv2.LINE_AA)
    # Letter label in centre
    glow(img, letter, cx2, y+h-15, FD, 1.6, color, 3, 4)

def grade_label(pct):
    if pct >= 95: return "S", GOLD
    if pct >= 80: return "A", GRN
    if pct >= 65: return "B", CYAN
    if pct >= 45: return "C", ORG
    return "D", RED

# ──────────────────────────────────────────────────────────
#  GAME STATE
# ──────────────────────────────────────────────────────────
@dataclass
class GS:
    screen:     str   = "menu"
    difficulty: str   = "debutant"
    game_mode:  str   = "letters"
    paused:     bool  = False
    # game
    queue:      list  = field(default_factory=list)
    idx:        int   = 0
    hold:       int   = 0
    hold_max:   int   = 15
    t_item:     float = 10.0
    t_start:    float = 0.0
    scroll_x:   list  = field(default_factory=list)
    # scores
    correct:    int   = 0
    wrong:      int   = 0
    skipped:    int   = 0
    score:      int   = 0
    streak:     int   = 0
    best:       int   = 0
    rate:       float = 0.0
    history:    list  = field(default_factory=list)  # True/False per item
    # countdown
    cd:         int   = 3
    cd_start:   float = 0.0
    # detection
    det_letter: str   = ""
    det_conf:   float = 0.0
    det_mode:   str   = ""
    buf_words:  deque = field(default_factory=lambda: deque(maxlen=SEQ_LEN))
    det_word:   str   = ""
    # feedback
    flash_col:  tuple = (0,0,0)
    flash_t:    float = 0.0
    popup:      str   = ""
    popup_t:    float = 0.0
    popup_col:  tuple = GRN
    # song
    song_idx:   int   = 0
    song_score: int   = 0
    song_wrong: list  = field(default_factory=list)
    song_phase: str   = "detect"
    song_pt:    float = 0.0
    song_best_pred: str   = ""
    song_best_conf: float = 0.0
    song_buf:   deque = field(default_factory=lambda: deque(maxlen=SEQ_LEN))
    song_spoken:int   = -1
    song_pred:  str   = ""
    song_conf:  float = 0.0
    song_history: list = field(default_factory=list)
    # training
    train_items: list  = field(default_factory=list)
    train_idx:   int   = 0
    train_mode:  str   = "letters"
    train_hold:  int   = 0
    train_steps: int   = 0
    train_det:   str   = ""
    train_conf:  float = 0.0
    train_buf:   deque = field(default_factory=lambda: deque(maxlen=SEQ_LEN))
    train_stars: list  = field(default_factory=list)  # stars earned per item

    @property
    def current(self):
        return self.queue[self.idx] if self.idx < len(self.queue) else ""

    def rate_update(self):
        tot = self.correct+self.wrong+self.skipped
        self.rate = self.correct/tot*100 if tot else 0.0

GAP = 190
SPD = 2.4

def start_game(gs):
    lvl = LEVELS[gs.difficulty]
    gs.queue     = list(lvl["items"])
    gs.t_item    = lvl["t"]
    gs.hold_max  = lvl["hold"]
    gs.game_mode = lvl["mode"]
    gs.idx = gs.hold = gs.correct = gs.wrong = gs.skipped = gs.score = gs.streak = gs.best = 0
    gs.rate = 0.0
    gs.history = []
    gs.buf_words = deque(maxlen=SEQ_LEN)
    gs.det_word = ""; gs.det_letter = ""
    gs.scroll_x = [float(SW + 80 + i*GAP) for i in range(len(gs.queue))]
    gs.screen = "countdown"
    gs.cd = 3; gs.cd_start = time.time()
    gs.paused = False

def advance_item(gs, result):
    elapsed = time.time()-gs.t_start
    if result == "correct":
        gs.correct += 1
        bonus_time = max(0,int((gs.t_item-elapsed)*15))
        bonus_streak = gs.streak * 10
        gs.score += 100 + bonus_time + bonus_streak
        gs.streak += 1; gs.best = max(gs.best, gs.streak)
        gs.flash_col = GRN; gs.flash_t = time.time()
        gs.history.append(True)
        if gs.streak >= 5:
            gs.popup = f"x{gs.streak} STREAK!"; gs.popup_col = GOLD; gs.popup_t = time.time()
            beep_streak()
            spawn_burst(HW//2, CAM_H//2, GOLD, 30)
        else:
            gs.popup = "GREAT!"; gs.popup_col = GRN; gs.popup_t = time.time()
            beep_ok() if gs.streak < 3 else beep_perfect()
            spawn_burst(HW//2, CAM_H//2, GRN, 15)
    elif result == "wrong":
        gs.wrong += 1; gs.streak = 0
        gs.flash_col = RED; gs.flash_t = time.time()
        gs.popup = "MISSED!"; gs.popup_col = RED; gs.popup_t = time.time()
        gs.history.append(False)
        beep_fail()
    else:
        gs.skipped += 1; gs.score = max(0,gs.score-25); gs.streak = 0
        gs.history.append(False)
    gs.rate_update()
    gs.hold = 0; gs.idx += 1
    gs.buf_words = deque(maxlen=SEQ_LEN); gs.det_word = ""
    if gs.idx >= len(gs.queue):
        gs.screen = "result"
    else:
        gs.t_start = time.time()

# ──────────────────────────────────────────────────────────
#  RENDER — SHARED HELPERS
# ──────────────────────────────────────────────────────────
def draw_cam_left(s, cam_frame, border_col=CYAN):
    if cam_frame is not None:
        s[0:CAM_H, 0:HW] = cam_frame
    else:
        s[0:CAM_H, 0:HW] = BG
        txc(s, "Camera...", HW//2, CAM_H//2, FS, .7, GRY, 1)
    # Neon border
    cv2.rectangle(s, (1,1), (HW-1, CAM_H-1), border_col, 2)
    # Corner accents
    L = 20
    for (ax,ay),(dx,dy) in [((1,1),(L,0)),((1,1),(0,L)),((HW-1,1),(-L,0)),((HW-1,1),(0,L)),
                            ((1,CAM_H-1),(L,0)),((1,CAM_H-1),(0,-L)),((HW-1,CAM_H-1),(-L,0)),((HW-1,CAM_H-1),(0,-L))]:
        cv2.line(s,(ax,ay),(ax+dx,ay+dy),GOLD,3,cv2.LINE_AA)

def draw_hud_panel(s, x, y, w, h, col):
    """Frosted glass-style HUD panel."""
    overlay = s.copy()
    cv2.rectangle(overlay, (x,y), (x+w,y+h), DRK, -1)
    cv2.addWeighted(overlay, 0.75, s, 0.25, 0, s)
    cv2.rectangle(s, (x,y), (x+w,y+h), col, 1)

def keybar(s, y, keys):
    """Draw a small keyboard shortcut bar."""
    x = 8
    for key, action in keys:
        w1,_ = cv2.getTextSize(key, FS, .32, 1)[0]
        w2,_ = cv2.getTextSize(action, FS, .30, 1)[0]
        cv2.rectangle(s, (x,y-10), (x+w1+6, y+4), MID, -1)
        cv2.rectangle(s, (x,y-10), (x+w1+6, y+4), GRY, 1)
        tx(s, key, x+3, y, FS, .32, WHT, 1)
        x += w1+10
        tx(s, action, x, y, FS, .30, GRY, 1)
        x += w2+16

# ──────────────────────────────────────────────────────────
#  RENDER — MENU
# ──────────────────────────────────────────────────────────
def render_menu(gs, cam_frame, t_now):
    s = np.zeros((SH,SW,3),np.uint8); s[:]=BG

    # Background: faint camera feed
    if cam_frame is not None:
        bg = cv2.resize(cam_frame,(SW,SH))
        cv2.addWeighted(bg, 0.06, s, 0.94, 0, s)

    # Ambient particles
    update_particles(_menu_particles)
    draw_particles(s, _menu_particles)

    cx = SW//2
    pulse = 0.82 + 0.18 * math.sin(t_now * 2.2)

    # ── LOGO (centred, large) ──────────────────────────────
    logo = _load_logo("menu_hero", 175, 175, circle=True)
    if logo is not None:
        # Pulsing glow ring behind logo
        ring_r = int(92 + 4*math.sin(t_now*3))
        ring_c = tuple(int(v*pulse) for v in PINK)
        cv2.circle(s,(cx,100),ring_r+6,tuple(int(v*.25) for v in PINK),-1,cv2.LINE_AA)
        cv2.circle(s,(cx,100),ring_r+2,ring_c,3,cv2.LINE_AA)
        overlay_logo(s, logo, cx, 100)
        # Second thin ring
        cv2.circle(s,(cx,100),ring_r+12,tuple(int(v*.12) for v in PURP),2,cv2.LINE_AA)
        # Tagline below logo
        txc(s,"Learn American Sign Language by playing",cx,198,FS,.48,GRY,1)
    else:
        # Fallback text title
        glow(s,"JUST",cx,85,FD,4.2,tuple(int(v*pulse) for v in PINK),9,16)
        glow(s,"SIGN",cx,158,FD,4.2,PURP,9,10)
        txc(s,"Learn American Sign Language by playing",cx,195,FS,.48,GRY,1)

    cv2.line(s,(100,210),(SW-100,210),EDGE,1)

    # ── Buttons row 1 ─────────────────────────────────────
    row1 = [("debutant","1",cx-395),("intermediaire","2",cx-185),("expert","3",cx+25),("song","4",cx+235)]
    row2 = [("train_letters","5",cx-210),("train_words","6",cx+10)]

    def draw_btn(key, num, bx, by, bw=210, bh=130):
        if key=="song":
            col=PURP; lbl="SONG"; subdesc="Bohemian Rhapsody"; icon="S"; dis=False
        elif key=="train_letters":
            col=TEAL; lbl="LEARN ABC"; subdesc="Alphabet + visuals"; icon="A"; dis=False
        elif key=="train_words":
            col=ORG; lbl="LEARN WORDS"; subdesc="ASL words + videos"; icon="W"; dis=False
        else:
            lvl=LEVELS[key]; col=lvl["color"]; lbl=lvl["label"]
            subdesc=lvl["desc"]; icon=lvl["icon"]
            dis=(key=="intermediaire" and _word_model is None)
        sel=(gs.difficulty==key)
        if dis: col=GRY
        bg_c = tuple(int(v*.22) for v in col) if sel else DRK
        rounded_rect(s,bx,by,bw,bh,8,bg_c)
        if sel and not dis:
            rounded_rect(s,bx,by,bw,5,4,col)
        rounded_rect(s,bx,by,bw,bh,8,(0,0,0),col,2 if sel else 1)
        glow(s,f"{icon}{num}",bx+bw//2,by+44,FD,1.55,col if not dis else GRY,3,4)
        txc(s,lbl,bx+bw//2,by+70,FD,.43,col if not dis else GRY,1)
        txc(s,subdesc,bx+bw//2,by+90,FS,.28,WHT if not dis else GRY,1)
        if dis: txc(s,"(model missing)",bx+bw//2,by+110,FS,.23,RED,1)
        else:   txc(s,f"[{num}]",bx+bw//2,by+110,FS,.27,GRY,1)

    for key,num,bx in row1: draw_btn(key,num,bx,225)
    for key,num,bx in row2: draw_btn(key,num,bx,375,200,112)

    # ── Play button ───────────────────────────────────────
    t_pulse2 = 0.5 + 0.5*math.sin(t_now*3)
    play_c = tuple(int(PINK[i]*0.65 + PINK[i]*0.35*t_pulse2) for i in range(3))
    rounded_rect(s,cx-120,506,240,52,12,DRK,play_c,2)
    glow(s,"PLAY  [SPACE]",cx,538,FD,.88,play_c,2,3)
    txc(s,"[Q] Quit",cx,572,FS,.38,GRY,1)

    # Music status badge
    if _mp3_path and Path(_mp3_path).exists():
        rounded_rect(s,cx-145,582,290,22,5,DRK,GRN,1)
        txc(s,f"Music: {Path(_mp3_path).name}",cx,598,FS,.35,GRN,1)
    elif gs.difficulty=="song":
        txc(s,"Launch with --mp3 for real music",cx,594,FS,.32,ORG,1)

    # Small logo badge bottom-right
    logo_sm = _load_logo("hud_badge", 40, 40, circle=True)
    overlay_logo(s, logo_sm, SW-28, SH-24)
    tx(s,"v4",SW-52,SH-8,FS,.30,GRY,1)

    disco_shimmer(s, t_now, 0.04)
    draw_scanlines(s)
    return s

# ──────────────────────────────────────────────────────────
#  RENDER — COUNTDOWN
# ──────────────────────────────────────────────────────────
def render_countdown(gs, cam_frame):
    s = np.zeros((SH,SW,3),np.uint8)
    if cam_frame is not None:
        bg = cv2.resize(cam_frame,(SW,SH))
        cv2.addWeighted(bg, .3, s, .7, 0, s)
    cx = SW//2
    n = gs.cd
    col = GRN if n==1 else (ORG if n==2 else PURP)

    # Animated ring
    angle = int((time.time()-gs.cd_start)*360) % 360
    cv2.ellipse(s,(cx,SH//2),(90,90),0,0,angle,col,4,cv2.LINE_AA)

    glow(s,"GET READY!",cx,SH//2-115,FD,1.8,PINK,3,8)
    glow(s,str(n),cx,SH//2+55,FD,8.0,col,14,26)

    # Logo corner badge
    logo_sm = _load_logo("hud_badge",44,44,circle=True)
    overlay_logo(s, logo_sm, 30, 28)

    if gs.queue:
        lvl = LEVELS.get(gs.difficulty,{})
        mode = "words" if gs.game_mode=="words" else "letters"
        txc(s,f"First: {gs.queue[0].replace('_',' ')}",cx,SH//2+148,FS,.70,WHT,1)
        txc(s,f"{len(gs.queue)} {mode} - [M] to cancel",cx,SH//2+178,FS,.40,GRY,1)
    return s

# ──────────────────────────────────────────────────────────
#  RENDER — PLAYING
# ──────────────────────────────────────────────────────────
def render_playing(gs, cam_frame):
    s = np.zeros((SH,SW,3),np.uint8); s[:]=BG

    item = gs.current
    elapsed = time.time()-gs.t_start
    rem = max(0.0, gs.t_item-elapsed)
    ratio = rem/gs.t_item
    fc = GRN if ratio>.55 else (ORG if ratio>.25 else RED)

    # ── Left: live camera ──────────────────────────────────
    draw_cam_left(s, cam_frame, fc)

    # Detection HUD (top-left)
    draw_hud_panel(s, 5, 5, 345, 74, fc)
    if gs.game_mode=="letters":
        det = gs.det_letter
        dc = GRN if det==item else (RED if det else GRY)
        tx(s,"Detected letter:", 12, 27, FS, .42, GRY)
        tx(s, det or "---", 175, 33, FD, 1.0, dc, 2)
        badge = "AI" if gs.det_mode=="AI" else "geo"
        rounded_rect(s, 302,12,35,18,4,MID,GRY,1)
        tx(s, badge, 306, 26, FS, .33, GRY)
    else:
        det = gs.det_word
        dc = GRN if det==item else (ORG if det else GRY)
        tx(s,"Detected:", 12, 27, FS, .42, GRY)
        tx(s, (det.replace("_"," ") if det else "---")[:14], 110, 33, FD, .80, dc, 2)
    if gs.det_conf > 0:
        cp = int(gs.det_conf*100)
        tx(s, f"{cp}%", 285, 27, FS, .36, dc)
    # Score line
    tx(s, f"Score  {gs.score}", 12, 56, FS, .40, GOLD)
    streak_col = GOLD if gs.streak >= 3 else WHT
    tx(s, f"x{gs.streak}", 145, 56, FS, .40, streak_col)
    tx(s, f"{gs.rate:.0f}%", 215, 56, FS, .38, GRN if gs.rate>=70 else (ORG if gs.rate>=40 else RED))

    # Word buffer bar
    if gs.game_mode=="words":
        hbar(s,6,78,160,5,len(gs.buf_words),SEQ_LEN,PINK)
        tx(s,"buffer",170,83,FS,.28,GRY)

    # Hold bar (bottom of cam)
    hr = gs.hold / gs.hold_max
    cv2.rectangle(s,(2,CAM_H-22),(HW-2,CAM_H-2),DRK,-1)
    hbar(s, 2, CAM_H-22, HW-4, 20, gs.hold, gs.hold_max, GRN if hr>.6 else (ORG if hr>.3 else GRY))
    hold_label = "HOLD!" if hr>.5 else f"hold {int(hr*100)}%"
    txc(s, hold_label, HW//2, CAM_H-6, FS, .38, WHT, 1)

    # ── Right: reference panel ─────────────────────────────
    ref = np.zeros((CAM_H,HW,3),np.uint8); ref[:]=PANEL
    if gs.game_mode=="letters":
        txc(ref,"SIGN THIS LETTER",HW//2,24,FS,.50,PINK,1)
        cv2.line(ref,(20,36),(HW-20,36),EDGE,1)
        glow(ref,item,HW//2,138,FD,5.5,CYAN,9,6)
        draw_hand_illustration(ref,item,18,145,HW-36,CAM_H-235,CYAN)
        hint = LETTER_HINTS.get(item,"")
        rounded_rect(ref, 8, CAM_H-68, HW-16, 58, 6, DRK, PINK, 1)
        tx(ref,"Tip: "+hint[:36],14,CAM_H-44,FS,.38,GOLD)
    else:
        disp = item.replace("_"," ")
        txc(ref,"SIGN THIS WORD",HW//2,24,FS,.50,ORG,1)
        cv2.line(ref,(20,36),(HW-20,36),EDGE,1)
        glow(ref,disp,HW//2,110,FD,2.5,ORG,6,5)
        hint = WORD_HINTS.get(item,"")
        # Steps preview
        steps = WORD_STEPS.get(item,[])
        y0 = 140
        for i,st in enumerate(steps[:3]):
            sc = tuple(int(v*.5) for v in ORG)
            cv2.circle(ref,(22,y0+i*32),9,ORG,-1,cv2.LINE_AA)
            tx(ref,str(i+1),18,y0+5+i*32,FS,.33,BG)
            tx(ref,st[:36],36,y0+5+i*32,FS,.34,WHT)
        rounded_rect(ref,8,CAM_H-68,HW-16,58,6,DRK,ORG,1)
        tx(ref,"Tip: "+hint[:36],14,CAM_H-44,FS,.38,GOLD)

    # Timer on panel
    tcy = CAM_H-108
    rounded_rect(ref,HW//2-85,tcy-26,170,40,6,DRK)
    glow(ref,f"{rem:.1f}s",HW//2,tcy,FD,1.1,fc,2)
    hbar(ref,50,tcy+6,HW-100,5,rem,gs.t_item,fc)
    tx(ref,f"{gs.idx+1}/{len(gs.queue)}  {gs.correct}ok {gs.wrong}x",10,CAM_H-18,FS,.34,GRY)
    s[0:CAM_H,HW:SW]=ref

    # Divider
    cv2.line(s,(HW,0),(HW,CAM_H),EDGE,1)
    cv2.line(s,(HW+1,0),(HW+1,CAM_H),tuple(int(v*.3) for v in CYAN),1)

    # ── Bottom scrolling lane ──────────────────────────────
    sy = CAM_H
    cv2.rectangle(s,(0,sy),(SW,SH),(10,6,20),-1)
    # Neon top line
    cv2.line(s,(0,sy),(SW,sy),PINK,2)
    cv2.line(s,(0,sy+1),(SW,sy+1),tuple(int(v*.2) for v in PINK),1)

    # Target zone
    TX = 155
    rounded_rect(s,TX-58,sy+4,116,BOT_H-8,8,DRK,PINK,2)
    glow(s,"HERE",TX,sy+BOT_H//2+8,FS,.50,PINK,1,3)

    # Progress bar
    hbar(s,TX+68,sy+6,SW-TX-76,6,gs.idx,len(gs.queue),PINK)
    tx(s,f"{gs.idx+1}/{len(gs.queue)}",TX+68,sy+22,FS,.30,GRY)
    tx(s,"[S] Skip  [P] Pause  [M] Menu  [Q] Quit",SW-355,sy+20,FS,.34,GRY)

    # History dots
    for i,ok in enumerate(gs.history[-20:]):
        dx = TX+75 + i*13
        dy = sy+32
        if dx < SW-10:
            cv2.circle(s,(dx,dy),4,GRN if ok else RED,-1,cv2.LINE_AA)

    # Scroll items
    scy = sy + BOT_H//2 + 10
    for i,(item_i,sx) in enumerate(zip(gs.queue,gs.scroll_x)):
        ix = int(sx)
        if ix<-90 or ix>SW+60: continue
        dt = item_i.replace("_"," ")
        is_cur = (i==gs.idx)
        is_done = (i<gs.idx)
        result_ok = gs.history[i] if i < len(gs.history) else None

        if is_cur:
            rounded_rect(s,ix-62,scy-66,124,76,8,(0,20,40),PINK,2)
            glow(s,dt[:8],ix,scy,FD,2.4 if gs.game_mode=="words" else 2.9,CYAN,4,5)
        elif is_done:
            dot_c = GRN if result_ok else RED
            cv2.circle(s,(ix,scy-30),4,dot_c,-1,cv2.LINE_AA)
            tx(s,dt[:6],ix-18,scy+8,FD,.52,tuple(int(v*.4) for v in dot_c),1)
        else:
            dr = sx-TX; fade=max(.12,1.0-dr/(SW*.55)); sc2=max(.4,1.3*fade); sh=int(170*fade)
            txc(s,dt[:8],ix,scy,FD,sc2,(sh//4,sh//3,sh),max(1,int(sc2)))

    # Flash overlay
    if time.time()-gs.flash_t < .25:
        ov=s[0:CAM_H].copy()
        cv2.rectangle(ov,(0,0),(SW,CAM_H),gs.flash_col,-1)
        cv2.addWeighted(ov,.14,s[0:CAM_H],.86,0,s[0:CAM_H])

    # Popup
    if time.time()-gs.popup_t < .7 and gs.popup:
        glow(s,gs.popup,HW//2,CAM_H//2,FD,3.2,gs.popup_col,7,7)

    # Particles
    update_particles(_particles)
    draw_particles(s, _particles)

    # Logo badge top-right corner
    logo_sm = _load_logo("hud_badge",44,44,circle=True)
    overlay_logo(s, logo_sm, SW-28, 28, alpha=0.75)

    draw_scanlines(s, 0.025)
    return s

# ──────────────────────────────────────────────────────────
#  RENDER — PAUSE
# ──────────────────────────────────────────────────────────
def render_pause(gs, cam_frame):
    s = np.zeros((SH,SW,3),np.uint8)
    if cam_frame is not None:
        bg=cv2.resize(cam_frame,(SW,SH)); cv2.addWeighted(bg,.15,s,.85,0,s)
    cx = SW//2
    rounded_rect(s,cx-200,SH//2-130,400,260,16,DRK,CYAN,2)
    glow(s,"PAUSED",cx,SH//2-70,FD,2.2,CYAN,4,8)
    cv2.line(s,(cx-160,SH//2-40),(cx+160,SH//2-40),EDGE,1)
    txc(s,f"Score: {gs.score}  Streak: x{gs.streak}",cx,SH//2+5,FS,.55,GOLD,1)
    txc(s,f"{gs.correct} correct  {gs.wrong} missed  {gs.skipped} skipped",cx,SH//2+38,FS,.46,WHT,1)
    txc(s,"[P] Resume  [M] Menu  [Q] Quit",cx,SH//2+90,FS,.48,GRY,1)
    return s

# ──────────────────────────────────────────────────────────
#  RENDER — RESULT
# ──────────────────────────────────────────────────────────
def render_result(gs):
    s = np.zeros((SH,SW,3),np.uint8); s[:]=BG
    cx = SW//2

    update_particles(_particles)
    draw_particles(s, _particles)

    # Logo badge top-left
    logo_med = _load_logo("result_badge",55,55,circle=True)
    overlay_logo(s, logo_med, 38, 35)

    mt = "WORDS" if gs.game_mode=="words" else "LETTERS"
    glow(s,f"RESULTS  -  {mt}",cx,55,FD,1.5,GOLD,4,7)
    cv2.line(s,(80,70),(SW-80,70),EDGE,1)

    rt = gs.rate
    grade, grade_col = grade_label(rt)

    # Grade circle
    cv2.circle(s,(cx,180),72,DRK,-1,cv2.LINE_AA)
    cv2.circle(s,(cx,180),72,grade_col,3,cv2.LINE_AA)
    glow(s,grade,cx,203,FD,3.8,grade_col,7,10)

    # Stats columns
    glow(s,str(gs.score),cx-240,195,FD,2.8,GOLD,5,6)
    txc(s,"SCORE",cx-240,228,FS,.44,GOLD,1)
    glow(s,f"{rt:.0f}%",cx+240,195,FD,2.8,GRN if rt>=70 else ORG,5,6)
    txc(s,"SUCCESS",cx+240,228,FS,.44,GRN if rt>=70 else ORG,1)

    # Stat cards
    stats=[
        (f"Correct: {gs.correct}", GRN),
        (f"Missed:  {gs.wrong}",   RED),
        (f"Skipped: {gs.skipped}", ORG),
        (f"Best streak: x{gs.best}", GOLD),
    ]
    for i,(st,c) in enumerate(stats):
        bx = cx-265 if i%2==0 else cx+15
        by = 255 + (i//2)*60
        rounded_rect(s,bx,by,250,48,8,DRK,c,1)
        tx(s,st,bx+14,by+30,FS,.52,c,1)

    # History mini-bar
    if gs.history:
        txc(s,"Sign history:",cx,398,FS,.40,GRY,1)
        bw = min(20, int(440/max(len(gs.history),1)))
        ox = cx - len(gs.history)*bw//2
        for i,ok in enumerate(gs.history):
            c2 = GRN if ok else RED
            cv2.rectangle(s,(ox+i*bw,408),(ox+i*bw+bw-2,422),c2,-1)

    # Grade message
    msg,mc = [
        ("",""),
        ("PERFECT RUN!", GOLD), ("EXCELLENT!", GOLD),
        ("WELL DONE!", GRN), ("GOOD JOB!", GRN),
        ("KEEP GOING!", CYAN), ("KEEP PRACTICING!", ORG)
    ][-1 if rt<40 else (-2 if rt<55 else (-3 if rt<65 else (-4 if rt<80 else (-5 if rt<95 else -6))))]
    glow(s,msg,cx,456,FD,1.1,mc,2,5)

    # Buttons
    rounded_rect(s,cx-270,480,240,52,10,DRK,GRN,2)
    txc(s,"PLAY AGAIN [SPACE]",cx-150,512,FS,.52,GRN,1)
    rounded_rect(s,cx+30,480,240,52,10,DRK,WHT,1)
    txc(s,"MENU [M]",cx+150,512,FS,.52,WHT,1)

    draw_scanlines(s, 0.02)
    return s

# ──────────────────────────────────────────────────────────
#  RENDER — SONG
# ──────────────────────────────────────────────────────────
def _song_bottom(s, gs):
    sy = CAM_H; cx = SW//2
    total = len(SONG_LYRICS)

    # Dark bottom area
    cv2.rectangle(s,(0,sy),(SW,SH),(8,4,18),-1)
    cv2.line(s,(0,sy),(SW,sy),PURP,2)
    cv2.line(s,(0,sy+1),(SW,sy+1),tuple(int(v*.2) for v in PURP),1)

    # Progress bar with checkpoint marks
    hbar(s,4,sy+4,SW-8,8,gs.song_idx,total,PURP)
    pct_str = f"{gs.song_idx+1}/{total} | {gs.song_score} hits"
    txc(s,pct_str,cx,sy+24,FS,.44,WHT,1)

    lyric = SONG_LYRICS[gs.song_idx] if gs.song_idx<total else None
    if lyric:
        # Current lyric — karaoke style
        glow(s,f'"{lyric["text"]}"',cx,sy+52,FD,.92,CYAN,2,4)
        txc(s,f"ASL sign: {lyric['sign']}",cx,sy+78,FS,.48,PURP,1)

    # Upcoming lyrics (fading)
    future = SONG_LYRICS[gs.song_idx+1:gs.song_idx+4]
    for i,l in enumerate(future):
        a = max(50,165-i*55)
        txc(s,l["text"],cx,sy+104+i*26,FS,.35,(a//4,a//3,a),1)

    # Key hints
    if lyric:
        tip = SIGN_HINTS.get(lyric['sign'],'')[:40]
        tx(s,f"[M] Menu  [Q] Quit    Tip: {tip}",6,SH-7,FS,.30,GOLD,1)
    # Tiny logo badge right edge
    logo_xs2 = _load_logo("panel_xs",26,26,circle=True)
    overlay_logo(s, logo_xs2, SW-16, sy+15, alpha=0.65)

def render_song_detect(gs, cam_frame):
    s = np.zeros((SH,SW,3),np.uint8); s[:]=BG
    total = len(SONG_LYRICS)
    if gs.song_idx >= total: return s
    lyric = SONG_LYRICS[gs.song_idx]
    elapsed = time.time()-gs.song_pt
    rem = max(0.0, SONG_DETECT_SEC-elapsed)
    ratio = rem/SONG_DETECT_SEC
    fc = GRN if ratio>.55 else (ORG if ratio>.25 else RED)

    draw_cam_left(s, cam_frame, fc)

    # HUD
    draw_hud_panel(s,5,5,340,70,fc)
    tx(s,"Detected:", 12, 26, FS, .40, GRY)
    dc = GRN if gs.song_pred==lyric["sign"] else (ORG if gs.song_pred else GRY)
    tx(s, gs.song_pred or "---", 120, 32, FD, .88, dc, 2)
    if gs.song_conf > 0:
        tx(s, f"{int(gs.song_conf*100)}%", 280, 26, FS, .36, dc)
    tx(s, f"Score: {gs.song_score}/{total}", 12, 56, FS, .38, GOLD)

    # Countdown arc on cam
    arc_angle = int(ratio * 360)
    cv2.ellipse(s,(HW//2,CAM_H-30),(50,14),0,0,arc_angle,fc,4,cv2.LINE_AA)
    txc(s,f"{rem:.1f}s",HW//2,CAM_H-26,FS,.42,WHT,1)

    # Sign panel right
    ref = np.zeros((CAM_H,HW,3),np.uint8); ref[:]=PANEL
    sign = lyric["sign"]
    txc(ref,"ASL SIGN",HW//2,24,FS,.50,PURP,1)
    # Small logo in panel
    logo_xs = _load_logo("panel_xs",28,28,circle=True)
    overlay_logo(ref, logo_xs, HW-20, 20, alpha=0.7)
    cv2.line(ref,(20,37),(HW-20,37),EDGE,1)
    glow(ref,sign,HW//2,130,FD,2.8,PURP,7,6)
    hint = SIGN_HINTS.get(sign,"")
    words2 = hint.split(); line2=""; lines2=[]
    for w in words2:
        if len(line2+" "+w)>32: lines2.append(line2); line2=w
        else: line2=(line2+" "+w).strip()
    if line2: lines2.append(line2)
    for i,l in enumerate(lines2[:3]): txc(ref,l,HW//2,200+i*30,FS,.42,WHT,1)
    cv2.circle(ref,(HW//2-60,320),30,PURP,2,cv2.LINE_AA)
    cv2.circle(ref,(HW//2+60,320),30,PURP,2,cv2.LINE_AA)
    txc(ref,"HANDS + BODY",HW//2,368,FS,.38,TEAL,1)
    rounded_rect(ref,8,CAM_H-62,HW-16,52,6,DRK,PURP,1)
    tx(ref,"Tip: "+hint[:40],12,CAM_H-36,FS,.34,GOLD)
    s[0:CAM_H,HW:SW]=ref
    cv2.line(s,(HW,0),(HW,CAM_H),EDGE,1)
    cv2.line(s,(HW+1,0),(HW+1,CAM_H),tuple(int(v*.25) for v in PURP),1)
    _song_bottom(s,gs)
    draw_scanlines(s,0.02)
    return s

def render_song_feedback(gs, cam_frame):
    s = np.zeros((SH,SW,3),np.uint8); s[:]=BG
    if cam_frame is not None:
        bg=cv2.resize(cam_frame,(SW,CAM_H)); cv2.addWeighted(bg,.3,s[0:CAM_H],.7,0,s[0:CAM_H])
    cx = SW//2
    col = GRN if gs.song_phase=="ok" else RED
    msg = "NICE!" if gs.song_phase=="ok" else "MISSED!"

    # Full-cam tint
    cv2.rectangle(s,(0,0),(SW,CAM_H),tuple(int(c*.12) for c in col),-1)
    cv2.rectangle(s,(3,3),(SW-3,CAM_H-3),col,4)

    glow(s,msg,cx,CAM_H//2-25,FD,2.8,col,6,10)

    if gs.song_idx < len(SONG_LYRICS):
        lyric = SONG_LYRICS[gs.song_idx]
        txc(s,f"Sign: {lyric['sign']}",cx,CAM_H//2+32,FS,.60,WHT,1)
        if gs.song_best_pred:
            c2 = GRN if gs.song_phase=="ok" else ORG
            txc(s,f"Detected: {gs.song_best_pred}  ({int(gs.song_best_conf*100)}%)",cx,CAM_H//2+64,FS,.46,c2,1)

    if gs.song_phase=="ok":
        spawn_burst(cx, CAM_H//2, GRN, 20)
    _song_bottom(s,gs)
    update_particles(_particles); draw_particles(s,_particles)
    return s

def render_song_score(gs, cam_frame):
    s = np.zeros((SH,SW,3),np.uint8); s[:]=BG; cx=SW//2
    if cam_frame is not None:
        bg=cv2.resize(cam_frame,(SW,SH)); cv2.addWeighted(bg,.06,s,.94,0,s)

    update_particles(_particles); draw_particles(s,_particles)

    # Logo badge top-left
    logo_med = _load_logo("result_badge",55,55,circle=True)
    overlay_logo(s, logo_med, 38, 35)

    total = len(SONG_LYRICS)
    glow(s,"SONG COMPLETE!",cx,55,FD,1.7,GOLD,4,8)
    cv2.line(s,(80,70),(SW-80,70),EDGE,1)

    rt = int(gs.song_score/total*100)
    grade, gc = grade_label(rt)

    # Grade circle
    cv2.circle(s,(cx,175),68,DRK,-1,cv2.LINE_AA)
    cv2.circle(s,(cx,175),68,gc,3,cv2.LINE_AA)
    glow(s,grade,cx,198,FD,3.6,gc,7,10)

    glow(s,f"{gs.song_score}/{total}",cx-220,188,FD,2.8,GOLD,5,6)
    txc(s,"SIGNS HIT",cx-220,222,FS,.42,GOLD,1)
    glow(s,f"{rt}%",cx+220,188,FD,2.8,GRN if rt>=70 else ORG,5,6)
    txc(s,"SUCCESS",cx+220,222,FS,.42,GRN if rt>=70 else ORG,1)

    if gs.song_wrong:
        txc(s,"Practice these signs:",cx,268,FS,.44,ORG,1)
        missed = sorted(set(gs.song_wrong))
        per_row = 7
        for row in range(math.ceil(len(missed)/per_row)):
            chunk = missed[row*per_row:(row+1)*per_row]
            txc(s,"  ".join(chunk),cx,294+row*26,FS,.38,WHT,1)

    # Mini history
    if gs.song_history:
        txc(s,"Performance:",cx,360,FS,.38,GRY,1)
        bw = min(18, int(420/max(len(gs.song_history),1)))
        ox = cx - len(gs.song_history)*bw//2
        for i,ok in enumerate(gs.song_history):
            cv2.rectangle(s,(ox+i*bw,370),(ox+i*bw+bw-2,382),GRN if ok else RED,-1)

    msg2,mc=("PERFECT RUN!",GOLD) if rt>=95 else (("EXCELLENT!",GOLD) if rt>=80 else (("WELL DONE!",GRN) if rt>=65 else (("KEEP GOING!",CYAN) if rt>=45 else ("KEEP PRACTICING!",ORG))))
    glow(s,msg2,cx,418,FD,1.1,mc,2,5)

    rounded_rect(s,cx-265,448,240,52,10,DRK,GRN,2)
    txc(s,"PLAY AGAIN [SPACE]",cx-145,480,FS,.50,GRN,1)
    rounded_rect(s,cx+25,448,240,52,10,DRK,WHT,1)
    txc(s,"MENU [M]",cx+145,480,FS,.50,WHT,1)

    draw_scanlines(s,0.02)
    return s

# ──────────────────────────────────────────────────────────
#  RENDER — TRAINING
# ──────────────────────────────────────────────────────────
def render_train(gs, cam_frame):
    s = np.zeros((SH,SW,3),np.uint8); s[:]=BG
    if not gs.train_items: return s
    item = gs.train_items[gs.train_idx]
    is_letters = (gs.train_mode=="letters")
    col = TEAL if is_letters else ORG
    total = len(gs.train_items)

    # Left: live camera
    draw_cam_left(s, cam_frame, col)
    tx(s,"LIVE CAMERA",10,20,FS,.42,col,1)
    # Logo badge top-right
    logo_sm = _load_logo("hud_badge",44,44,circle=True)
    overlay_logo(s, logo_sm, SW-28, 28, alpha=0.70)

    # Detection HUD
    det = gs.train_det; conf = gs.train_conf
    dc = GRN if det==item else (ORG if det else GRY)
    draw_hud_panel(s,5,28,320,68,dc)
    tx(s,"Detected:", 12, 48, FS, .40, GRY)
    disp = det.replace("_"," ") if det else "---"
    tx(s, disp[:14], 108, 54, FD, .82, dc, 2)
    if conf > 0: tx(s,f"{int(conf*100)}%",250,48,FS,.34,dc)

    if det==item:
        glow(s,"PERFECT!",HW//2,CAM_H//2,FD,2.8,GRN,7,8)
        spawn_burst(HW//2,CAM_H//2,GRN,8)

    # Hold progress
    hr = gs.train_hold/60
    cv2.rectangle(s,(2,CAM_H-24),(HW-2,CAM_H-2),DRK,-1)
    hbar(s,2,CAM_H-24,HW-4,22,gs.train_hold,60,GRN if det==item else GRY)
    hold_p = int(min(hr,1.0)*100)
    txc(s,f"Hold to unlock steps  {hold_p}%",HW//2,CAM_H-7,FS,.34,WHT,1)

    # Stars earned for this item
    stars = gs.train_stars[gs.train_idx] if gs.train_idx < len(gs.train_stars) else 0
    star_str = "*"*stars + "."*(3-stars)
    tx(s, f"Stars: {star_str}", 8, CAM_H-44, FS, .45, GOLD, 1)

    # Right: instruction panel
    ref = np.zeros((CAM_H,HW,3),np.uint8); ref[:]=PANEL

    if is_letters:
        # Header
        rounded_rect(ref,8,8,HW-16,30,6,DRK,col,1)
        txc(ref,f"LETTER:  {item}",HW//2,28,FD,1.0,col,2)
        cv2.line(ref,(15,44),(HW-15,44),EDGE,1)
        draw_hand_illustration(ref,item,18,50,HW-36,250,col)
        hint = LETTER_HINTS.get(item,"")
        rounded_rect(ref,8,308,HW-16,28,5,DRK,col,1)
        txc(ref,hint,HW//2,327,FS,.40,GOLD,1)
    else:
        disp_w = item.replace("_"," ")
        rounded_rect(ref,8,8,HW-16,30,6,DRK,col,1)
        txc(ref,f"WORD:  {disp_w}",HW//2,28,FD,.95,col,2)
        cv2.line(ref,(15,44),(HW-15,44),EDGE,1)
        glow(ref,disp_w,HW//2,115,FD,2.2,col,5,5)
        hint = WORD_HINTS.get(item,"")
        rounded_rect(ref,8,148,HW-16,26,5,DRK,col,1)
        txc(ref,hint[:36],HW//2,165,FS,.36,GOLD,1)

    # Steps — unlock progressively
    steps = LETTER_STEPS.get(item,[]) if is_letters else WORD_STEPS.get(item,[])
    y0 = 345 if is_letters else 188
    txc(ref,"STEPS:",HW//2,y0-10,FS,.40,WHT,1)
    for i,step in enumerate(steps[:3]):
        unlocked = i < gs.train_steps
        by2 = y0+10+i*32
        # Step card
        step_col = col if unlocked else GRY
        bg_s = tuple(int(v*.15) for v in col) if unlocked else DRK
        rounded_rect(ref,8,by2,HW-16,28,5,bg_s,step_col,1)
        if unlocked:
            cv2.circle(ref,(22,by2+14),9,col,-1,cv2.LINE_AA)
            tx(ref,str(i+1),18,by2+19,FS,.33,BG)
            tx(ref,step[:36],36,by2+19,FS,.33,WHT)
        else:
            cv2.circle(ref,(22,by2+14),9,GRY,-1,cv2.LINE_AA)
            tx(ref,str(i+1),18,by2+19,FS,.33,DRK)
            txc(ref,"Hold sign to unlock...",HW//2,by2+19,FS,.32,GRY,1)

    # Video button
    rounded_rect(ref,8,CAM_H-54,HW-16,46,8,DRK,(60,40,180),2)
    cv2.rectangle(ref,(12,CAM_H-50),(16+28,CAM_H-12),(60,40,180),-1)
    txc(ref,"WATCH VIDEO [V]",HW//2,CAM_H-24,FS,.44,WHT,1)
    txc(ref,"Opens in browser",HW//2,CAM_H-9,FS,.26,GRY,1)

    s[0:CAM_H,HW:SW]=ref
    cv2.line(s,(HW,0),(HW,CAM_H),EDGE,1)
    cv2.line(s,(HW+1,0),(HW+1,CAM_H),tuple(int(v*.25) for v in col),1)

    # Bottom bar
    sy = CAM_H
    cv2.rectangle(s,(0,sy),(SW,SH),(8,4,16),-1)
    cv2.line(s,(0,sy),(SW,sy),col,2)
    cx = SW//2

    mode_lbl = "LETTERS" if is_letters else "WORDS"
    txc(s,f"TRAINING {mode_lbl} - {gs.train_idx+1}/{total}",cx,sy+22,FD,.68,col,1)
    hint2 = LETTER_HINTS.get(item,"") if is_letters else WORD_HINTS.get(item,"")
    txc(s,hint2,cx,sy+46,FS,.42,GOLD,1)
    hbar(s,8,sy+58,SW-16,5,gs.train_idx,total,col)

    # Nav buttons
    nav_btns = [("[< PREV]",cx-330,RED),("[SPACE] NEXT >",cx-80,GRN),("[V] Video",cx+125,(80,60,210)),("[M] Menu",cx+305,GRY)]
    ny = sy+90
    for label,bx,c3 in nav_btns:
        wt,_=cv2.getTextSize(label,FS,.38,1)[0]
        rounded_rect(s,bx-wt//2-8,ny-14,wt+16,24,5,DRK,c3,1)
        txc(s,label,bx,ny,FS,.38,c3,1)

    update_particles(_particles); draw_particles(s,_particles)
    draw_scanlines(s,0.02)
    return s

# ──────────────────────────────────────────────────────────
#  CAMERA THREAD
# ──────────────────────────────────────────────────────────
class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._display = None
        self._mp_raw  = None
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self._running:
            ret, f = self.cap.read()
            if ret:
                f = cv2.resize(f, (HW, CAM_H))
                with self._lock:
                    self._mp_raw  = f.copy()
                    self._display = cv2.flip(f, 1)

    def get(self):
        with self._lock:
            if self._display is None: return None, None
            return self._display.copy(), self._mp_raw.copy()

    def stop(self):
        self._running = False
        self.cap.release()

# ──────────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────────
def main():
    global _mp3_path

    parser = argparse.ArgumentParser(description="JUST SIGN v4")
    parser.add_argument("--mp3", default=None, help="Path to Bohemian Rhapsody MP3")
    args = parser.parse_args()
    _mp3_path = args.mp3

    load_models()

    # MediaPipe setup
    mp_h   = mp.solutions.hands
    mp_hol = mp.solutions.holistic
    mp_draw= mp.solutions.drawing_utils
    hands  = mp_h.Hands(static_image_mode=False, max_num_hands=1,
                         model_complexity=0,
                         min_detection_confidence=0.5, min_tracking_confidence=0.4)
    holistic = mp_hol.Holistic(static_image_mode=False, model_complexity=0,
                                min_detection_confidence=0.4, min_tracking_confidence=0.4,
                                enable_segmentation=False)
    stl = mp_draw.DrawingSpec(color=CYAN, thickness=2, circle_radius=3)
    stc = mp_draw.DrawingSpec(color=PINK, thickness=1)
    print(f"✓ MediaPipe {mp.__version__}")

    cam = Camera()
    gs  = GS()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, SW, SH)

    cam_display = np.zeros((CAM_H,HW,3),np.uint8)
    t_last_menu_particle = 0.0

    while True:
        t_now = time.time()
        disp, mp_raw = cam.get()

        if disp is not None:
            cam_display = disp.copy()
            rgb = cv2.cvtColor(mp_raw, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            # ── Letter detection ────────────────────────────
            if gs.screen in ("playing","train") and (
                (gs.screen=="playing" and gs.game_mode=="letters") or
                (gs.screen=="train"   and gs.train_mode=="letters")
            ) and not gs.paused:
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    hl = res.multi_hand_landmarks[0]
                    hand_label = "Right"
                    if res.multi_handedness:
                        hand_label = res.multi_handedness[0].classification[0].label
                    L, c, m = detect_letter(hl, hand_label)
                    mp_draw.draw_landmarks(cam_display, hl, mp_h.HAND_CONNECTIONS, stl, stc)
                    if gs.screen=="playing": gs.det_letter=L or ""; gs.det_conf=c; gs.det_mode=m
                    else: gs.train_det=L or ""; gs.train_conf=c
                else:
                    if gs.screen=="playing": gs.det_letter=""; gs.det_conf=0.0
                    else: gs.train_det=""; gs.train_conf=0.0

            # ── Word detection (intermediate) ────────────────
            elif gs.screen=="playing" and gs.game_mode=="words" and not gs.paused:
                r2 = holistic.process(rgb)
                if r2.right_hand_landmarks:
                    mp_draw.draw_landmarks(cam_display, r2.right_hand_landmarks, mp_hol.HAND_CONNECTIONS, stl, stc)
                if r2.left_hand_landmarks:
                    mp_draw.draw_landmarks(cam_display, r2.left_hand_landmarks, mp_hol.HAND_CONNECTIONS, stl, stc)
                gs.buf_words.append(extract_holistic_feats(r2))
                if len(gs.buf_words) >= SEQ_LEN:
                    sf = extract_seq_feats(list(gs.buf_words))
                    w, c = predict_word(sf)
                    gs.det_word=w or ""; gs.det_conf=c

            # ── Word detection (training) ────────────────────
            elif gs.screen=="train" and gs.train_mode=="words":
                r2 = holistic.process(rgb)
                if r2.right_hand_landmarks:
                    mp_draw.draw_landmarks(cam_display, r2.right_hand_landmarks, mp_hol.HAND_CONNECTIONS, stl, stc)
                gs.train_buf.append(extract_holistic_feats(r2))
                if len(gs.train_buf) >= SEQ_LEN:
                    sf = extract_seq_feats(list(gs.train_buf))
                    w, c = predict_word(sf)
                    gs.train_det=w or ""; gs.train_conf=c

            # ── Song detection ───────────────────────────────
            elif gs.screen=="song" and gs.song_phase=="detect":
                r2 = holistic.process(rgb)
                if r2.right_hand_landmarks:
                    mp_draw.draw_landmarks(cam_display, r2.right_hand_landmarks, mp_hol.HAND_CONNECTIONS, stl, stc)
                if r2.left_hand_landmarks:
                    mp_draw.draw_landmarks(cam_display, r2.left_hand_landmarks, mp_hol.HAND_CONNECTIONS, stl, stc)
                gs.song_buf.append(extract_holistic_feats(r2))
                if len(gs.song_buf) >= SEQ_LEN:
                    sf = extract_seq_feats(list(gs.song_buf))
                    p, c = predict_word(sf)
                    gs.song_pred=p or ""; gs.song_conf=c
                    if p and c > gs.song_best_conf:
                        gs.song_best_pred=p; gs.song_best_conf=c

        # ── Menu ambient particles ─────────────────────────
        if gs.screen=="menu" and t_now - t_last_menu_particle > 0.15:
            spawn_menu_particle()
            t_last_menu_particle = t_now
        elif gs.screen != "menu":
            _menu_particles.clear()

        # ── Countdown logic ────────────────────────────────
        if gs.screen=="countdown":
            elapsed = t_now - gs.cd_start
            gs.cd = max(1, math.ceil(3-elapsed))
            if elapsed >= 3.0:
                gs.screen="playing"; gs.t_start=t_now

        # ── Game logic ─────────────────────────────────────
        elif gs.screen=="playing" and gs.current and not gs.paused:
            for i in range(len(gs.scroll_x)):
                if i >= gs.idx: gs.scroll_x[i] -= SPD
            elapsed = t_now - gs.t_start
            if elapsed >= gs.t_item:
                advance_item(gs, "wrong")
            else:
                det = gs.det_letter if gs.game_mode=="letters" else gs.det_word
                if det == gs.current:
                    gs.hold = min(gs.hold+1, gs.hold_max)
                    if gs.hold >= gs.hold_max: advance_item(gs,"correct")
                else:
                    gs.hold = max(0, gs.hold-1)

        # ── Song logic ─────────────────────────────────────
        elif gs.screen=="song":
            total = len(SONG_LYRICS)
            if gs.song_phase=="detect" and gs.song_idx < total:
                lyric = SONG_LYRICS[gs.song_idx]
                if gs.song_spoken != gs.song_idx:
                    speak(lyric["text"]); gs.song_spoken=gs.song_idx
                elapsed = t_now - gs.song_pt
                if elapsed >= SONG_DETECT_SEC:
                    target = lyric["sign"]
                    model_words = _word_model["words"] if _word_model else []
                    if target not in model_words:
                        ok = True
                    else:
                        ok = (gs.song_best_pred==target and gs.song_best_conf>=0.45)
                    gs.song_phase = "ok" if ok else "miss"
                    gs.song_history.append(ok)
                    if ok: gs.song_score+=1; beep_ok(); spawn_burst(SW//2, CAM_H//2, GRN, 15)
                    else: gs.song_wrong.append(target); beep_fail()
                    gs.song_pt = t_now

            elif gs.song_phase in ("ok","miss"):
                if t_now - gs.song_pt >= 1.0:
                    gs.song_idx += 1
                    if gs.song_idx >= total:
                        gs.song_phase="score"; music_stop()
                        spawn_burst(SW//2, 300, GOLD, 50)
                    else:
                        gs.song_phase="detect"; gs.song_pt=t_now
                        gs.song_best_pred=""; gs.song_best_conf=0.0
                        gs.song_pred=""; gs.song_conf=0.0
                        gs.song_buf=deque(maxlen=SEQ_LEN)

        # ── Training logic ─────────────────────────────────
        elif gs.screen=="train" and gs.train_items:
            det = gs.train_det
            item = gs.train_items[gs.train_idx]
            if det == item:
                gs.train_hold = min(gs.train_hold+1, 120)
                gs.train_steps = min(3, gs.train_hold//30)
                # Award star
                stars = gs.train_hold // 40
                if gs.train_idx < len(gs.train_stars):
                    gs.train_stars[gs.train_idx] = min(3, max(gs.train_stars[gs.train_idx], stars))
            else:
                gs.train_hold = max(0, gs.train_hold-1)

        # ── RENDER ─────────────────────────────────────────
        if   gs.screen=="menu":
            screen = render_menu(gs, cam_display, t_now)
        elif gs.screen=="countdown":
            screen = render_countdown(gs, cam_display)
        elif gs.screen=="playing":
            if gs.paused:
                screen = render_pause(gs, cam_display)
            else:
                screen = render_playing(gs, cam_display)
        elif gs.screen=="result":
            screen = render_result(gs)
        elif gs.screen=="train":
            screen = render_train(gs, cam_display)
        elif gs.screen=="song":
            if   gs.song_phase=="detect":         screen=render_song_detect(gs,cam_display)
            elif gs.song_phase in ("ok","miss"):  screen=render_song_feedback(gs,cam_display)
            elif gs.song_phase=="score":          screen=render_song_score(gs,cam_display)
            else: screen=render_song_detect(gs,cam_display)
        else:
            screen = np.zeros((SH,SW,3),np.uint8)

        cv2.imshow(WIN, screen)
        k = cv2.waitKey(1) & 0xFF

        # ── KEYS ───────────────────────────────────────────

        if k == ord('q'):
            break

        elif k in (ord('m'), ord('M'), 27):   # M / ESC → menu
            music_stop()
            gs.screen="menu"; gs.difficulty="debutant"; gs.paused=False
            _particles.clear()

        elif k in (ord('p'), ord('P')) and gs.screen=="playing":
            gs.paused = not gs.paused

        elif k == ord('1') and gs.screen=="menu": gs.difficulty="debutant"
        elif k == ord('2') and gs.screen=="menu": gs.difficulty="intermediaire"
        elif k == ord('3') and gs.screen=="menu": gs.difficulty="expert"
        elif k == ord('4') and gs.screen=="menu": gs.difficulty="song"
        elif k == ord('5') and gs.screen=="menu": gs.difficulty="train_letters"
        elif k == ord('6') and gs.screen=="menu": gs.difficulty="train_words"

        elif k in (ord(' '), 13):
            d = gs.difficulty

            if gs.screen=="menu":
                if d in ("debutant","intermediaire","expert"):
                    start_game(gs)
                elif d=="song":
                    music_stop(); gs.screen="song"
                    gs.song_idx=0; gs.song_score=0; gs.song_wrong=[]; gs.song_history=[]
                    gs.song_phase="detect"; gs.song_pt=t_now
                    gs.song_best_pred=""; gs.song_best_conf=0.0
                    gs.song_pred=""; gs.song_conf=0.0; gs.song_spoken=-1
                    gs.song_buf=deque(maxlen=SEQ_LEN)
                    music_play(0.0)
                elif d=="train_letters":
                    gs.screen="train"; gs.train_mode="letters"
                    gs.train_items=list(LETTERS)
                    gs.train_stars=[0]*len(LETTERS)
                    gs.train_idx=0; gs.train_hold=0; gs.train_steps=0
                    gs.train_det=""; gs.train_conf=0.0
                    gs.train_buf=deque(maxlen=SEQ_LEN)
                elif d=="train_words":
                    gs.screen="train"; gs.train_mode="words"
                    gs.train_items=list(WORDS)
                    gs.train_stars=[0]*len(WORDS)
                    gs.train_idx=0; gs.train_hold=0; gs.train_steps=0
                    gs.train_det=""; gs.train_conf=0.0
                    gs.train_buf=deque(maxlen=SEQ_LEN)

            elif gs.screen=="result":
                if gs.difficulty in ("debutant","intermediaire","expert"):
                    start_game(gs)

            elif gs.screen=="song" and gs.song_phase=="score":
                music_stop()
                gs.song_idx=0; gs.song_score=0; gs.song_wrong=[]; gs.song_history=[]
                gs.song_phase="detect"; gs.song_pt=t_now
                gs.song_best_pred=""; gs.song_best_conf=0.0
                gs.song_pred=""; gs.song_conf=0.0; gs.song_spoken=-1
                gs.song_buf=deque(maxlen=SEQ_LEN)
                music_play(0.0)

            elif gs.screen=="train" and gs.train_items:
                gs.train_idx=min(len(gs.train_items)-1, gs.train_idx+1)
                gs.train_hold=0; gs.train_steps=0
                gs.train_det=""; gs.train_conf=0.0
                gs.train_buf=deque(maxlen=SEQ_LEN)

        elif k==ord('s') and gs.screen=="playing" and not gs.paused:
            advance_item(gs,"skip")

        elif k==81 and gs.screen=="train" and gs.train_items:   # left arrow
            gs.train_idx=max(0,gs.train_idx-1)
            gs.train_hold=0; gs.train_steps=0
            gs.train_det=""; gs.train_conf=0.0
            gs.train_buf=deque(maxlen=SEQ_LEN)

        elif k==ord('v') and gs.screen=="train" and gs.train_items:
            item=gs.train_items[gs.train_idx]
            url=LETTER_VIDEO if gs.train_mode=="letters" else WORD_VIDEO.get(item,
                f"https://www.lifeprint.com/asl101/pages-signs/{item[0].lower()}/{item.lower().replace('_','')}.htm")
            webbrowser.open(url)

    cam.stop()
    hands.close()
    holistic.close()
    music_stop()
    cv2.destroyAllWindows()
    print("Thanks for playing JUST SIGN!")


if __name__ == "__main__":
    main()
