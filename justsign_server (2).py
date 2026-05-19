"""
JUST SIGN — WebSocket Backend
==============================
Lance la détection MediaPipe + webcam et envoie l'état en temps réel
au frontend HTML via WebSocket.

Usage:
    python justsign_server.py [--mp3 chemin/vers/bohemian.mp3]

Le frontend s'ouvre automatiquement dans le navigateur.
"""

import cv2, mediapipe as mp, numpy as np
import time, math, pickle, threading, argparse, webbrowser, random, json, base64
import asyncio, websockets
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from collections import deque

# ── OPTIONAL IMPORTS ──────────────────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False

try:
    import pyttsx3
    _tts = pyttsx3.init()
    _tts.setProperty('rate', 155)
    _HAS_TTS = True
    _tts_lock = threading.Lock()
except Exception:
    _HAS_TTS = False

# ── GAME DATA (identique à v5) ─────────────────────────────
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
WORDS   = ["HELLO","THANK_YOU","PLEASE","SORRY","YES","NO",
           "HELP","LOVE","FRIEND","FAMILY","EAT","DRINK",
           "WATER","NAME","GOOD","BAD"]
SEQ_LEN = 30
SONG_DETECT_SEC = 4.0

SONG_LYRICS = [
    {"text":"Is this the real life?","sign":"REAL"},
    {"text":"Is this just fantasy?","sign":"DREAM"},
    {"text":"Caught in a landslide","sign":"STUCK"},
    {"text":"No escape from reality","sign":"ESCAPE"},
    {"text":"Open your eyes","sign":"LOOK"},
    {"text":"Look up to the skies","sign":"LOOK"},
    {"text":"I'm just a poor boy","sign":"POOR"},
    {"text":"I need no sympathy","sign":"PITY"},
    {"text":"Easy come, easy go","sign":"HAPPEN"},
    {"text":"Little high, little low","sign":"SOMETIMES"},
    {"text":"Anyway the wind blows","sign":"WHATEVER"},
    {"text":"Doesn't really matter to me","sign":"DON'T-MATTER"},
    {"text":"Mama,","sign":"MOTHER"},
    {"text":"just killed a man","sign":"KILL-HIM"},
    {"text":"Put a gun against his head","sign":"PAIN"},
    {"text":"now he's dead","sign":"DEATH"},
    {"text":"Mama, life had just begun","sign":"LIFE"},
    {"text":"But now I've gone","sign":"LEAVE"},
    {"text":"and thrown it all away","sign":"RUIN"},
    {"text":"Mama, ooh","sign":"MOTHER"},
    {"text":"Didn't mean to make you cry","sign":"CRY"},
    {"text":"If I'm not back again tomorrow","sign":"CONTINUE"},
    {"text":"Carry on, carry on","sign":"CONTINUE"},
    {"text":"as if nothing really matters","sign":"DON'T-MATTER"},
]

LETTER_HINTS = {
    "A":"Fist, thumb on side","B":"4 fingers up, thumb folded",
    "C":"Curved C shape","D":"Index up, others in circle",
    "E":"Fingers bent like a claw","F":"Thumb-index circle, 3 up",
    "G":"Index+thumb horizontal","H":"Index+middle horizontal",
    "I":"Pinky only","K":"Index+middle V, thumb between",
    "L":"Index up + thumb out","M":"Thumb under 3 fingers",
    "N":"Thumb under 2 fingers","O":"All fingers in O shape",
    "P":"Index pointing down","Q":"Index+thumb pointing down",
    "R":"Crossed index+middle","S":"Fist, thumb over fingers",
    "T":"Thumb between index+middle","U":"Index+middle together up",
    "V":"Index+middle V shape","W":"3 fingers spread",
    "X":"Hooked index finger","Y":"Thumb+pinky out",
}

WORD_HINTS = {
    "HELLO":"Open hand, forehead to out","THANK_YOU":"Flat hand, chin forward",
    "PLEASE":"Hand circle on chest","SORRY":"A-fist circle on chest",
    "YES":"Fist nodding","NO":"Index+middle snap to thumb",
    "HELP":"Fist on palm, lift up","LOVE":"Arms crossed on chest",
    "FRIEND":"Hooked index fingers link","FAMILY":"F-hands make circle",
    "EAT":"Pinched hand to mouth","DRINK":"C-hand to mouth, tilt",
    "WATER":"W taps chin twice","NAME":"H-hands cross each other",
    "GOOD":"Flat hand chin forward","BAD":"Hand mouth, flip down",
}

SIGN_HINTS = {
    "REAL":"Index from lips forward","DREAM":"Bent index from temple out",
    "STUCK":"V-hand blocked under left","ESCAPE":"Index escapes from fist",
    "LOOK":"V at eyes then forward","POOR":"Hand slides under left elbow",
    "PITY":"Middle finger circles on chest","HAPPEN":"Two index fingers pivot down",
    "SOMETIMES":"Index circles then S","WHATEVER":"Two W-hands spread apart",
    "DON'T-MATTER":"Flat hand waves dismissively","MOTHER":"Thumb taps chin",
    "KILL-HIM":"Index slices under palm","DEATH":"Flat hand flips over",
    "LIFE":"Two L-hands rise up torso","LEAVE":"Open hand sweeps outward",
    "RUIN":"Two R-hands rub downward","CRY":"Index fingers trace tears on cheeks",
    "CONTINUE":"Two thumbs slide forward",
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
    "DRINK":    ["Curve hand like holding a cup","Bring to mouth","Tilt like drinking"],
    "WATER":    ["W shape with 3 fingers","Tap chin twice","W + two taps on chin"],
    "NAME":     ["Index+middle horizontal (H)","Cross with other hand same","Two H hands cross"],
    "GOOD":     ["Flat hand at chin","Move forward and downward","Like sending something forward"],
    "BAD":      ["Flat hand at mouth","Flip hand downward and away","End with palm facing up"],
}

LETTER_VIDEO = "https://www.youtube.com/watch?v=tkMg8g8vVUo"
WORD_VIDEO = {k:"https://www.lifeprint.com/asl101/pages-signs/{}/{}.htm".format(
    k[0].lower(), k.lower().replace('_','')) for k in WORDS}

# ── MODELS ────────────────────────────────────────────────
_letter_model = None
_word_model   = None

def load_models():
    global _letter_model, _word_model
    for p in ["asl_model.pkl", Path(__file__).parent/"asl_model.pkl"]:
        if Path(p).exists():
            try:
                with open(p,"rb") as f: _letter_model=pickle.load(f)
                print("✓ Letter model loaded")
            except: pass
            break
    for p in ["asl_words_model.pkl", Path(__file__).parent/"asl_words_model.pkl"]:
        if Path(p).exists():
            try:
                with open(p,"rb") as f: _word_model=pickle.load(f)
                print("✓ Word model loaded")
            except: pass
            break

# ── DETECTION (identique à v5) ─────────────────────────────
def _feats(lm):
    pts=np.array([[l.x,l.y,l.z] for l in lm.landmark])
    w=pts[0]; pts-=w
    sc=np.max(np.abs(pts)); pts/=(sc+1e-7)
    return pts.flatten().tolist()

def detect_letter(lm, hand_label="Right"):
    if _letter_model:
        feats=_feats(lm)
        try:
            proba=_letter_model.predict_proba([feats])[0]
            idx=int(np.argmax(proba)); conf=float(proba[idx])
            if conf<0.45: return None,conf,None
            label=_letter_model.classes_[idx]
            return label,conf,None
        except: pass
    # Geometric fallback (simplified)
    pts=np.array([[l.x,l.y] for l in lm.landmark])
    tip=[pts[8],pts[12],pts[16],pts[20]]
    mcp=[pts[5],pts[9],pts[13],pts[17]]
    ext=[t[1]<m[1] for t,m in zip(tip,mcp)]
    thumb_ext=pts[4][0]<pts[3][0] if hand_label=="Right" else pts[4][0]>pts[3][0]
    n_ext=sum(ext)
    if n_ext==0 and not thumb_ext: return "A",0.6,None
    if n_ext==4 and not thumb_ext: return "B",0.6,None
    if n_ext==4 and thumb_ext: return "B",0.55,None
    if n_ext==1 and ext[0]: return "D",0.55,None
    if n_ext==0 and thumb_ext: return "S",0.55,None
    return None,0.0,None

def extract_holistic_feats(results):
    # Copie exacte de justsign_v4.py — ne pas modifier
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
    if not _word_model or seq_feats is None: return None,0.0
    X=_word_model["scaler"].transform([seq_feats]); pr=_word_model["model"].predict_proba(X)[0]
    idx=int(np.argmax(pr)); c=float(pr[idx])
    if c<0.5: return None,c
    return _word_model["label_encoder"].inverse_transform([idx])[0],c

# ── SOUND ─────────────────────────────────────────────────
_mp3_path = None

def music_play(offset=0.0):
    if _HAS_PYGAME and _mp3_path and Path(_mp3_path).exists():
        try:
            pygame.mixer.music.load(_mp3_path)
            pygame.mixer.music.play(start=offset)
        except: pass

def music_stop():
    if _HAS_PYGAME:
        try: pygame.mixer.music.stop()
        except: pass

def speak(text):
    if _HAS_TTS:
        def _s():
            with _tts_lock: _tts.say(text); _tts.runAndWait()
        threading.Thread(target=_s, daemon=True).start()

def beep_ok():
    if _HAS_PYGAME:
        try:
            sr=44100; d=0.12; f=880
            t=np.linspace(0,d,int(sr*d),False)
            w=(np.sin(2*np.pi*f*t)*32767*0.4).astype(np.int16)
            s=pygame.sndarray.make_sound(np.column_stack([w,w]))
            s.play()
        except: pass

def beep_fail():
    if _HAS_PYGAME:
        try:
            sr=44100; d=0.18; f=220
            t=np.linspace(0,d,int(sr*d),False)
            w=(np.sin(2*np.pi*f*t)*32767*0.3).astype(np.int16)
            s=pygame.sndarray.make_sound(np.column_stack([w,w]))
            s.play()
        except: pass

# ── GAME STATE ────────────────────────────────────────────
@dataclass
class GS:
    screen: str = "menu"
    menu_sel: int = 0
    paused: bool = False
    mode: str = ""
    # Train
    train_mode: str = "letters"
    train_items: list = field(default_factory=list)
    train_idx: int = 0
    train_hold: int = 0
    train_steps: int = 0
    train_det: str = ""
    train_conf: float = 0.0
    train_stars: list = field(default_factory=list)
    train_buf: deque = field(default_factory=lambda: deque(maxlen=SEQ_LEN))
    # Song
    song_idx: int = 0
    song_score: int = 0
    song_phase: str = "detect"
    song_pt: float = 0.0
    song_pred: str = ""
    song_conf: float = 0.0
    song_best_pred: str = ""
    song_best_conf: float = 0.0
    song_spoken: int = -1
    song_wrong: list = field(default_factory=list)
    song_history: list = field(default_factory=list)
    song_buf: deque = field(default_factory=lambda: deque(maxlen=SEQ_LEN))

# ── CAMERA ────────────────────────────────────────────────
class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._display = None
        self._mp_raw  = None
        self._lock    = threading.Lock()
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self._running:
            ret, f = self.cap.read()
            if ret:
                f = cv2.resize(f, (640, 480))
                with self._lock:
                    self._mp_raw    = f.copy()
                    self._display   = cv2.flip(f, 1)

    def get(self):
        with self._lock:
            if self._display is None: return None, None
            return self._display.copy(), self._mp_raw.copy()

    def stop(self):
        self._running = False
        self.cap.release()

# ── WEBSOCKET SERVER ──────────────────────────────────────
_clients: set = set()
_gs: GS = None
_cam: Camera = None
_cam_jpeg: bytes = b""
_cam_lock = threading.Lock()

def frame_to_jpeg(frame, quality=60):
    """Encode un frame OpenCV en JPEG base64."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode('ascii')

def build_state(gs: GS, cam_b64: str) -> str:
    """Construit le JSON d'état complet à envoyer au frontend."""
    state = {
        "screen": gs.screen,
        "cam": cam_b64,
    }

    if gs.screen == "menu":
        state["menu"] = {"sel": gs.menu_sel}

    elif gs.screen == "train":
        item = gs.train_items[gs.train_idx] if gs.train_items else ""
        is_letters = gs.train_mode == "letters"
        steps_data = LETTER_STEPS.get(item, []) if is_letters else WORD_STEPS.get(item, [])
        hint = LETTER_HINTS.get(item, "") if is_letters else WORD_HINTS.get(item, "")
        video_url = LETTER_VIDEO if is_letters else WORD_VIDEO.get(item, "")
        stars_list = gs.train_stars if gs.train_stars else []

        state["train"] = {
            "mode":       gs.train_mode,
            "item":       item,
            "idx":        gs.train_idx,
            "total":      len(gs.train_items),
            "hold":       gs.train_hold,
            "hold_max":   60,
            "steps":      steps_data,
            "steps_done": gs.train_steps,
            "hint":       hint,
            "det":        gs.train_det,
            "conf":       round(gs.train_conf * 100),
            "correct":    gs.train_det == item,
            "stars":      stars_list[:gs.train_idx+1] if stars_list else [],
            "video_url":  video_url,
            "paused":     gs.paused,
        }

    elif gs.screen == "song":
        total = len(SONG_LYRICS)
        lyric = SONG_LYRICS[gs.song_idx] if gs.song_idx < total else {}
        elapsed = time.time() - gs.song_pt
        rem = max(0.0, SONG_DETECT_SEC - elapsed) if gs.song_phase == "detect" else 0.0
        hp = int(gs.song_score / max(1, gs.song_idx + 1) * 100)

        state["song"] = {
            "idx":       gs.song_idx,
            "total":     total,
            "score":     gs.song_score,
            "phase":     gs.song_phase,
            "lyric":     lyric.get("text", ""),
            "sign":      lyric.get("sign", ""),
            "hint":      SIGN_HINTS.get(lyric.get("sign",""), ""),
            "rem":       round(rem, 1),
            "pred":      gs.song_pred,
            "conf":      round(gs.song_conf * 100),
            "correct":   gs.song_pred == lyric.get("sign",""),
            "accuracy":  hp,
            "wrong":     gs.song_wrong,
            "paused":    gs.paused,
            "next_lyric": SONG_LYRICS[gs.song_idx+1]["text"] if gs.song_idx+1 < total else "",
        }

    return json.dumps(state)

async def ws_handler(ws):
    """Gère une connexion WebSocket."""
    global _clients
    _clients.add(ws)
    print(f"✓ Browser connected ({len(_clients)} clients)")
    try:
        async for msg in ws:
            handle_command(json.loads(msg))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _clients.discard(ws)
        print(f"Browser disconnected ({len(_clients)} clients)")

def handle_command(cmd: dict):
    """Traite les commandes envoyées par le frontend."""
    global _gs
    gs = _gs
    action = cmd.get("action", "")
    t_now = time.time()

    if action == "key":
        k = cmd.get("key", "")
        if k == "quit":
            import os; os._exit(0)
        elif k == "menu":
            music_stop(); gs.screen = "menu"; gs.paused = False
        elif k == "pause" and gs.screen in ("train", "song"):
            gs.paused = not gs.paused
        elif k == "sel" and gs.screen == "menu":
            gs.menu_sel = cmd.get("val", 0)
        elif k == "left" and gs.screen == "menu":
            gs.menu_sel = (gs.menu_sel - 1) % 3
        elif k == "right" and gs.screen == "menu":
            gs.menu_sel = (gs.menu_sel + 1) % 3
        elif k == "play" and gs.screen == "menu":
            sel = gs.menu_sel
            if sel == 0:
                gs.screen = "train"; gs.train_mode = "letters"; gs.mode = "letters"
                gs.train_items = list(LETTERS); gs.train_stars = [0]*len(LETTERS)
                gs.train_idx = 0; gs.train_hold = 0; gs.train_steps = 0
                gs.train_det = ""; gs.train_conf = 0.0
                gs.train_buf = deque(maxlen=SEQ_LEN)
            elif sel == 1:
                gs.screen = "train"; gs.train_mode = "words"; gs.mode = "words"
                gs.train_items = list(WORDS); gs.train_stars = [0]*len(WORDS)
                gs.train_idx = 0; gs.train_hold = 0; gs.train_steps = 0
                gs.train_det = ""; gs.train_conf = 0.0
                gs.train_buf = deque(maxlen=SEQ_LEN)
            elif sel == 2:
                music_stop(); gs.screen = "song"; gs.mode = "song"
                gs.song_idx = 0; gs.song_score = 0; gs.song_wrong = []
                gs.song_history = []; gs.song_phase = "detect"
                gs.song_pt = t_now; gs.song_best_pred = ""
                gs.song_best_conf = 0.0; gs.song_pred = ""
                gs.song_conf = 0.0; gs.song_spoken = -1
                gs.song_buf = deque(maxlen=SEQ_LEN); music_play(0.0)
        elif k == "next" and gs.screen == "train" and gs.train_items:
            gs.train_idx = min(len(gs.train_items)-1, gs.train_idx+1)
            gs.train_hold = 0; gs.train_steps = 0
            gs.train_det = ""; gs.train_conf = 0.0
            gs.train_buf = deque(maxlen=SEQ_LEN)
        elif k == "prev" and gs.screen == "train" and gs.train_items:
            gs.train_idx = max(0, gs.train_idx-1)
            gs.train_hold = 0; gs.train_steps = 0
            gs.train_det = ""; gs.train_conf = 0.0
            gs.train_buf = deque(maxlen=SEQ_LEN)
        elif k == "video" and gs.screen == "train" and gs.train_items:
            item = gs.train_items[gs.train_idx]
            url = LETTER_VIDEO if gs.train_mode=="letters" else WORD_VIDEO.get(item,"")
            if url: webbrowser.open(url)
        elif k == "song_replay" and gs.screen == "song" and gs.song_phase == "score":
            music_stop()
            gs.song_idx = 0; gs.song_score = 0; gs.song_wrong = []
            gs.song_history = []; gs.song_phase = "detect"
            gs.song_pt = t_now; gs.song_best_pred = ""; gs.song_best_conf = 0.0
            gs.song_pred = ""; gs.song_conf = 0.0; gs.song_spoken = -1
            gs.song_buf = deque(maxlen=SEQ_LEN); music_play(0.0)

async def broadcast_loop():
    """Envoie l'état au frontend ~30fps."""
    global _clients, _gs, _cam_jpeg
    while True:
        if _clients and _gs is not None:
            with _cam_lock:
                cam_b64 = frame_to_jpeg(_cam_jpeg) if isinstance(_cam_jpeg, np.ndarray) else ""
            msg = build_state(_gs, cam_b64)
            dead = set()
            for ws in list(_clients):
                try:
                    await ws.send(msg)
                except:
                    dead.add(ws)
            _clients -= dead
        await asyncio.sleep(1/30)

async def main_async():
    async with websockets.serve(ws_handler, "localhost", 8765):
        print("✓ WebSocket server on ws://localhost:8765")
        await broadcast_loop()

def run_ws_server():
    asyncio.run(main_async())

# ── MAIN ──────────────────────────────────────────────────
def main():
    global _gs, _cam, _cam_jpeg, _mp3_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--mp3", default=None)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    _mp3_path = args.mp3

    load_models()

    mp_h   = mp.solutions.hands
    mp_hol = mp.solutions.holistic
    mp_draw= mp.solutions.drawing_utils
    hands  = mp_h.Hands(static_image_mode=False, max_num_hands=1,
                        model_complexity=0, min_detection_confidence=0.5,
                        min_tracking_confidence=0.4)
    holistic = mp_hol.Holistic(static_image_mode=False, model_complexity=1,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5,
                               refine_face_landmarks=True)

    stl = mp_draw.DrawingSpec(color=(34,211,238), thickness=2, circle_radius=3)
    stc = mp_draw.DrawingSpec(color=(255,45,149), thickness=1)

    _cam = Camera()
    _gs  = GS()

    # Démarrer le serveur WS dans un thread dédié
    ws_thread = threading.Thread(target=run_ws_server, daemon=True)
    ws_thread.start()

    # Attendre un peu puis ouvrir le navigateur
    time.sleep(1.2)
    html_path = Path(__file__).parent / "justsign_ui.html"
    if html_path.exists():
        webbrowser.open(f"file://{html_path.resolve()}")
        print(f"✓ Opening browser: {html_path}")
    else:
        print("⚠ justsign_ui.html not found next to this script")

    t_menu_p = 0.0
    print("✓ Detection loop running — Ctrl+C to quit")

    try:
        while True:
            t_now = time.time()
            gs = _gs

            disp, mp_raw = _cam.get()
            cam_display = disp.copy() if disp is not None else np.zeros((480,640,3),np.uint8)

            if disp is not None:
                if gs.screen == "train":
                    rgb = cv2.cvtColor(mp_raw, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    if gs.train_mode == "letters":
                        res = hands.process(rgb)
                        if res.multi_hand_landmarks:
                            hl = res.multi_hand_landmarks[0]
                            hand_label = "Right"
                            if res.multi_handedness:
                                hand_label = res.multi_handedness[0].classification[0].label
                            L, c, _ = detect_letter(hl, hand_label)
                            mp_draw.draw_landmarks(cam_display, hl,
                                mp_h.HAND_CONNECTIONS, stl, stc)
                            gs.train_det = L or ""; gs.train_conf = c
                        else:
                            gs.train_det = ""; gs.train_conf = 0.0
                    else:
                        r2 = holistic.process(rgb)
                        if r2.right_hand_landmarks:
                            mp_draw.draw_landmarks(cam_display, r2.right_hand_landmarks,
                                mp_hol.HAND_CONNECTIONS, stl, stc)
                        gs.train_buf.append(extract_holistic_feats(r2))
                        if len(gs.train_buf) >= SEQ_LEN:
                            sf = extract_seq_feats(list(gs.train_buf))
                            w, c = predict_word(sf)
                            gs.train_det = w or ""; gs.train_conf = c

                elif gs.screen == "song" and gs.song_phase == "detect":
                    rgb = cv2.cvtColor(mp_raw, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    r2 = holistic.process(rgb)
                    if r2.right_hand_landmarks:
                        mp_draw.draw_landmarks(cam_display, r2.right_hand_landmarks,
                            mp_hol.HAND_CONNECTIONS, stl, stc)
                    if r2.left_hand_landmarks:
                        mp_draw.draw_landmarks(cam_display, r2.left_hand_landmarks,
                            mp_hol.HAND_CONNECTIONS, stl, stc)
                    gs.song_buf.append(extract_holistic_feats(r2))
                    if len(gs.song_buf) >= SEQ_LEN:
                        sf = extract_seq_feats(list(gs.song_buf))
                        p, c = predict_word(sf)
                        gs.song_pred = p or ""; gs.song_conf = c
                        if p and c > gs.song_best_conf:
                            gs.song_best_pred = p; gs.song_best_conf = c

            with _cam_lock:
                _cam_jpeg = cam_display

            # Game logic — train
            if gs.screen == "train" and gs.train_items and not gs.paused:
                item = gs.train_items[gs.train_idx]
                if gs.train_det == item:
                    gs.train_hold = min(gs.train_hold+1, 120)
                    gs.train_steps = min(3, gs.train_hold//30)
                    stars = gs.train_hold//40
                    if gs.train_idx < len(gs.train_stars):
                        gs.train_stars[gs.train_idx] = min(3, max(
                            gs.train_stars[gs.train_idx], stars))
                else:
                    gs.train_hold = max(0, gs.train_hold-1)

            # Game logic — song
            elif gs.screen == "song" and not gs.paused:
                total = len(SONG_LYRICS)
                if gs.song_phase == "detect" and gs.song_idx < total:
                    lyric = SONG_LYRICS[gs.song_idx]
                    if gs.song_spoken != gs.song_idx:
                        speak(lyric["text"]); gs.song_spoken = gs.song_idx
                    elapsed = t_now - gs.song_pt
                    if elapsed >= SONG_DETECT_SEC:
                        target = lyric["sign"]
                        model_words = _word_model["words"] if _word_model else []
                        ok = True if target not in model_words else (
                            gs.song_best_pred == target and gs.song_best_conf >= 0.45)
                        gs.song_phase = "ok" if ok else "miss"
                        gs.song_history.append(ok)
                        if ok:
                            gs.song_score += 1; beep_ok()
                        else:
                            gs.song_wrong.append(target); beep_fail()
                        gs.song_pt = t_now
                elif gs.song_phase in ("ok", "miss"):
                    if t_now - gs.song_pt >= 1.2:
                        gs.song_idx += 1
                        if gs.song_idx >= total:
                            gs.song_phase = "score"; music_stop()
                        else:
                            gs.song_phase = "detect"; gs.song_pt = t_now
                            gs.song_best_pred = ""; gs.song_best_conf = 0.0
                            gs.song_pred = ""; gs.song_conf = 0.0
                            gs.song_buf = deque(maxlen=SEQ_LEN)

            time.sleep(1/60)

    except KeyboardInterrupt:
        print("\nQuit.")
    finally:
        _cam.stop()
        hands.close()
        holistic.close()
        music_stop()

if __name__ == "__main__":
    main()
