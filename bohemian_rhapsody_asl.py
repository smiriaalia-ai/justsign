"""
BOHEMIAN RHAPSODY — Base de données ASL
========================================
Tous les noms de signes correspondent EXACTEMENT aux mots
de la liste WORDS dans train_2__1_.py.

60 moments de jeu | 40 signes uniques
"""

SONG_INFO = {
    "title":    "Bohemian Rhapsody",
    "artist":   "Queen",
    "duration": 355,
    "key":      "BOHEMIAN_RHAPSODY",
}

# ─── Hints (repris depuis train_2__1_.py) ─────────────────────────────────────
SIGN_HINTS = {
    "REAL":         "Index droit part des lèvres vers l'avant",
    "DREAM":        "Index plié glisse depuis le front vers l'extérieur",
    "STUCK":        "V droit bloqué sous la main gauche",
    "ESCAPE":       "Index droit sort rapidement d'un poing fermé",
    "LOOK":         "V pointé vers les yeux puis vers l'objet",
    "POOR":         "Main droite glisse sous le coude gauche",
    "PITY":         "Majeur fait des cercles sur la poitrine",
    "HAPPEN":       "Deux index pointent vers le haut, pivotent vers le bas",
    "SOMETIMES":    "Index droit fait un cercle puis S",
    "WHATEVER":     "W des deux mains s'écartent",
    "DON'T-MATTER": "Main plate agitée devant soi",
    "MOTHER":       "Pouce de la main ouverte tapote le menton",
    "KILL-HIM":     "Index tranche sous la paume + pointage vers lui",
    "DEATH":        "Main plate se retourne (de face à dos)",
    "LIFE":         "Deux mains L remontent le long du torse",
    "LEAVE":        "Main ouverte, balayage vers l'extérieur",
    "RUIN":         "Deux R frottent l'un contre l'autre vers le bas",
    "CRY":          "Index tracent des larmes sur les joues",
    "CONTINUE":     "Deux pouces joints glissent vers l'avant",
    "LATE":         "Main derrière l'épaule, signe 'pas encore'",
    "AFRAID":       "Mains croisées devant la poitrine, écartement rapide",
    "PAIN":         "Index se touchent par le bout douloureusement",
    "GOODBYE":      "Main ouverte, salut de la main",
    "TRUTH":        "Index part des lèvres vers l'avant",
    "NOT-WANT":     "Mains en griffes retournées vers l'extérieur",
    "BORN":         "Main droite plate glisse depuis le ventre vers l'avant",
    "DANCE":        "Index et majeur 'dansent' sur la paume opposée",
    "THUNDER":      "T puis index zigzague (tonnerre)",
    "FEAR":         "Doigts écartés tremblent devant la poitrine",
    "LOVE-ME":      "Bras croisés sur la poitrine + pointage vers soi",
    "GOD":          "Main ouverte du front vers le bas",
    "FORBID":       "Poing droit frappe la paume gauche ouverte",
    "DEVIL":        "Mains en Y sur les tempes (cornes)",
    "PRESERVE":     "Deux mains P bougent vers le bas simultanément",
    "ME":           "Pointage index vers sa propre poitrine",
    "NO":           "Index + majeur se ferment sur le pouce",
    "NONE":         "Deux O devant soi, écartement vers les côtés",
    "SAD":          "Deux mains ouvertes descendent devant le visage",
    "RUN-AWAY":     "Index et pouce courent, puis balayage vers l'extérieur",
    "DIE":          "Main plate se retourne, paume vers le bas",
}

# ─── Paroles alignées — noms de signes = liste WORDS de train_2__1_.py ────────
SONG_LYRICS = [

    # ══ INTRO (0:00 → 0:49) ══════════════════════════════
    {"text": "Is this the real life?",           "sign": "REAL",         "t": 1.0},
    {"text": "Is this just fantasy?",            "sign": "DREAM",        "t": 6.0},
    {"text": "Caught in a landslide",            "sign": "STUCK",        "t": 11.0},
    {"text": "No escape from reality",           "sign": "ESCAPE",       "t": 16.0},
    {"text": "Open your eyes",                   "sign": "LOOK",         "t": 21.0},
    {"text": "Look up to the skies and see",     "sign": "LOOK",         "t": 26.0},
    {"text": "I'm just a poor boy",              "sign": "POOR",         "t": 31.0},
    {"text": "I need no sympathy",               "sign": "PITY",         "t": 36.0},
    {"text": "Because it's easy come, easy go",  "sign": "HAPPEN",       "t": 41.0},
    {"text": "Little high, little low",          "sign": "SOMETIMES",    "t": 46.0},
    {"text": "Anyway the wind blows",            "sign": "WHATEVER",     "t": 51.0},
    {"text": "Doesn't really matter to me",      "sign": "DON'T-MATTER", "t": 56.0},

    # ══ COUPLET 1 (0:49 → 1:37) ══════════════════════════
    {"text": "Mama,",                            "sign": "MOTHER",       "t": 62.0},
    {"text": "just killed a man",                "sign": "KILL-HIM",     "t": 67.0},
    {"text": "Put a gun against his head",       "sign": "PAIN",         "t": 72.0},
    {"text": "now he's dead",                    "sign": "DEATH",        "t": 79.0},
    {"text": "Mama, life had just begun",        "sign": "LIFE",         "t": 87.0},
    {"text": "But now I've gone",                "sign": "LEAVE",        "t": 92.0},
    {"text": "and thrown it all away",           "sign": "RUIN",         "t": 97.0},

    # ══ REFRAIN 1 (1:37 → 2:17) ══════════════════════════
    {"text": "Mama, ooh",                        "sign": "MOTHER",       "t": 107.0},
    {"text": "Didn't mean to make you cry",      "sign": "CRY",          "t": 112.0},
    {"text": "If I'm not back again tomorrow",   "sign": "CONTINUE",     "t": 117.0},
    {"text": "Carry on, carry on",               "sign": "CONTINUE",     "t": 122.0},
    {"text": "as if nothing really matters",     "sign": "DON'T-MATTER", "t": 127.0},

    # ══ COUPLET 2 (2:17 → 2:43) ══════════════════════════
    {"text": "Too late, my time has come",       "sign": "LATE",         "t": 150.0},
    {"text": "Sends shivers down my spine",      "sign": "AFRAID",       "t": 155.0},
    {"text": "Body's aching all the time",       "sign": "PAIN",         "t": 160.0},
    {"text": "Goodbye, everybody",               "sign": "GOODBYE",      "t": 165.0},
    {"text": "I've got to go",                   "sign": "LEAVE",        "t": 170.0},
    {"text": "Gotta leave you all behind",       "sign": "RUN-AWAY",     "t": 175.0},
    {"text": "and face the truth",               "sign": "TRUTH",        "t": 180.0},

    # ══ REFRAIN 2 (2:43 → 3:15) ══════════════════════════
    {"text": "Mama, ooh (anyway the wind blows)","sign": "MOTHER",       "t": 188.0},
    {"text": "I don't want to die",              "sign": "NOT-WANT",     "t": 193.0},
    {"text": "I sometimes wish I'd never",       "sign": "SOMETIMES",    "t": 198.0},
    {"text": "been born at all",                 "sign": "BORN",         "t": 203.0},

    # ══ SECTION OPÉRA (3:15 → 4:07) — rythme rapide ═════
    {"text": "I see a little silhouette",        "sign": "LOOK",         "t": 213.0},
    {"text": "Scaramouche, will you do the",     "sign": "DANCE",        "t": 217.0},
    {"text": "Thunderbolts and lightning",       "sign": "THUNDER",      "t": 221.0},
    {"text": "Very, very frightening me",        "sign": "FEAR",         "t": 225.0},
    {"text": "nobody loves me",                  "sign": "LOVE-ME",      "t": 233.0},
    {"text": "Bismillah! No!",                   "sign": "GOD",          "t": 241.0},
    {"text": "We will not let you go",           "sign": "FORBID",       "t": 244.0},
    {"text": "Let him go!",                      "sign": "LEAVE",        "t": 247.0},
    {"text": "Beelzebub has a devil",            "sign": "DEVIL",        "t": 251.0},
    {"text": "put aside for me",                 "sign": "PRESERVE",     "t": 255.0},

    # ══ SECTION ROCK (4:07 → 4:55) ═══════════════════════
    {"text": "So you think you can stone me",    "sign": "PAIN",         "t": 260.0},
    {"text": "and spit in my eye",               "sign": "ME",           "t": 265.0},
    {"text": "So you think you can love me",     "sign": "LOVE-ME",      "t": 269.0},
    {"text": "and leave me to die",              "sign": "DIE",          "t": 274.0},
    {"text": "Oh baby, can't do this to me",     "sign": "NO",           "t": 279.0},
    {"text": "Just gotta get out",               "sign": "ESCAPE",       "t": 284.0},

    # ══ OUTRO (4:55 → 5:55) ══════════════════════════════
    {"text": "Nothing really matters",           "sign": "NONE",         "t": 305.0},
    {"text": "Anyone can see",                   "sign": "LOOK",         "t": 311.0},
    {"text": "Nothing really matters",           "sign": "DON'T-MATTER", "t": 317.0},
    {"text": "Nothing really matters to me",     "sign": "SAD",          "t": 323.0},
    {"text": "Anyway the wind blows...",         "sign": "WHATEVER",     "t": 333.0},
]


def get_required_signs():
    """Retourne les signes uniques nécessaires — tous présents dans train_2__1_.py."""
    return sorted(set(l["sign"] for l in SONG_LYRICS))


def print_summary():
    signs = get_required_signs()
    print(f"\n🎸 {SONG_INFO['title']} — {SONG_INFO['artist']}")
    print(f"   {len(SONG_LYRICS)} moments de jeu | {len(signs)} signes uniques")
    print(f"\n📋 Signes à entraîner dans train_2__1_.py ({len(signs)}) :")
    for i, s in enumerate(signs, 1):
        hint = SIGN_HINTS.get(s, "—")
        print(f"   {i:>2}. {s:<20} {hint}")


if __name__ == "__main__":
    print_summary()
