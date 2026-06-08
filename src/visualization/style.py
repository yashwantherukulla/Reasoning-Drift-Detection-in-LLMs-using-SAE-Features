import matplotlib.pyplot as plt

PAIR_COLORS = {
    ("clean", "helpful_hint"): "#2196F3",
    ("clean", "misleading_hint"): "#F44336",
    ("helpful_hint", "misleading_hint"): "#7B1FA2",
}

PAIR_LABELS = {
    ("clean", "helpful_hint"): "clean ↔ helpful",
    ("clean", "misleading_hint"): "clean ↔ misleading",
    ("helpful_hint", "misleading_hint"): "helpful ↔ misleading",
}

PAIR_MARKERS = {
    ("clean", "helpful_hint"): "o",
    ("clean", "misleading_hint"): "s",
    ("helpful_hint", "misleading_hint"): "^",
}

CAT_COLORS = {
    "arithmetic": "#FF9800",
    "gsm8k": "#4CAF50",
    "logical": "#2196F3",
    "symbolic": "#9C27B0",
    "arith": "#FF9800",  # fallbacks
    "logic": "#2196F3",
    "symb": "#9C27B0",
}

LAYERS = [6, 12, 18, 20, 24, 27]

def set_global_style(dpi=150):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": "white",
        }
    )

def add_category(df):
    df = df.copy()
    df["category"] = df["problem_id"].str.extract(r"^([a-z]+)")
    df["category"] = df["category"].replace({
        "gsm": "gsm8k", 
        "arith": "arithmetic", 
        "logic": "logical", 
        "symb": "symbolic"
    })
    return df
