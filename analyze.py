"""
Project 1 - Decoding Parameter & Sampling Analysis
Results analysis: CSV files -> tables + graphs (determinism / hallucination / structure)

Folder layout expected (same as your project):
    analyze_results.py
    prompts/prompts.json
    results/baseline_results.csv, temperature_results.csv, top_k_results.csv, top_p_results.csv

Run:
    pip install pandas numpy matplotlib openpyxl
    python analyze_results.py

Output goes to  results/analysis/{figures,tables}/

How the experiment is read
--------------------------
* baseline_results.csv holds the default setting (T=0.7, top_k=40, top_p=0.9).
  It is re-used as the "baseline" point inside every sweep (the sweep files do
  not contain it), so every curve passes through the baseline.
* Each (prompt, setting) was run 5 times with seeds 1..5.  "Determinism" therefore
  means: how similar are the answers across these 5 seeds at the same setting.
* Hallucination / structure are scored automatically (see the rules below).  The
  hallucination labels are HEURISTIC; a random sample is exported to
  tables/manual_review_sample.csv so you can check them by hand and report
  the agreement in your PDF.
"""
import json
import re
import sys
import warnings
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================== CONFIG
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "results"
PROMPTS_FILE = BASE_DIR / "prompts" / "prompts.json"
OUT_DIR = RESULTS_DIR / "analysis"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

GROUPS = ["temperature", "top_k", "top_p"]
GROUP_NAME = {"temperature": "Temperature", "top_k": "Top-K", "top_p": "Top-P"}
BASELINE = {"temperature": 0.7, "top_k": 40, "top_p": 0.9}  # must match common.py

MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# set in main()
MODELS, LABEL, COL, XS, CAT_N = [], {}, {}, {}, Counter()

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "axes.axisbelow": True, "figure.dpi": 110, "savefig.dpi": 170,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# ============================================================== TEXT METRICS
WORD = re.compile(r"\w+", re.UNICODE)
FOREIGN = re.compile(r"[Ѐ-ӿ؀-ۿ぀-ヿ㐀-鿿가-힯]")
END_OK = re.compile(r"[.!?。！？\"'”’)\]}`*]$")
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def toks(s):
    return WORD.findall(s.lower())


def pair_sim(a, b):
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def distinct_n(s, n=2):
    t = toks(s)
    if len(t) < n:
        return 1.0
    grams = list(zip(*[t[i:] for i in range(n)]))
    return len(set(grams)) / len(grams)


def n_sentences(s):
    parts = re.split(r"(?<=[.!?])\s+|(?<=[。！？])", s.strip())
    return len([p for p in parts if p.strip()])


def expected_sentences(prompt):
    m = re.search(r"\b(one|two|three|four|five|six|\d+)\s+sentences?\b", prompt, re.I)
    if not m:
        return None
    v = m.group(1).lower()
    return NUM_WORDS.get(v) or int(v)


# ---------------- hallucination labelling (heuristic, rule-based)
def _rx(parts):
    return re.compile("|".join(parts), re.I)


# the answer admits the thing does not exist / is unknown
EXISTENCE = _rx([
    r"\bnot (aware|familiar)\b", r"\b(couldn'?t|could not|can'?t|cannot|unable to) find\b",
    r"\bno (such|record|evidence|country|known)\b", r"\b(doesn'?t|does not|don'?t|do not) (really )?(exist|correspond)\b",
    r"\bnon-?existent\b", r"\bfictional\b", r"\bmade-?up\b", r"\bnot a real\b",
    r"\bnever (been|happened)\b", r"\bhasn'?t (been|even)\b", r"\bnot (an? )?(real|recognized|existing)\b",
])
# the answer admits it lacks the knowledge / ability
KNOWLEDGE_GAP = _rx([
    r"\b(don'?t|do not) have (any |that |this |the |specific |enough |access|real-?time|information)",
    r"\b(can'?t|cannot|not able to|unable to) (predict|verify|confirm|access|browse)",
    r"\b(don'?t|do not) (actually )?know\b", r"\bnot sure\b",
])
# policy-style refusal (not an honest "I don't know")
REFUSAL = _rx([
    r"\b(can'?t|cannot|won'?t|not able to|unable to) (assist|help|answer|engage|comment|summari[sz]e|provide)",
    r"\bsensitive\b", r"\bpolitical\b", r"\bcopyright", r"\bintellectual property\b", r"\binfring",
])
# fabricated detail given even though the answer hedged first
FABRICATION = _rx([
    r"\bhowever,? i (did )?find\b", r"\bhowever,? i can provide you with (information|details)",
    r"\bfictional character (created|from|by)\b", r"\bpseudonym\b",
    r"\bwas first (proved|proven|introduced|published)\b", r"\bis also known as\b", r"\bare affiliated with\b",
])


def label_hallucination(t):
    """hallucinated = confident (or hedge-then-invent) answer about something that does not exist;
    abstain = honest 'does not exist / I don't know';  refusal = policy-style refusal."""
    if not t:
        return "empty"
    if FABRICATION.search(t):
        return "hallucinated"
    if EXISTENCE.search(t):
        return "abstain"
    if REFUSAL.search(t):
        return "refusal"
    if KNOWLEDGE_GAP.search(t):
        return "abstain"
    return "hallucinated"


# ---------------- JSON structure
FENCED = re.compile(r"^```[\w-]*\s*\n(.*?)\n?```$", re.S)


def parse_json_response(t):
    """-> (format, object).  pure = raw JSON only | fenced = ```json block only |
    prose = JSON found inside extra text | invalid"""
    try:
        return "pure", json.loads(t)
    except Exception:
        pass
    m = FENCED.match(t)
    if m:
        try:
            return "fenced", json.loads(m.group(1))
        except Exception:
            pass
    for pat in (r"```[\w-]*\s*\n?(.*?)```", r"(\{.*\})", r"(\[.*\])"):
        for mm in re.finditer(pat, t, re.S):
            try:
                return "prose", json.loads(mm.group(1))
            except Exception:
                continue
    return "invalid", None


def all_keys(o, acc=None):
    acc = set() if acc is None else acc
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(str(k).lower())
            all_keys(v, acc)
    elif isinstance(o, list):
        for v in o:
            all_keys(v, acc)
    return acc


# ============================================================== LOADING & SCORING
RESP_COLS = ["acc", "hall", "abstain", "refusal", "empty_resp", "fmt_pure", "fmt_fenced", "fmt_prose",
             "fmt_invalid", "json_parsable", "schema_exact", "schema_lenient", "sent_ok", "truncated",
             "foreign", "n_words", "distinct2"]


def read_csv(name):
    p = RESULTS_DIR / name
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    df = pd.read_csv(p, encoding="utf-8-sig")
    df["response"] = df["response"].fillna("").astype(str)
    df["error"] = df["error"].fillna("").astype(str)
    return df


def load_prompts():
    if not PROMPTS_FILE.exists():
        raise SystemExit(f"prompts.json not found: {PROMPTS_FILE}")
    items = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    return {int(p["id"]): p for p in items}


def load_data():
    base = read_csv("baseline_results.csv")
    frames = []
    for g in GROUPS:
        sweep = read_csv(f"{g}_results.csv")
        for df, is_base in ((sweep, False), (base, True)):
            d = df.copy()
            d["group"], d["x"], d["is_baseline"] = g, d[g], is_base
            frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    bad = data["error"] != ""
    if bad.any():
        print(f"[warning] {int(bad.sum())} rows with an error were dropped")
        data = data[~bad].reset_index(drop=True)
    return data


def score_response(text, m):
    t = text.strip()
    cat = m["category"]
    o = dict.fromkeys(RESP_COLS, np.nan)
    o["label"], o["json_fmt"] = "", ""
    o["empty_resp"] = float(len(t) == 0)
    o["truncated"] = float(len(t) > 0 and not END_OK.search(t))   # hit the 200-token cap mid-sentence
    o["foreign"] = float(bool(FOREIGN.search(t)))                  # non-Latin script drift
    o["n_words"] = len(t.split())
    o["distinct2"] = distinct_n(t)

    exp = m.get("expected_answer")
    if isinstance(exp, str) and exp.strip():
        o["acc"] = float(set(toks(exp)) <= set(toks(t)))

    if cat == "hallucination":
        lab = label_hallucination(t)
        o["label"] = lab
        o["hall"], o["abstain"], o["refusal"] = (float(lab == k) for k in ("hallucinated", "abstain", "refusal"))

    if cat == "json":
        fmt, obj = parse_json_response(t)
        o["json_fmt"] = fmt
        for k in ("pure", "fenced", "prose", "invalid"):
            o[f"fmt_{k}"] = float(fmt == k)
        o["json_parsable"] = float(fmt != "invalid")
        req = m.get("required_keys") or []
        o["schema_exact"] = float(isinstance(obj, dict) and all(k in obj for k in req))
        o["schema_lenient"] = float(obj is not None and set(k.lower() for k in req) <= all_keys(obj))

    k = expected_sentences(m["prompt"])
    if k:
        o["sent_ok"] = float(n_sentences(t) == k)
    return o


def score_all(data, meta):
    sc = pd.DataFrame([score_response(r.response, meta[int(r.prompt_id)]) for r in data.itertuples(index=False)])
    out = pd.concat([data.reset_index(drop=True), sc], axis=1)
    out["prompt"] = out["prompt_id"].map(lambda i: meta[int(i)]["prompt"])
    return out


def per_prompt(resp):
    """One row per (model, group, setting, prompt) -> determinism over the 5 repeats + mean of response scores."""
    rows = []
    for (model, group, x, pid), g in resp.groupby(["model", "group", "x", "prompt_id"], sort=True):
        g = g.sort_values("repeat")
        tk = [toks(t) for t in g["response"]]
        pairs = list(combinations(range(len(tk)), 2))
        norm = [" ".join(t) for t in tk]
        r = {
            "model": model, "group": group, "x": x, "prompt_id": pid, "category": g["category"].iloc[0],
            "sim": float(np.mean([pair_sim(tk[i], tk[j]) for i, j in pairs])) if pairs else 1.0,
            "unique_ratio": len(set(norm)) / len(norm),
            "exact_match": float(np.mean([norm[i] == norm[j] for i, j in pairs])) if pairs else 1.0,
            "latency_median": float(g["response_time_sec"].median()),
        }
        r.update(g[RESP_COLS].astype(float).mean().to_dict())
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================== AGGREGATION / TABLES
def setting_label(group, x):
    return f"{x:g}" + (" (base)" if abs(float(x) - BASELINE[group]) < 1e-9 else "")


def by_setting(pp, col, cats=None):
    d = pp if cats is None else pp[pp["category"].isin(cats)]
    return d.groupby(["model", "group", "x"])[col].mean()


def table_frame(pp, spec):
    out = None
    for name, col, cats in spec:
        s = by_setting(pp, col, cats).rename(name)
        out = s.to_frame() if out is None else out.join(s, how="outer")
    out = out.reset_index()
    out["_m"] = out["model"].map(MODELS.index)
    out["_g"] = out["group"].map(GROUPS.index)
    out = out.sort_values(["_m", "_g", "x"]).reset_index(drop=True)
    out.insert(0, "Model", out["model"].map(LABEL))
    out.insert(1, "Parameter", out["group"].map(GROUP_NAME))
    out.insert(2, "Setting", [setting_label(g, x) for g, x in zip(out["group"], out["x"])])
    return out.drop(columns=["model", "group", "x", "_m", "_g"]).round(3)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def save_table(df, name, title):
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)
    (TAB_DIR / f"{name}.md").write_text(f"### {title}\n\n{md_table(df)}\n", encoding="utf-8")
    try:
        (TAB_DIR / f"{name}.tex").write_text(df.to_latex(index=False, na_rep="-", float_format="%.3f"), encoding="utf-8")
    except Exception:
        pass


# ============================================================== PLOTTING
def contrast_text(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 165 else "white"


def style_x(ax, group):
    xs = XS[group]
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:g}" + ("*" if abs(x - BASELINE[group]) < 1e-9 else "") for x in xs])
    ax.set_xlabel(GROUP_NAME[group], color=INK2)
    ax.grid(axis="x", visible=False)


def line_panel(ax, pp, group, col, cats, title, ylabel, ylim=(0, 1.05)):
    xs = XS[group]
    d = pp[pp["group"] == group]
    if cats:
        d = d[d["category"].isin(cats)]
    b = xs.index(BASELINE[group])
    ax.axvspan(b - 0.4, b + 0.4, color="#f1f0ee", zorder=0, lw=0)
    for k, m in enumerate(MODELS):
        dm = d[d["model"] == m]
        g = dm.groupby("x")[col].agg(["mean", "std", "count"]).reindex(xs)
        n = int(g["count"].max()) if g["count"].notna().any() else 0
        sem = (g["std"] / np.sqrt(g["count"])).fillna(0).values if n >= 6 else None
        ax.errorbar(range(len(xs)), g["mean"].values, yerr=sem, color=COL[m], marker="o", ms=6.5, lw=2,
                    capsize=3, mec="white", mew=1.2, label=LABEL[m], zorder=3)
        if n < 6:  # few prompts -> show each prompt as a faint dot instead of a (meaningless) error bar
            off = (k - (len(MODELS) - 1) / 2) * 0.12
            for i, x in enumerate(xs):
                v = dm[dm["x"] == x][col].dropna().values
                ax.scatter(np.full(len(v), i + off), v, s=16, color=COL[m], alpha=0.35, zorder=2, lw=0)
    n_prompts = pp[pp["category"].isin(cats)]["prompt_id"].nunique() if cats else pp["prompt_id"].nunique()
    ax.set_title(f"{title} ({n_prompts} prompts)", loc="left")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    style_x(ax, group)


def add_legend(fig, y=1.0):
    handles = [plt.Line2D([0], [0], color=COL[m], marker="o", lw=2, mec="white", label=LABEL[m]) for m in MODELS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#f1f0ee", label="* baseline setting"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, y))


PANELS = [
    ("sim", None, "Output consistency", "mean pairwise similarity"),
    ("sim", ["determinism"], "Consistency, creative prompts", "mean pairwise similarity"),
    ("unique_ratio", None, "Distinct outputs", "share of 5 runs that are distinct"),
    ("acc", ["factual"], "Factual accuracy", "accuracy"),
    ("hall", ["hallucination"], "Hallucination rate", "share of responses"),
    ("json_parsable", ["json"], "JSON parsable", "share of responses"),
    ("fmt_pure", ["json"], "JSON-only compliance (no prose/fence)", "share of responses"),
    ("sent_ok", ["determinism"], "Sentence-count instruction followed", "share of responses"),
    ("foreign", None, "Language drift (non-Latin script)", "share of responses"),
]


def plot_group(pp, group):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5), constrained_layout=True)
    for ax, (col, cats, title, yl) in zip(axes.ravel(), PANELS):
        line_panel(ax, pp, group, col, cats, title, yl)
    add_legend(fig, 1.045)
    fig.suptitle(f"Effect of {GROUP_NAME[group]} on determinism, hallucination and structure", y=1.1, fontsize=13,
                 fontweight="bold")
    fig.savefig(FIG_DIR / f"effect_{group}.png", bbox_inches="tight")
    plt.close(fig)


def plot_overview(pp):
    rows = [("sim", None, "Output consistency", "mean pairwise similarity"),
            ("hall", ["hallucination"], "Hallucination rate", "share of responses"),
            ("fmt_pure", ["json"], "JSON-only compliance", "share of responses")]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), constrained_layout=True)
    for i, (col, cats, title, yl) in enumerate(rows):
        for j, g in enumerate(GROUPS):
            line_panel(axes[i, j], pp, g, col, cats, title, yl)
    add_legend(fig, 1.05)
    fig.suptitle("Overview: sampling parameters vs determinism, hallucination and structure", y=1.11, fontsize=13,
                 fontweight="bold")
    fig.savefig(FIG_DIR / "overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(pp, meta, group):
    xs = XS[group]
    pids = sorted(pp["prompt_id"].unique())
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.2 * len(MODELS) + 5.5, 0.46 * len(pids) + 2.4),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, m in zip(axes, MODELS):
        d = pp[(pp["group"] == group) & (pp["model"] == m)]
        M = d.pivot_table(index="prompt_id", columns="x", values="sim").reindex(index=pids, columns=xs)
        im = ax.imshow(M.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{x:g}" + ("*" if abs(x - BASELINE[group]) < 1e-9 else "") for x in xs])
        ax.set_xlabel(GROUP_NAME[group])
        ax.set_title(LABEL[m], loc="left")
        ax.grid(False)
    axes[0].set_yticks(range(len(pids)))
    axes[0].set_yticklabels([f"P{p} {meta[p]['category'][:5]}: {meta[p]['prompt'][:30]}..." for p in pids], fontsize=8)
    cb = fig.colorbar(im, ax=list(axes), shrink=0.8)
    cb.set_label("mean pairwise similarity across 5 runs (1 = identical)")
    fig.suptitle(f"Per-prompt output consistency vs {GROUP_NAME[group]}   (* = baseline)", fontweight="bold")
    fig.savefig(FIG_DIR / f"heatmap_consistency_{group}.png", bbox_inches="tight")
    plt.close(fig)


def stacked_grid(resp, col, order, colors, names, cat, fname, title):
    d = resp[resp["category"] == cat]
    order = [k for k in order if (d[col] == k).any()]
    fig, axes = plt.subplots(len(MODELS), len(GROUPS), figsize=(4.8 * len(GROUPS), 3.5 * len(MODELS)),
                             sharey=True, constrained_layout=True, squeeze=False)
    for i, m in enumerate(MODELS):
        for j, g in enumerate(GROUPS):
            ax, xs = axes[i][j], XS[g]
            sub = d[(d["model"] == m) & (d["group"] == g)]
            frac = pd.crosstab(sub["x"], sub[col], normalize="index").reindex(index=xs, columns=order, fill_value=0)
            bottom = np.zeros(len(xs))
            for k in order:
                v = frac[k].values
                ax.bar(range(len(xs)), v, bottom=bottom, color=colors[k], edgecolor="white", linewidth=1.6,
                       width=0.62, label=names[k], zorder=3)
                for idx, val in enumerate(v):
                    if val >= 0.09:
                        ax.text(idx, bottom[idx] + val / 2, f"{val:.0%}", ha="center", va="center", fontsize=8,
                                color=contrast_text(colors[k]), zorder=4)
                bottom += v
            ax.set_ylim(0, 1.0)
            style_x(ax, g)
            if i == 0:
                ax.set_title(GROUP_NAME[g], loc="left")
            if j == 0:
                ax.set_ylabel(f"{LABEL[m]}\nshare of responses")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k], label=names[k]) for k in order]
    fig.legend(handles=handles, loc="upper center", ncol=len(order), frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.suptitle(title + "   (* = baseline)", y=1.13, fontsize=12.5, fontweight="bold")
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_comparison(pp):
    mets = [("sim", None, "Consistency"), ("acc", ["factual"], "Factual\naccuracy"),
            ("hall", ["hallucination"], "Hallucination\nrate"), ("fmt_pure", ["json"], "JSON-only"),
            ("schema_lenient", ["json"], "JSON keys\npresent"), ("sent_ok", ["determinism"], "Sentence-count\nfollowed"),
            ("foreign", None, "Language\ndrift")]
    d = pp[(pp["group"] == "temperature") & (pp["x"] == BASELINE["temperature"])]
    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    w = 0.8 / len(MODELS)
    for k, m in enumerate(MODELS):
        vals = []
        for col, cats, _ in mets:
            dd = d[d["model"] == m]
            dd = dd[dd["category"].isin(cats)] if cats else dd
            vals.append(dd[col].mean())
        pos = np.arange(len(mets)) + (k - (len(MODELS) - 1) / 2) * w
        ax.bar(pos, vals, width=w - 0.04, color=COL[m], label=LABEL[m], zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels([m[2] for m in mets])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score (0-1)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", ncol=len(MODELS))
    ax.set_title("Model comparison at the baseline setting (T=0.7, top_k=40, top_p=0.9)", loc="left")
    fig.savefig(FIG_DIR / "model_comparison_baseline.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================== FINDINGS
def write_findings(t_det, t_hall, t_str):
    lines = ["# Auto-generated key numbers (check them against the figures before writing the report)\n"]
    for model in t_det["Model"].unique():
        lines.append(f"## {model}")
        for g in GROUP_NAME.values():
            a = t_det[(t_det.Model == model) & (t_det.Parameter == g)]
            b = t_hall[(t_hall.Model == model) & (t_hall.Parameter == g)]
            c = t_str[(t_str.Model == model) & (t_str.Parameter == g)]
            hi, lo = a.loc[a["Consistency (all)"].idxmax()], a.loc[a["Consistency (all)"].idxmin()]
            hmin, hmax = b.loc[b["Hallucination rate"].idxmax()], b.loc[b["Hallucination rate"].idxmin()]
            jmax, jmin = c.loc[c["JSON-only"].idxmax()], c.loc[c["JSON-only"].idxmin()]
            lines.append(
                f"- **{g}**: consistency highest at {hi['Setting']} ({hi['Consistency (all)']:.2f}), lowest at "
                f"{lo['Setting']} ({lo['Consistency (all)']:.2f}); hallucination rate highest at {hmin['Setting']} "
                f"({hmin['Hallucination rate']:.2f}), lowest at {hmax['Setting']} ({hmax['Hallucination rate']:.2f}); "
                f"JSON-only best at {jmax['Setting']} ({jmax['JSON-only']:.2f}), worst at {jmin['Setting']} "
                f"({jmin['JSON-only']:.2f}).")
        lines.append("")
    (TAB_DIR / "key_numbers.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================== MAIN
def main():
    global MODELS, LABEL, COL, XS, CAT_N
    for d in (FIG_DIR, TAB_DIR):
        d.mkdir(parents=True, exist_ok=True)

    meta = load_prompts()
    data = load_data()
    MODELS = list(dict.fromkeys(data["model"]))
    pretty = {"llama3.2:1b": "Llama 3.2 (1B)", "qwen2.5:1.5b": "Qwen 2.5 (1.5B)"}
    LABEL = {m: pretty.get(m, m) for m in MODELS}
    COL = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(MODELS)}
    XS = {g: sorted(data.loc[data["group"] == g, "x"].unique().tolist()) for g in GROUPS}
    CAT_N = Counter(p["category"] for p in meta.values())
    print(f"Loaded {len(data)} responses | models: {MODELS} | prompts: {dict(CAT_N)}")

    resp = score_all(data, meta)
    pp = per_prompt(resp)

    # ---------- tables
    t_det = table_frame(pp, [
        ("Consistency (all)", "sim", None), ("Consistency (creative)", "sim", ["determinism"]),
        ("Distinct-output ratio", "unique_ratio", None), ("Exact-match rate", "exact_match", None),
        ("Distinct-2 (repetition)", "distinct2", None)])
    t_hall = table_frame(pp, [
        ("Factual accuracy", "acc", ["factual"]), ("Hallucination rate", "hall", ["hallucination"]),
        ("Honest abstention", "abstain", ["hallucination"]), ("Refusal rate", "refusal", ["hallucination"])])
    t_str = table_frame(pp, [
        ("JSON parsable", "json_parsable", ["json"]), ("JSON-only", "fmt_pure", ["json"]),
        ("JSON fenced", "fmt_fenced", ["json"]), ("JSON + prose", "fmt_prose", ["json"]),
        ("JSON invalid", "fmt_invalid", ["json"]), ("Keys exact", "schema_exact", ["json"]),
        ("Keys (nested ok)", "schema_lenient", ["json"]), ("Sentence-count ok", "sent_ok", ["determinism"]),
        ("Language drift", "foreign", None), ("Cut at token cap", "truncated", None)])
    t_eff = table_frame(pp, [("Median latency (s)", "latency_median", None), ("Avg words", "n_words", None)])

    save_table(t_det, "table1_determinism", "Table 1 - Determinism (similarity across 5 seeds)")
    save_table(t_hall, "table2_hallucination", "Table 2 - Hallucination")
    save_table(t_str, "table3_structure", "Table 3 - Structure / instruction following")
    save_table(t_eff, "table4_latency_length", "Table 4 - Latency and length")
    write_findings(t_det, t_hall, t_str)

    pp.round(4).to_csv(TAB_DIR / "per_prompt_metrics.csv", index=False)
    keep = ["model", "group", "x", "prompt_id", "category", "repeat", "label", "json_fmt", "acc", "schema_exact",
            "schema_lenient", "sent_ok", "truncated", "foreign", "n_words", "response_time_sec", "prompt", "response"]
    resp[keep].to_csv(TAB_DIR / "response_level_scores.csv", index=False)

    # manual-review sample (validate the heuristic hallucination labels by hand)
    hs = resp[resp["category"] == "hallucination"]
    sample = hs.sample(n=min(60, len(hs)), random_state=0)[
        ["model", "group", "x", "prompt_id", "repeat", "prompt", "response", "label"]].rename(columns={"label": "auto_label"})
    sample["manual_label"] = ""   # fill with: hallucinated / abstain / refusal
    sample.to_csv(TAB_DIR / "manual_review_sample.csv", index=False)

    try:
        with pd.ExcelWriter(TAB_DIR / "all_tables.xlsx") as xw:
            for name, df in (("1_Determinism", t_det), ("2_Hallucination", t_hall), ("3_Structure", t_str),
                             ("4_Latency", t_eff)):
                df.to_excel(xw, sheet_name=name, index=False)
    except Exception as e:  # openpyxl missing
        print("[note] Excel file skipped:", e)

    # ---------- figures
    plot_overview(pp)
    for g in GROUPS:
        plot_group(pp, g)
        plot_heatmap(pp, meta, g)
    stacked_grid(resp, "label", ["hallucinated", "abstain", "refusal", "empty"],
                 {"hallucinated": "#e34948", "abstain": "#1baf7a", "refusal": "#8c8b87", "empty": "#d9d8d4"},
                 {"hallucinated": "Hallucinated (confident / invented)", "abstain": "Honest 'unknown / does not exist'",
                  "refusal": "Policy-style refusal", "empty": "Empty"},
                 "hallucination", "hallucination_outcomes.png", "How models answer unanswerable prompts")
    stacked_grid(resp, "json_fmt", ["pure", "fenced", "prose", "invalid"],
                 {"pure": "#2a78d6", "fenced": "#6250d6", "prose": "#eda100", "invalid": "#e34948"},
                 {"pure": "JSON only", "fenced": "JSON in ``` fence", "prose": "JSON + extra text",
                  "invalid": "Not valid JSON"},
                 "json", "json_format_breakdown.png", "Format of answers to 'JSON only' prompts")
    plot_baseline_comparison(pp)

    # ---------- console summary
    pd.set_option("display.width", 220, "display.max_columns", 30)
    for t, name in ((t_det, "DETERMINISM"), (t_hall, "HALLUCINATION"), (t_str, "STRUCTURE")):
        print(f"\n===== {name} =====")
        print(t.to_string(index=False))
    print(f"\nTables  -> {TAB_DIR}\nFigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()