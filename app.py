"""
ProfileIQ - Profile Outcome Predictor
=====================================
A single-file Flask web app. Backend, HTML, CSS and JavaScript all live in this
one app.py (the page is served with render_template_string).

Project folder (all three files in the ROOT of your GitHub repo):
    app.py
    requirements.txt
    svm.pkl

Local run : python app.py
Render    : gunicorn app:app      (Build: pip install -r requirements.txt)

What was found by inspecting svm.pkl (nothing was retrained or changed):
  * Model type      : scikit-learn SVC (RBF kernel, C=1.0), saved with scikit-learn 1.6.1
  * Input features  : Age, Gender, Region, Occupation, Income   (this exact order)
  * Output classes  : 'no' and 'yes'
  * Preprocessing   : none stored inside the pickle - the model takes raw numbers
  * Probabilities   : NOT available (the model was saved with probability=False),
                      so the app shows the model's decision score instead of a
                      made-up confidence percentage.
  * Gender / Region / Occupation are stored as numeric codes. The pickle does not
    contain the names behind those codes, so the dropdowns show the codes until you
    add the real names in CATEGORY_LABELS below.
"""
import logging
import math
import os
import pickle
import warnings

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from werkzeug.exceptions import HTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("profileiq")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm.pkl")

# ============================================================================
# 1. BRANDING  (edit these to match what your yes / no target really means)
# ============================================================================
APP_NAME = "ProfileIQ"
APP_TAGLINE = "Smart profile-based outcome prediction"
PAGE_TITLE = "ProfileIQ | Profile Outcome Predictor"
OUTCOME_CAPTION = "Does this person have a laptop?"
# What the model's two answers mean (raw class -> text shown in the app).
OUTCOME_LABELS = {"yes": "Has a laptop", "no": "No laptop"}
OUTCOME_SENTENCES = {"yes": "This person is predicted to HAVE a laptop.",
                     "no": "This person is predicted NOT to have a laptop."}

# ============================================================================
# 2. CATEGORY NAMES  <-- fill these in with your real names
# ============================================================================
# The model only understands numbers. Put the real name for each code here, the
# same way you encoded the data before training. Example format:
#     "Gender": {0: "Name for code 0", 1: "Name for code 1"}
# Only options that have a name here appear in the dropdown. (A field left as {} shows "Code 0", "Code 1", ...)
#
# To print your real mapping, run this once in your notebook
# (df = your ORIGINAL dataframe before encoding; assumes LabelEncoder's
# alphabetical numbering):
#     for c in ["Gender", "Region", "Occupation"]:
#         print(c, dict(enumerate(sorted(df[c].dropna().astype(str).unique()))))
CATEGORY_LABELS = {
    "Gender":     {0: "Female", 1: "Male"},
    "Region":     {0: "City", 1: "Countryside"},
    # The model also knows an occupation with code 1 (alphabetically between Banker and
    # Student) whose name is not known yet. Add it here as   1: "Name",   and it will appear.
    "Occupation": {0: "Banker", 2: "Student", 3: "Teacher"},
}
CATEGORICAL_COLUMNS = ["Gender", "Region", "Occupation"]

# Optional: force the list of valid codes. Leave as None to read them from the
# model itself (the values that appear in its stored support vectors).
CATEGORY_CODES = {"Gender": None, "Region": None, "Occupation": None}

# ============================================================================
# 3. LOAD THE MODEL (the site still opens if something is wrong with the file)
# ============================================================================
FALLBACK_FEATURES = ["Age", "Gender", "Region", "Occupation", "Income"]

model, MODEL_ERROR = None, None
try:
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception:
        import joblib  # used only if the file was saved with joblib
        model = joblib.load(MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    MODEL_ERROR = f"{type(exc).__name__}: {exc}"
    log.error("Could not load svm.pkl -> %s", MODEL_ERROR)

FEATURES = list(getattr(model, "feature_names_in_", FALLBACK_FEATURES))   # exact training order
CLASSES = [str(c) for c in getattr(model, "classes_", ["no", "yes"])]
HAS_PROBA = model is not None and hasattr(model, "predict_proba")          # False for this model
HAS_SCORE = model is not None and hasattr(model, "decision_function")


# ============================================================================
# 4. FORM FIELDS
# ============================================================================
def category_codes(name):
    """Valid numeric codes for a categorical column."""
    if CATEGORY_CODES.get(name):
        return sorted(int(c) for c in CATEGORY_CODES[name])
    sv = getattr(model, "support_vectors_", None)
    if sv is not None and name in FEATURES:
        col = sv[:, FEATURES.index(name)]
        return sorted(int(c) for c in np.unique(col))
    return [0, 1]


FIELD_META = {
    "Age":        dict(label="Age",        hint="Age of the person, in years",                  icon="user",      group="Personal details",  min=1, max=120, step=1, integer=True, placeholder="Enter age"),
    "Gender":     dict(label="Gender",     hint="Select the gender category",                   icon="users",     group="Personal details"),
    "Region":     dict(label="Region",     hint="Select the region category",                   icon="pin",       group="Location & work"),
    "Occupation": dict(label="Occupation", hint="Select the occupation category",               icon="briefcase", group="Location & work"),
    "Income":     dict(label="Income",     hint="Income value, in the same units as your data", icon="cash",      group="Financial",         min=0, max=1_000_000_000, step=1, placeholder="Enter income", wide=True),
}


def build_fields():
    fields = []
    for name in FEATURES:
        meta = dict(FIELD_META.get(name, {}))
        meta.setdefault("label", name.replace("_", " "))
        meta.setdefault("hint", name)
        meta.setdefault("icon", "dot")
        meta.setdefault("group", "Other details")
        meta["name"] = name
        if name in CATEGORICAL_COLUMNS:
            labels = CATEGORY_LABELS.get(name, {})
            codes = category_codes(name)
            if labels:                                   # show only the named options
                hidden = [c for c in codes if c not in labels]
                if hidden:
                    log.warning("%s: no name given for code(s) %s, so they are hidden in the dropdown.", name, hidden)
                codes = [c for c in codes if c in labels]
            meta["kind"] = "category"
            meta["options"] = [
                {"code": c, "label": str(labels.get(c, f"Code {c}")), "named": c in labels}
                for c in codes
            ]
        else:
            meta["kind"] = "number"
            meta.setdefault("min", -1e12)
            meta.setdefault("max", 1e12)
            meta.setdefault("step", "any")
            meta.setdefault("placeholder", "Enter value")
        fields.append(meta)
    return fields


FIELDS = build_fields()
MISSING_NAMES = [f["label"] for f in FIELDS
                 if f["kind"] == "category" and any(not o["named"] for o in f["options"])]
LABELS_MISSING = bool(MISSING_NAMES)

# ============================================================================
# 5. ROUTES
# ============================================================================
app = Flask(__name__)


@app.get("/")
def index():
    cfg = {
        "appName": APP_NAME,
        "caption": OUTCOME_CAPTION,
        "outcomeLabels": {c: OUTCOME_LABELS.get(c, c.capitalize()) for c in CLASSES},
        "outcomeSentences": {c: OUTCOME_SENTENCES.get(c, f"Predicted: {c}") for c in CLASSES},
        "fields": FIELDS,
        "classes": CLASSES,
        "hasProba": HAS_PROBA,
        "hasScore": HAS_SCORE,
        "labelsMissing": MISSING_NAMES,
        "modelOk": model is not None,
    }
    return render_template_string(PAGE, cfg=cfg, page_title=PAGE_TITLE,
                                  app_name=APP_NAME, tagline=APP_TAGLINE)


@app.get("/health")
def health():
    return jsonify(status="ok" if model is not None else "model_not_loaded",
                   model_error=MODEL_ERROR, category_names_set=not LABELS_MISSING)


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


def validate(payload):
    """Check each input and convert it to the number the model expects."""
    row, errors = {}, {}
    for f in FIELDS:
        name, raw = f["name"], payload.get(f["name"])
        if raw is None or str(raw).strip() == "":
            errors[name] = "This field is required."
            continue

        if f["kind"] == "category":
            allowed = {str(o["code"]): o["code"] for o in f["options"]}
            if str(raw).strip() in allowed:
                row[name] = float(allowed[str(raw).strip()])
            else:
                errors[name] = "Please choose one of the listed options."
            continue

        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Enter a valid number."
            continue
        if math.isnan(value) or math.isinf(value):
            errors[name] = "Enter a valid number."
        elif not (f["min"] <= value <= f["max"]):
            errors[name] = f"Enter a value between {f['min']:,.0f} and {f['max']:,.0f}."
        elif f.get("integer") and value != int(value):
            errors[name] = "Enter a whole number."
        else:
            row[name] = value
    return row, errors


@app.post("/predict")
def predict():
    if model is None:
        return jsonify(error="The model file could not be loaded on the server. "
                             "Make sure svm.pkl is in the repo root."), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send the form values as JSON."), 400

    row, errors = validate(payload)
    if errors:
        return jsonify(error="Please fix the highlighted fields.", fields=errors), 400

    X = np.array([[row[n] for n in FEATURES]], dtype=float)      # exact training order
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                           # silence feature-name notice
        label = str(model.predict(X)[0])

        score = None
        if HAS_SCORE:
            flat = np.ravel(model.decision_function(X))
            if flat.size == 1:                                     # binary model: one signed score
                score = round(float(flat[0]), 5)

        probabilities, confidence = None, None
        if HAS_PROBA:                                              # only if the model supports it
            p = model.predict_proba(X)[0]
            probabilities = {c: round(float(v) * 100, 2) for c, v in zip(CLASSES, p)}
            confidence = round(float(np.max(p)) * 100, 2)

    return jsonify(prediction=label, score=score, confidence=confidence, probabilities=probabilities)


@app.errorhandler(Exception)
def on_error(exc):
    """Always reply with clean JSON instead of an ugly HTML error page."""
    if isinstance(exc, HTTPException):
        return jsonify(error=exc.description), exc.code
    log.exception("Unhandled error")
    return jsonify(error="Unexpected server error. Please try again."), 500


# ============================================================================
# 6. THE WHOLE PAGE (HTML + CSS + JavaScript) - served by render_template_string
#    {% raw %} blocks stop Jinja from touching the CSS/JS braces.
# ============================================================================
PAGE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#101827">
<title>{{ page_title }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Sora:wght@500;600;700&display=swap" rel="stylesheet">
<script>
  document.documentElement.classList.add("js");
  try { document.documentElement.dataset.theme = localStorage.getItem("theme") || "dark"; } catch (e) {}
</script>
<style>
{% raw %}
/* ============================ COLOUR SYSTEM ============================ */
:root{
  --primary:#98E8DE;      /* Mint Turquoise */
  --navy:#101827;         /* Deep Navy */
  --teal:#163A40;         /* Dark Teal */
  --aqua:#B8FFF3;         /* Soft Aqua */
  --mint:#6FCFC1;         /* Muted Mint */
  --white:#F5FAFA;
  --grey:#AAB8C2;
  --radius:26px; --ease:cubic-bezier(.2,.8,.2,1);
}
[data-theme="dark"]{
  --bg1:#101827; --bg2:#163A40; --bg3:#0d2530;
  --text:#F5FAFA; --muted:#AAB8C2; --ink:#98E8DE;
  --glass:rgba(22,58,64,.42); --glass-hi:rgba(152,232,222,.10);
  --border:rgba(152,232,222,.22); --input:rgba(16,24,39,.68);
  --shadow:0 24px 70px rgba(0,0,0,.5),0 0 44px rgba(152,232,222,.07);
  --on-primary:#101827; --shape:rgba(152,232,222,.28); --aur:.55;
}
[data-theme="light"]{
  --bg1:#effaf9; --bg2:#d5f3ef; --bg3:#e6f8f5;
  --text:#101827; --muted:#4b5f6b; --ink:#0d6f65;
  --glass:rgba(255,255,255,.66); --glass-hi:rgba(152,232,222,.22);
  --border:rgba(22,58,64,.16); --input:rgba(255,255,255,.92);
  --shadow:0 20px 50px rgba(22,58,64,.14),0 0 40px rgba(111,207,193,.18);
  --on-primary:#101827; --shape:rgba(13,111,101,.22); --aur:.45;
}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
[hidden]{display:none!important}
html{scroll-behavior:smooth;scroll-padding-top:96px}
body{
  font-family:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text);min-height:100vh;min-height:100dvh;overflow-x:hidden;
  background:linear-gradient(125deg,var(--bg1),var(--bg2),var(--bg3),var(--bg2),var(--bg1));background-size:420% 420%;animation:bgFlow 36s ease-in-out infinite;
}
@keyframes bgFlow{0%,100%{background-position:0 50%}50%{background-position:100% 50%}}
svg.ic{width:1em;height:1em;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;flex:none}

/* ============================ BACKGROUND ============================ */
.bg{position:fixed;inset:0;overflow:hidden;pointer-events:none;z-index:0}
.aurora{position:absolute;width:70vmax;height:34vmax;left:-10vmax;border-radius:50%;filter:blur(70px);opacity:var(--aur);mix-blend-mode:screen;will-change:transform}
[data-theme="light"] .aurora{mix-blend-mode:multiply}
.au1{top:-12vmax;background:linear-gradient(90deg,rgba(152,232,222,.55),rgba(111,207,193,.0) 70%);animation:auroraA 24s ease-in-out infinite}
.au2{top:34vh;left:auto;right:-20vmax;background:linear-gradient(90deg,rgba(22,58,64,.0),rgba(184,255,243,.38));animation:auroraB 30s ease-in-out infinite}
.au3{bottom:-14vmax;left:10vmax;background:linear-gradient(90deg,rgba(111,207,193,.45),rgba(22,58,64,.0) 75%);animation:auroraA 34s ease-in-out infinite reverse}
@keyframes auroraA{0%,100%{transform:translate3d(0,0,0) rotate(-6deg) scaleY(1)}50%{transform:translate3d(14vmax,5vmax,0) rotate(8deg) scaleY(1.35)}}
@keyframes auroraB{0%,100%{transform:translate3d(0,0,0) rotate(10deg)}50%{transform:translate3d(-16vmax,-6vmax,0) rotate(-6deg) scaleY(1.3)}}
.grid-lines{position:absolute;inset:0;opacity:.45;
  background-image:linear-gradient(var(--border) 1px,transparent 1px),linear-gradient(90deg,var(--border) 1px,transparent 1px);background-size:70px 70px;
  -webkit-mask-image:radial-gradient(ellipse at 50% 25%,#000 0,transparent 68%);mask-image:radial-gradient(ellipse at 50% 25%,#000 0,transparent 68%);animation:gridMove 20s linear infinite}
@keyframes gridMove{to{background-position:70px 70px}}
.shape{position:absolute;left:var(--x);top:var(--y);width:var(--s);height:var(--s);
  transform:translate(calc(var(--mx,0)*var(--d)*1px),calc(var(--my,0)*var(--d)*1px));transition:transform .7s ease-out}
.shape i{position:absolute;inset:0;display:block;animation:floaty var(--t) ease-in-out infinite;animation-delay:var(--dl,0s)}
.shape.ring i{border:2px solid var(--shape);border-radius:50%;box-shadow:0 0 26px var(--shape),inset 0 0 26px var(--shape)}
.shape.ring2 i{border:2px dashed var(--shape);border-radius:50%;animation:floaty var(--t) ease-in-out infinite,spin 40s linear infinite}
.shape.hex i{background:linear-gradient(135deg,var(--shape),transparent);clip-path:polygon(25% 5%,75% 5%,100% 50%,75% 95%,25% 95%,0 50%)}
.shape.tri i{background:linear-gradient(160deg,var(--shape),transparent);clip-path:polygon(50% 0,100% 100%,0 100%)}
.shape.dia i{border:2px solid var(--shape);transform:rotate(45deg);animation:floatyR var(--t) ease-in-out infinite;border-radius:8px}
.shape.orb i{border-radius:50%;background:radial-gradient(circle at 30% 30%,var(--aqua),var(--mint) 55%,transparent 72%);opacity:.38;filter:blur(1px)}
@keyframes floaty{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-34px) rotate(14deg)}}
@keyframes floatyR{0%,100%{transform:rotate(45deg) translateY(0)}50%{transform:rotate(95deg) translateY(-30px)}}
@keyframes spin{to{transform:rotate(360deg)}}
#particles,#burst{position:fixed;inset:0;width:100%;height:100%;pointer-events:none}
#particles{z-index:0}#burst{z-index:80}

/* ============================ HEADER / NAV ============================ */
.wrap{position:relative;z-index:1;max-width:1240px;margin:0 auto;padding:0 clamp(14px,3vw,28px) max(48px,env(safe-area-inset-bottom))}
.header{position:sticky;top:12px;z-index:40;margin-top:14px;display:flex;align-items:center;gap:16px;padding:10px 12px 10px 16px;border-radius:22px;
  background:var(--glass);border:1px solid var(--border);backdrop-filter:blur(22px) saturate(160%);-webkit-backdrop-filter:blur(22px) saturate(160%);
  box-shadow:var(--shadow);animation:dropIn .8s var(--ease) both}
@keyframes dropIn{from{opacity:0;transform:translateY(-24px)}to{opacity:1;transform:none}}
.brand{display:flex;align-items:center;gap:12px;text-decoration:none;color:inherit;min-width:0}
.logo{width:44px;height:44px;flex:none;filter:drop-shadow(0 0 12px rgba(152,232,222,.6));animation:logoFloat 5s ease-in-out infinite}
@keyframes logoFloat{50%{transform:translateY(-3px) rotate(-4deg)}}
.brand strong{font-family:"Sora",sans-serif;font-size:1.1rem;display:block;letter-spacing:-.01em}
.brand small{display:block;color:var(--muted);font-size:.72rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nav{display:flex;gap:4px;margin-left:auto;padding:4px;border-radius:16px;background:var(--glass-hi);border:1px solid var(--border)}
.nav a{position:relative;display:flex;align-items:center;gap:8px;padding:9px 16px;border-radius:12px;color:var(--muted);text-decoration:none;font-size:.86rem;font-weight:500;transition:color .3s,background .3s,transform .3s var(--ease)}
.nav a:hover{color:var(--text);transform:translateY(-2px);background:var(--glass-hi)}
.nav a.active{color:var(--on-primary);background:linear-gradient(135deg,var(--primary),var(--mint));box-shadow:0 6px 20px rgba(152,232,222,.4)}
.icon-btn{width:44px;height:44px;border-radius:14px;border:1px solid var(--border);background:var(--glass-hi);color:var(--ink);cursor:pointer;font-size:1.1rem;display:grid;place-items:center;transition:transform .5s var(--ease),box-shadow .3s}
.icon-btn:hover{transform:rotate(30deg) scale(1.06);box-shadow:0 0 22px rgba(152,232,222,.45)}
[data-theme=dark] .sun,[data-theme=light] .moon{display:none}

/* ============================ HERO ============================ */
.hero{text-align:center;max-width:800px;margin:clamp(34px,7vw,76px) auto clamp(30px,5vw,54px)}
.pill{display:inline-flex;align-items:center;gap:9px;padding:8px 18px;border-radius:999px;font-size:.78rem;font-weight:500;color:var(--ink);background:var(--glass);border:1px solid var(--border);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)}
.pill .dot{width:8px;height:8px;border-radius:50%;background:var(--primary);box-shadow:0 0 12px var(--primary);animation:pulse 2s infinite}
@keyframes pulse{50%{transform:scale(1.8);opacity:.45}}
.hero h1{font-family:"Sora",sans-serif;font-size:clamp(2.2rem,6.6vw,4rem);line-height:1.06;letter-spacing:-.03em;margin:20px 0 16px}
.grad{background:linear-gradient(100deg,#98E8DE,#B8FFF3,#6FCFC1,#98E8DE);background-size:300% auto;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 7s linear infinite}
[data-theme=light] .grad{background-image:linear-gradient(100deg,#0d6f65,#13a597,#0d6f65);}
@keyframes shine{to{background-position:300% center}}
.hero p{color:var(--muted);font-size:clamp(.96rem,2.2vw,1.12rem);line-height:1.7;font-weight:300;max-width:640px;margin:0 auto}
.chips{display:flex;justify-content:center;flex-wrap:wrap;gap:10px;margin-top:26px}
.chip{display:inline-flex;align-items:center;gap:8px;padding:9px 16px;border-radius:14px;font-size:.82rem;color:var(--text);background:var(--glass);border:1px solid var(--border);transition:transform .3s var(--ease),box-shadow .3s}
.chip svg{color:var(--ink)}
.chip:hover{transform:translateY(-3px);box-shadow:0 10px 28px rgba(152,232,222,.28)}

/* ============================ LAYOUT / CARDS ============================ */
.main-grid{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:clamp(16px,2.4vw,28px);align-items:start}
.side{position:sticky;top:104px}
.section-head{display:flex;align-items:center;gap:14px;margin:clamp(40px,6vw,64px) 0 22px}
.section-head h2{font-family:"Sora",sans-serif;font-size:clamp(1.3rem,3vw,1.7rem);letter-spacing:-.02em}
.section-head::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}
.section-head svg{color:var(--ink);font-size:1.4rem}
.card{position:relative;border-radius:var(--radius);padding:clamp(18px,2.6vw,30px);overflow:hidden;
  background:linear-gradient(150deg,var(--glass-hi),var(--glass) 55%);border:1px solid var(--border);
  backdrop-filter:blur(24px) saturate(150%);-webkit-backdrop-filter:blur(24px) saturate(150%);box-shadow:var(--shadow),inset 0 1px 0 rgba(255,255,255,.12);transition:transform .3s ease-out,box-shadow .4s}
.card::before{content:"";position:absolute;inset:0;pointer-events:none;border-radius:inherit;opacity:0;transition:opacity .4s;
  background:radial-gradient(420px circle at var(--cx,50%) var(--cy,50%),rgba(152,232,222,.16),transparent 55%)}
.card:hover::before{opacity:1}
.card:hover{box-shadow:var(--shadow),0 0 50px rgba(152,232,222,.14)}
.card-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:20px}
.card-head h3{font-family:"Sora",sans-serif;font-size:1.08rem;font-weight:600;display:flex;align-items:center;gap:10px}
.card-head h3 svg{color:var(--ink);font-size:1.15rem}
.muted{color:var(--muted);font-size:.82rem}
/* entrance animation */
.js .reveal{opacity:0;transform:translateY(34px) scale(.98);transition:opacity .9s var(--ease),transform .9s var(--ease);transition-delay:var(--rd,0s)}
.js .reveal.in{opacity:1;transform:none}

/* ============================ FORM ============================ */
.notice{display:flex;gap:12px;align-items:flex-start;padding:13px 15px;border-radius:16px;margin-bottom:20px;font-size:.82rem;line-height:1.55;background:rgba(152,232,222,.09);border:1px dashed var(--border)}
.notice svg{color:var(--ink);margin-top:3px;font-size:1rem}
.notice code{background:var(--glass-hi);padding:1px 6px;border-radius:6px;font-size:.78rem}
.notice button{margin-left:auto;background:none;border:0;color:var(--muted);cursor:pointer;font-size:1.1rem;line-height:1}
.progress{margin:-4px 0 22px}
.progress-top{display:flex;justify-content:space-between;font-size:.78rem;color:var(--muted);margin-bottom:8px}
.progress-top b{color:var(--ink)}
.progress-bar{height:7px;border-radius:99px;background:var(--border);overflow:hidden}
.progress-bar i{display:block;height:100%;width:0;border-radius:inherit;transition:width .7s var(--ease);background:linear-gradient(90deg,var(--mint),var(--primary),var(--aqua),var(--mint));background-size:200% 100%;animation:flow 3s linear infinite;box-shadow:0 0 14px rgba(152,232,222,.7)}
@keyframes flow{to{background-position:200% 0}}
.section{margin-bottom:24px}
.section-title{display:flex;align-items:center;gap:10px;font-family:"Sora",sans-serif;font-size:.76rem;letter-spacing:.16em;text-transform:uppercase;color:var(--ink);margin-bottom:14px}
.section-title::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}
.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 16px}
.field.wide{grid-column:1/-1}
.field{position:relative}
.field label{display:flex;align-items:center;gap:8px;font-size:.86rem;font-weight:600;margin-bottom:8px;transition:color .3s}
.field label svg{color:var(--ink);font-size:.95rem}
.field:focus-within label{color:var(--ink)}
.control{position:relative}
.control input,.control select{width:100%;padding:14px 16px;border-radius:15px;font:inherit;font-size:.95rem;color:var(--text);background:var(--input);border:1px solid var(--border);outline:none;appearance:none;-webkit-appearance:none;transition:border-color .25s,box-shadow .3s,transform .3s var(--ease)}
.control input::placeholder{color:var(--muted);opacity:.75}
.control input::-webkit-outer-spin-button,.control input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
.control input[type=number]{-moz-appearance:textfield}
.control input:hover,.control select:hover{border-color:rgba(152,232,222,.55)}
.control input:focus,.control select:focus{border-color:var(--primary);box-shadow:0 0 0 4px rgba(152,232,222,.2),0 10px 28px rgba(152,232,222,.2);transform:translateY(-2px)}
.control select:invalid{color:var(--muted)}
.control select option{background:#163A40;color:#F5FAFA}
.chev{position:absolute;right:16px;top:50%;transform:translateY(-50%);pointer-events:none;color:var(--muted);font-size:.9rem}
.underline{position:absolute;left:14px;right:14px;bottom:0;height:2px;border-radius:2px;background:linear-gradient(90deg,var(--mint),var(--aqua));transform:scaleX(0);transition:transform .45s var(--ease);box-shadow:0 0 10px var(--primary)}
.control:focus-within .underline{transform:scaleX(1)}
.hint{display:block;margin-top:7px;color:var(--muted);font-size:.76rem;line-height:1.4}
.error-text{display:none;margin-top:6px;color:#ff8a9b;font-size:.78rem;font-weight:500}
[data-theme=light] .error-text{color:#c4264a}
.field.invalid .control input,.field.invalid .control select{border-color:#ff8a9b;box-shadow:0 0 0 4px rgba(255,138,155,.15);animation:shake .4s}
.field.invalid .error-text{display:block}
@keyframes shake{20%,60%{transform:translateX(-5px)}40%,80%{transform:translateX(5px)}}
.field.filled label::after{content:"";margin-left:auto;width:18px;height:18px;border-radius:50%;background:var(--primary) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23101827' stroke-width='3.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m5 12 5 5L20 7'/%3E%3C/svg%3E") center/11px no-repeat;animation:tick .45s cubic-bezier(.3,1.6,.5,1)}
@keyframes tick{from{transform:scale(0) rotate(-90deg)}to{transform:none}}

/* ============================ BUTTONS ============================ */
.actions{display:flex;gap:12px;margin-top:8px}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:10px;border:0;cursor:pointer;font:600 .96rem "Inter",sans-serif;padding:16px 22px;border-radius:17px;position:relative;overflow:hidden;color:var(--text);transition:transform .3s var(--ease),box-shadow .3s,background .3s}
.btn svg{font-size:1.05rem}
.btn.primary{flex:1;color:var(--on-primary);background:linear-gradient(110deg,var(--primary),var(--aqua),var(--mint),var(--primary));background-size:300% 100%;animation:flow 7s linear infinite;box-shadow:0 12px 32px rgba(152,232,222,.4),0 0 0 1px rgba(184,255,243,.4) inset}
.btn.primary::after{content:"";position:absolute;top:0;left:-60%;width:40%;height:100%;transform:skewX(-20deg);background:linear-gradient(100deg,transparent,rgba(255,255,255,.65),transparent);animation:sweep 4s ease-in-out infinite}
@keyframes sweep{0%{left:-60%}55%,100%{left:130%}}
.btn.primary:hover{transform:translateY(-3px) scale(1.01);box-shadow:0 20px 46px rgba(152,232,222,.55)}
.btn:active{transform:scale(.97)}
.btn.ghost{background:var(--glass-hi);border:1px solid var(--border)}
.btn.ghost:hover{transform:translateY(-2px);box-shadow:0 10px 26px rgba(152,232,222,.25);border-color:var(--primary)}
.btn.small{padding:10px 14px;font-size:.8rem;border-radius:12px}
.btn[disabled]{opacity:.7;cursor:progress}
.ripple{position:absolute;border-radius:50%;background:rgba(255,255,255,.55);transform:scale(0);animation:ripple .65s ease-out forwards;pointer-events:none}
@keyframes ripple{to{transform:scale(4);opacity:0}}

/* ============================ RESULT ============================ */
.result-card{border-color:rgba(152,232,222,.45);animation:glowPulse 4s ease-in-out infinite}
@keyframes glowPulse{0%,100%{box-shadow:var(--shadow),0 0 26px rgba(152,232,222,.16)}50%{box-shadow:var(--shadow),0 0 56px rgba(152,232,222,.36)}}
.empty{text-align:center;padding:14px 6px 4px;color:var(--muted);font-size:.9rem;line-height:1.65}
.orb{width:94px;height:94px;margin:0 auto 18px;border-radius:50%;display:grid;place-items:center;font-size:2.2rem;color:var(--on-primary);position:relative;background:radial-gradient(circle at 30% 25%,var(--aqua),var(--mint) 70%);box-shadow:0 0 46px rgba(152,232,222,.6);animation:bob 4s ease-in-out infinite}
.orb::before,.orb::after{content:"";position:absolute;inset:-10px;border-radius:50%;border:2px solid var(--primary);opacity:0;animation:wave 3s ease-out infinite}
.orb::after{animation-delay:1.5s}
@keyframes wave{0%{transform:scale(.8);opacity:.7}100%{transform:scale(1.7);opacity:0}}
@keyframes bob{50%{transform:translateY(-9px)}}
/* loading: radar scanner */
.loader{text-align:center;padding:6px 0 2px}
.radar{position:relative;width:170px;height:170px;margin:0 auto 18px;border-radius:50%}
.radar .ring{position:absolute;border-radius:50%;border:1.5px solid var(--border)}
.radar .r1{inset:0}.radar .r2{inset:26px}.radar .r3{inset:52px}
.radar .sweep{position:absolute;inset:0;border-radius:50%;background:conic-gradient(from 0deg,rgba(152,232,222,.55),rgba(152,232,222,0) 28%);animation:spin 1.5s linear infinite}
.radar .core{position:absolute;inset:74px;border-radius:50%;background:var(--primary);box-shadow:0 0 22px var(--primary);animation:pulse 1.2s ease-in-out infinite}
.radar .blip{position:absolute;width:9px;height:9px;border-radius:50%;background:var(--aqua);box-shadow:0 0 12px var(--aqua);animation:blip 1.5s ease-in-out infinite}
.radar .b1{left:30px;top:44px}.radar .b2{right:34px;top:96px;animation-delay:.5s}.radar .b3{left:78px;bottom:22px;animation-delay:1s}
@keyframes blip{0%,100%{opacity:.15;transform:scale(.6)}50%{opacity:1;transform:scale(1.3)}}
.loader p{color:var(--muted);font-weight:500;min-height:1.5em}
.dots::after{content:"";animation:dots 1.4s steps(4) infinite}
@keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%,100%{content:"..."}}
/* result content */
.result:not([hidden]){animation:resultIn .75s cubic-bezier(.3,1.35,.5,1) both}
@keyframes resultIn{from{opacity:0;transform:translateY(16px) scale(.92)}to{opacity:1;transform:none}}
.cap{text-align:center;color:var(--muted);font-size:.74rem;letter-spacing:.2em;text-transform:uppercase}
.outcome{position:relative;display:flex;align-items:center;justify-content:center;margin:16px auto 24px;width:min(78%,210px);aspect-ratio:1}
.outcome .halo{position:absolute;inset:6%;border-radius:50%;background:radial-gradient(circle,var(--cc) 0,transparent 68%);opacity:.32;filter:blur(18px);animation:halo 3.2s ease-in-out infinite}
@keyframes halo{50%{transform:scale(1.18);opacity:.5}}
.outcome .ring{position:absolute;inset:0;border-radius:50%;border:2px dashed var(--cc);opacity:.55;animation:spin 26s linear infinite}
.outcome .ring2{position:absolute;inset:12%;border-radius:50%;border:2px solid var(--cc);opacity:.35;animation:spin 18s linear infinite reverse;border-top-color:transparent}
.outcome strong{position:relative;font-family:"Sora",sans-serif;font-size:clamp(2.6rem,9vw,3.8rem);letter-spacing:-.03em;color:var(--cc);text-shadow:0 0 28px var(--cc);text-transform:capitalize}
.res-yes{--cc:#98E8DE}.res-no{--cc:#B8C6FF}
[data-theme=light] .res-yes{--cc:#0d8f82}[data-theme=light] .res-no{--cc:#4a5fd0}
.verdict{display:flex;align-items:center;justify-content:center;gap:10px;margin:-6px 0 24px;padding:13px 16px;border-radius:16px;text-align:center;font-weight:600;font-size:.98rem;line-height:1.4;
  background:var(--glass-hi);border:1px solid var(--cc);box-shadow:0 0 26px -8px var(--cc)}
.verdict svg{font-size:1.25rem;color:var(--cc);flex:none}
.gauge{margin:6px 0 8px}
.gauge-top{display:flex;justify-content:space-between;align-items:baseline;font-size:.8rem;color:var(--muted);margin-bottom:10px}
.gauge-top b{font-family:"Sora",sans-serif;color:var(--text);font-size:1.05rem;font-variant-numeric:tabular-nums}
.gauge-track{position:relative;height:12px;border-radius:99px;background:linear-gradient(90deg,rgba(184,198,255,.5),var(--border) 50%,rgba(152,232,222,.55));box-shadow:inset 0 0 12px rgba(0,0,0,.25)}
.gauge-track::before{content:"";position:absolute;left:50%;top:-4px;bottom:-4px;width:2px;background:var(--muted);opacity:.6;border-radius:2px}
.gauge-marker{position:absolute;top:50%;left:50%;width:24px;height:24px;margin:-12px 0 0 -12px;border-radius:50%;background:radial-gradient(circle at 30% 30%,#fff,var(--cc));box-shadow:0 0 0 4px rgba(152,232,222,.2),0 0 22px var(--cc);transition:left 1.3s cubic-bezier(.3,1.3,.5,1)}
.gauge-ends{display:flex;justify-content:space-between;margin-top:10px;font-size:.74rem;color:var(--muted)}
.note{font-size:.76rem;color:var(--muted);line-height:1.55;text-align:center;margin-top:10px}
.probs{display:grid;gap:14px;margin-top:6px}
.prob-top{display:flex;justify-content:space-between;font-size:.85rem;margin-bottom:6px}
.bar{height:10px;border-radius:99px;background:var(--border);overflow:hidden}
.bar i{display:block;height:100%;width:0;border-radius:inherit;transition:width 1.2s var(--ease);position:relative;overflow:hidden;background:linear-gradient(90deg,var(--mint),var(--primary));box-shadow:0 0 12px rgba(152,232,222,.6)}
.bar i::after{content:"";position:absolute;inset:0;background:linear-gradient(100deg,transparent,rgba(255,255,255,.6),transparent);transform:translateX(-100%);animation:barShine 2.4s ease-in-out infinite}
@keyframes barShine{60%,100%{transform:translateX(100%)}}
.summary{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px;padding-top:18px;border-top:1px solid var(--border)}
.summary span{font-size:.76rem;padding:6px 11px;border-radius:10px;background:var(--glass-hi);border:1px solid var(--border);color:var(--muted)}
.summary b{color:var(--text);font-weight:600}

/* ============================ ANALYTICS ============================ */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px;margin-bottom:20px}
.stat{padding:22px}
.stat .ico{width:42px;height:42px;border-radius:13px;display:grid;place-items:center;font-size:1.15rem;margin-bottom:14px;color:var(--on-primary);background:linear-gradient(135deg,var(--primary),var(--mint));box-shadow:0 8px 22px rgba(152,232,222,.4)}
.stat .num{font-family:"Sora",sans-serif;font-size:clamp(1.9rem,4vw,2.4rem);font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat .lbl{color:var(--muted);font-size:.82rem;margin-top:2px}
.analytics-grid{display:grid;grid-template-columns:minmax(0,.8fr) minmax(0,2.2fr);gap:clamp(16px,2.4vw,28px);align-items:start}
.dist{display:grid;gap:18px}
.dist-row .top{display:flex;justify-content:space-between;font-size:.86rem;margin-bottom:8px}
.dist-row .top b{font-variant-numeric:tabular-nums}
.dist .bar{height:12px}
.table-wrap{overflow-x:auto;border-radius:16px;border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:.82rem;min-width:640px}
th{text-align:left;padding:12px 12px;font-weight:600;color:var(--ink);font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;background:var(--glass-hi);white-space:nowrap}
td{padding:12px 12px;border-top:1px solid var(--border);white-space:nowrap;font-variant-numeric:tabular-nums}
tbody tr{animation:rowIn .55s var(--ease) both;transition:background .25s}
tbody tr:hover{background:var(--glass-hi)}
@keyframes rowIn{from{opacity:0;transform:translateX(24px)}to{opacity:1;transform:none}}
.tag{display:inline-block;padding:4px 12px;border-radius:99px;font-weight:600;font-size:.76rem;background:rgba(152,232,222,.16);color:var(--ink);border:1px solid var(--border)}
.tag.no{background:rgba(184,198,255,.14);color:#B8C6FF}
[data-theme=light] .tag.no{color:#4a5fd0}
.empty-row{text-align:center;padding:30px 10px;color:var(--muted);font-size:.88rem}
.hist-actions{display:flex;gap:10px}

/* ============================ TOASTS / FOOTER ============================ */
.toasts{position:fixed;top:max(16px,env(safe-area-inset-top));right:16px;z-index:90;display:grid;gap:10px;width:min(370px,calc(100vw - 32px))}
.toast{position:relative;overflow:hidden;display:flex;align-items:flex-start;gap:12px;padding:15px 16px;border-radius:16px;font-size:.9rem;line-height:1.45;color:var(--text);background:var(--glass);border:1px solid var(--border);backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);box-shadow:var(--shadow);animation:toastIn .55s cubic-bezier(.3,1.4,.5,1) both}
[data-theme=dark] .toast{background:rgba(16,34,44,.92)}
[data-theme=light] .toast{background:rgba(255,255,255,.95)}
.toast svg{margin-top:2px;font-size:1.1rem;color:var(--ink)}
.toast.error svg{color:#ff8a9b}
.toast::after{content:"";position:absolute;left:0;bottom:0;height:3px;width:100%;background:var(--primary);transform-origin:left;animation:life var(--ms,4s) linear forwards}
.toast.error::after{background:#ff8a9b}
.toast.out{animation:toastOut .35s ease forwards}
@keyframes life{to{transform:scaleX(0)}}
@keyframes toastIn{from{opacity:0;transform:translateX(60px) scale(.9)}to{opacity:1;transform:none}}
@keyframes toastOut{to{opacity:0;transform:translateX(60px)}}
footer{text-align:center;margin-top:56px;color:var(--muted);font-size:.8rem}
footer b{color:var(--ink)}

/* ============================ RESPONSIVE ============================ */
@media (max-width:1020px){                      /* tablet */
  .main-grid,.analytics-grid{grid-template-columns:1fr}
  .side{position:static}
}
@media (max-width:760px){                       /* phone: nav becomes its own scrollable row */
  .header{flex-wrap:wrap;gap:10px 12px}
  .brand{flex:1}
  .nav{order:3;width:100%;margin-left:0;overflow-x:auto;justify-content:space-between}
  .nav a{flex:1;justify-content:center;padding:9px 10px}
}
@media (max-width:600px){
  .fields{grid-template-columns:1fr}.actions{flex-direction:column}.card{border-radius:21px}
  .brand small{display:none}.aurora{filter:blur(55px)}.grid-lines{background-size:48px 48px}
  .shape.hex,.shape.tri{display:none}
}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
{% endraw %}
</style>
</head>
<body>
{% raw %}
<div class="bg" aria-hidden="true">
  <div class="aurora au1"></div><div class="aurora au2"></div><div class="aurora au3"></div>
  <div class="grid-lines"></div>
  <div class="shape ring"  style="--x:6%;--y:16%;--s:150px;--d:14;--t:13s"><i></i></div>
  <div class="shape ring2" style="--x:78%;--y:8%;--s:210px;--d:-18;--t:17s;--dl:-4s"><i></i></div>
  <div class="shape hex"   style="--x:86%;--y:58%;--s:120px;--d:22;--t:15s;--dl:-2s"><i></i></div>
  <div class="shape tri"   style="--x:10%;--y:68%;--s:100px;--d:-20;--t:12s;--dl:-6s"><i></i></div>
  <div class="shape dia"   style="--x:48%;--y:82%;--s:70px;--d:16;--t:14s;--dl:-3s"><i></i></div>
  <div class="shape orb"   style="--x:70%;--y:24%;--s:90px;--d:-26;--t:11s;--dl:-5s"><i></i></div>
  <div class="shape orb"   style="--x:22%;--y:42%;--s:60px;--d:24;--t:10s;--dl:-7s"><i></i></div>
  <div class="shape ring"  style="--x:92%;--y:86%;--s:90px;--d:-12;--t:16s;--dl:-9s"><i></i></div>
</div>
<canvas id="particles" aria-hidden="true"></canvas>
<canvas id="burst" aria-hidden="true"></canvas>
<div class="toasts" id="toasts" aria-live="polite"></div>
{% endraw %}

<div class="wrap">
  <header class="header">
    <a class="brand" href="#home" aria-label="{{ app_name }} home">
      <svg class="logo" viewBox="0 0 48 48" aria-hidden="true">
        <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#B8FFF3"/><stop offset="1" stop-color="#6FCFC1"/></linearGradient></defs>
        <path d="M24 3 42 13.5v21L24 45 6 34.5v-21Z" fill="rgba(152,232,222,.14)" stroke="url(#lg)" stroke-width="2.4" stroke-linejoin="round"/>
        <path d="M15 29 21 21l5 5 7-10" fill="none" stroke="url(#lg)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="33" cy="16" r="2.6" fill="#B8FFF3"/>
      </svg>
      <div><strong>{{ app_name }}</strong><small>{{ tagline }}</small></div>
    </a>
    <nav class="nav" id="nav" aria-label="Sections">
      <a href="#home" data-target="home" class="active"><span data-i="home"></span>Home</a>
      <a href="#predict" data-target="predict"><span data-i="sparkles"></span>Predict</a>
      <a href="#analytics" data-target="analytics"><span data-i="chart"></span>Analytics</a>
    </nav>
    <button class="icon-btn" id="themeToggle" type="button" aria-label="Toggle dark / light mode"><span class="moon" data-i="moon"></span><span class="sun" data-i="sun"></span></button>
  </header>

  <main>
    <section class="hero reveal" id="home">
      <span class="pill"><span class="dot"></span>Instant profile-based prediction</span>
      <h1>Know the likely outcome<br><span class="grad">for any profile</span></h1>
      <p>Enter a person's age, gender, region, occupation and income. Our trained model returns a clear yes / no prediction in a moment, and keeps a live log of every result in your session.</p>
      <div class="chips">
        <span class="chip"><span data-i="target"></span><span id="chipInputs">5 inputs</span></span>
        <span class="chip"><span data-i="check"></span>Laptop: yes / no</span>
        <span class="chip"><span data-i="bolt"></span>Instant result</span>
        <span class="chip"><span data-i="download"></span>CSV export</span>
      </div>
    </section>

    <section id="predict">
      <div class="section-head reveal"><span data-i="sparkles"></span><h2>Predict</h2></div>
      <div class="main-grid">
        <div class="card reveal" id="formCard">
          <div class="card-head"><h3><span data-i="sliders"></span>Profile details</h3><span class="muted" id="countLabel"></span></div>
          <div class="notice" id="notice" hidden>
            <span data-i="info"></span>
            <span><b>Category names not set yet.</b> <span id="noticeFields"></span> still show the numeric codes the model was trained with. Add the real names to <code>CATEGORY_LABELS</code> in <code>app.py</code> to show them here.</span>
            <button type="button" id="noticeClose" aria-label="Dismiss">&times;</button>
          </div>
          <div class="progress"><div class="progress-top"><span>Profile completion</span><b id="progText">0%</b></div><div class="progress-bar"><i id="progBar"></i></div></div>
          <form id="predictForm" novalidate>
            <div id="sections"></div>
            <div class="actions">
              <button type="submit" class="btn primary" id="predictBtn"><span data-i="sparkles"></span><span id="predictLabel">Predict Outcome</span></button>
              <button type="button" class="btn ghost" id="resetBtn"><span data-i="reset"></span>Reset</button>
            </div>
          </form>
        </div>

        <aside class="side">
          <div class="card result-card reveal" id="resultCard" style="--rd:.12s">
            <div class="card-head"><h3><span data-i="target"></span>Prediction result</h3></div>
            <div class="empty" id="emptyState"><div class="orb" data-i="sparkles"></div><p>Complete the profile and press <b>Predict Outcome</b>. Your result appears here.</p></div>
            <div class="loader" id="loader" hidden>
              <div class="radar"><span class="ring r1"></span><span class="ring r2"></span><span class="ring r3"></span><span class="sweep"></span><span class="blip b1"></span><span class="blip b2"></span><span class="blip b3"></span><span class="core"></span></div>
              <p><span id="loadMsg">Reading the profile</span><span class="dots"></span></p>
            </div>
            <div class="result" id="result" hidden>
              <div class="cap" id="capText"></div>
              <div class="outcome" id="outcome"><span class="halo"></span><span class="ring"></span><span class="ring2"></span><strong id="predLabel">-</strong></div>
              <div class="verdict" id="verdict"><span id="verdictIcon"></span><span id="verdictText"></span></div>
              <div class="gauge" id="gauge" hidden>
                <div class="gauge-top"><span>Decision score</span><b id="scoreVal">0</b></div>
                <div class="gauge-track"><span class="gauge-marker" id="gaugeMarker"></span></div>
                <div class="gauge-ends"><span id="endNeg">no</span><span id="endPos">yes</span></div>
                <p class="note" id="scoreNote">The decision score is the signed distance from the model's decision boundary. It is not a probability.</p>
              </div>
              <div class="probs" id="probs" hidden></div>
              <div class="summary" id="summary"></div>
            </div>
          </div>
        </aside>
      </div>
    </section>

    <section id="analytics">
      <div class="section-head reveal"><span data-i="chart"></span><h2>Analytics</h2></div>
      <div class="stats" id="stats"></div>
      <div class="analytics-grid">
        <div class="card reveal">
          <div class="card-head"><h3><span data-i="chart"></span>Class-wise predictions</h3></div>
          <div class="dist" id="dist"></div>
        </div>
        <div class="card reveal" style="--rd:.1s">
          <div class="card-head">
            <div><h3><span data-i="clock"></span>Prediction history</h3><span class="muted">Predictions made during this session</span></div>
            <div class="hist-actions">
              <button class="btn small ghost" id="csvBtn" type="button"><span data-i="download"></span>CSV</button>
              <button class="btn small ghost" id="clearHistory" type="button"><span data-i="trash"></span>Clear</button>
            </div>
          </div>
          <div class="table-wrap">
            <table>
              <thead><tr id="histHead"></tr></thead>
              <tbody id="historyBody"></tbody>
            </table>
            <div class="empty-row" id="historyEmpty">Your prediction history will appear here.</div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <footer>Crafted with <b>Flask</b> &amp; <b>scikit-learn</b> &middot; Hosted on Render</footer>
</div>

<script>window.__CFG__ = {{ cfg | tojson }};</script>
<script>
{% raw %}
(() => {
"use strict";
const CFG = window.__CFG__;
const { fields, classes, hasProba, hasScore, labelsMissing, modelOk, caption, outcomeLabels, outcomeSentences } = CFG;
const $ = (id) => document.getElementById(id);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const HISTORY_KEY = "profileiq_history";
const Q = String.fromCharCode(34);
const cap = (s) => String(s).charAt(0).toUpperCase() + String(s).slice(1);
const POS = classes[classes.length - 1], NEG = classes[0];
const lab = (c) => outcomeLabels[c] || cap(c);          // friendly text for a class

/* ---------------- icons (inline SVG, no external library) ---------------- */
const P = {
  home: '<path d="m3 10 9-7 9 7v10a2 2 0 0 1-2 2h-4v-7H9v7H5a2 2 0 0 1-2-2Z"/>',
  sparkles: '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/>',
  chart: '<path d="M3 3v18h18"/><path d="M18 17V9M13 17V5M8 17v-3"/>',
  moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
  target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
  check: '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>',
  bolt: '<path d="M13 2 3 14h9l-1 8 10-12h-9Z"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
  trash: '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
  reset: '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
  sliders: '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
  alert: '<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>',
  clock: '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
  user: '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
  users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
  pin: '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
  briefcase: '<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
  cash: '<rect x="2" y="6" width="20" height="12" rx="2"/><circle cx="12" cy="12" r="2"/><path d="M6 12h.01M18 12h.01"/>',
  dot: '<circle cx="12" cy="12" r="3"/>',
  laptop: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M2 20h20"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  layers: '<path d="m12 2 10 5-10 5L2 7Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
  percent: '<path d="m19 5-14 14"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
  chev: '<path d="m6 9 6 6 6-6"/>',
};
const icon = (n) => `<svg class="ic" viewBox="0 0 24 24" aria-hidden="true">${P[n] || P.dot}</svg>`;
document.querySelectorAll("[data-i]").forEach((el) => { el.innerHTML = icon(el.dataset.i); });

/* ---------------- theme ---------------- */
$("themeToggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) {}
});

/* ---------------- toasts ---------------- */
function toast(type, msg, ms = 4200) {
  const el = document.createElement("div");
  el.className = "toast " + type;
  el.style.setProperty("--ms", ms + "ms");
  el.innerHTML = icon(type === "error" ? "alert" : type === "success" ? "check" : "info") + "<span></span>";
  el.querySelector("span").textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 350); }, ms);
}

/* ---------------- build the form from the server config ---------------- */
const form = $("predictForm");
const groups = [];
fields.forEach((f) => { if (!groups.includes(f.group)) groups.push(f.group); });
groups.forEach((g) => {
  const sec = document.createElement("div");
  sec.className = "section";
  sec.innerHTML = '<div class="section-title"><span></span></div><div class="fields"></div>';
  sec.querySelector(".section-title span").textContent = g;
  const box = sec.querySelector(".fields");
  fields.filter((f) => f.group === g).forEach((f) => {
    const d = document.createElement("div");
    d.className = "field" + (f.wide ? " wide" : "");
    d.dataset.name = f.name;
    let control;
    if (f.kind === "category") {
      control = '<div class="control"><select id="f_' + f.name + '" name="' + f.name + '" required><option value="" selected disabled>Select&hellip;</option></select>' +
        '<span class="chev">' + icon("chev") + '</span><i class="underline"></i></div>';
    } else {
      control = '<div class="control"><input id="f_' + f.name + '" name="' + f.name + '" type="number" inputmode="decimal" min="' + f.min + '" max="' + f.max +
        '" step="' + f.step + '" required><i class="underline"></i></div>';
    }
    d.innerHTML = '<label for="f_' + f.name + '">' + icon(f.icon) + '<span></span></label>' + control + '<small class="hint"></small><small class="error-text"></small>';
    d.querySelector("label span").textContent = f.label;
    d.querySelector(".hint").textContent = f.hint;
    if (f.kind === "category") {
      const sel = d.querySelector("select");
      f.options.forEach((o) => { const op = document.createElement("option"); op.value = String(o.code); op.textContent = o.label; sel.appendChild(op); });
    } else {
      d.querySelector("input").placeholder = f.placeholder || "Enter value";
    }
    box.appendChild(d);
  });
  $("sections").appendChild(sec);
});
const control = (name) => form.elements[name];
$("countLabel").textContent = fields.length + " inputs";
$("chipInputs").textContent = fields.length + " inputs";
if (labelsMissing.length) {
  $("noticeFields").textContent = labelsMissing.length > 1 ? labelsMissing.slice(0, -1).join(", ") + " and " + labelsMissing[labelsMissing.length - 1] : labelsMissing[0];
  $("notice").hidden = false;
}
$("noticeClose").addEventListener("click", () => ($("notice").hidden = true));

/* ---------------- completion progress ---------------- */
function updateProgress() {
  let done = 0;
  fields.forEach((f) => {
    const filled = String(control(f.name).value).trim() !== "";
    form.querySelector('.field[data-name="' + f.name + '"]').classList.toggle("filled", filled);
    if (filled) done++;
  });
  const pct = Math.round((done / fields.length) * 100);
  $("progBar").style.width = pct + "%";
  $("progText").textContent = done + "/" + fields.length + " \u00b7 " + pct + "%";
}
form.addEventListener("input", (e) => { const w = e.target.closest(".field"); if (w) w.classList.remove("invalid"); updateProgress(); });
form.addEventListener("change", updateProgress);
updateProgress();

function setFieldError(name, msg) {
  const w = form.querySelector('.field[data-name="' + name + '"]');
  if (!w) return;
  w.classList.add("invalid");
  w.querySelector(".error-text").textContent = msg;
}
const clearErrors = () => form.querySelectorAll(".field.invalid").forEach((f) => f.classList.remove("invalid"));

/* human-readable text of what the user entered (select shows its label) */
function displayValue(f) {
  const el = control(f.name);
  return f.kind === "category" ? el.options[el.selectedIndex].textContent : el.value.trim();
}

/* ---------------- result panel ---------------- */
const emptyState = $("emptyState"), loader = $("loader"), resultBox = $("result");
let hasLast = false;
function show(state) {
  emptyState.hidden = state !== "empty";
  loader.hidden = state !== "loading";
  resultBox.hidden = state !== "result";
}

const MESSAGES = ["Reading the profile", "Preparing the inputs", "Running the model", "Finalising the result"];
let msgTimer;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearErrors();
  const data = {}; let missing = 0;
  fields.forEach((f) => {
    data[f.name] = String(control(f.name).value).trim();
    if (data[f.name] === "") { missing++; setFieldError(f.name, "This field is required."); }
  });
  if (missing) {
    toast("error", "Please complete " + missing + " more field" + (missing > 1 ? "s" : "") + " before predicting.");
    const bad = form.querySelector(".field.invalid");
    if (bad) bad.scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }
  const shown = {}; fields.forEach((f) => { shown[f.name] = displayValue(f); });

  const btn = $("predictBtn");
  btn.disabled = true; $("predictLabel").textContent = "Predicting\u2026";
  show("loading");
  let m = 0; $("loadMsg").textContent = MESSAGES[0];
  msgTimer = setInterval(() => { m = (m + 1) % MESSAGES.length; $("loadMsg").textContent = MESSAGES[m]; }, 700);
  if (innerWidth < 1020) $("resultCard").scrollIntoView({ behavior: "smooth", block: "center" });

  try {
    const [res] = await Promise.all([
      fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) }),
      new Promise((r) => setTimeout(r, 1700)),            // keep the loading animation visible
    ]);
    let out = {};
    try { out = await res.json(); } catch (_) {}
    if (!res.ok) {
      if (out.fields) Object.keys(out.fields).forEach((k) => setFieldError(k, out.fields[k]));
      throw new Error(out.error || "The server returned an error (" + res.status + "). Please try again.");
    }
    renderResult(out, shown);
    addHistory(shown, out);
    toast("success", lab(out.prediction));
  } catch (err) {
    show(hasLast ? "result" : "empty");
    toast("error", err.message || "Network problem. Please try again.", 5600);
  } finally {
    clearInterval(msgTimer);
    btn.disabled = false; $("predictLabel").textContent = "Predict Outcome";
  }
});

function renderResult(out, shown) {
  hasLast = true;
  show("result");
  $("capText").textContent = caption;
  resultBox.className = "result " + (out.prediction === POS ? "res-yes" : "res-no");
  resultBox.style.animation = "none"; void resultBox.offsetWidth; resultBox.style.animation = "";   // replay the entrance animation for every new result
  $("predLabel").textContent = out.prediction;
  $("verdictIcon").innerHTML = icon(out.prediction === POS ? "laptop" : "x");
  $("verdictText").textContent = outcomeSentences[out.prediction] || ("Predicted: " + lab(out.prediction));

  // decision score gauge - only when the model returns a score
  const g = $("gauge");
  g.hidden = !(hasScore && out.score !== null && out.score !== undefined);
  if (!g.hidden) {
    $("scoreVal").textContent = (out.score >= 0 ? "+" : "") + out.score.toFixed(4);
    $("endNeg").textContent = lab(NEG); $("endPos").textContent = lab(POS);
    const pos = 50 + 50 * Math.tanh(out.score * 4);              // visual position only; exact score shown above
    const mk = $("gaugeMarker");
    mk.style.left = "50%";
    requestAnimationFrame(() => requestAnimationFrame(() => { mk.style.left = pos + "%"; }));
  }

  // probability bars - only shown if the server returned probabilities
  const pb = $("probs");
  pb.innerHTML = "";
  pb.hidden = !(hasProba && out.probabilities);
  if (!pb.hidden) {
    classes.forEach((c) => {
      const pct = out.probabilities[c] || 0;
      const row = document.createElement("div");
      row.innerHTML = '<div class="prob-top"><span></span><b></b></div><div class="bar"><i></i></div>';
      row.querySelector("span").textContent = lab(c);
      row.querySelector("b").textContent = pct.toFixed(1) + "%";
      pb.appendChild(row);
      const bar = row.querySelector("i");
      requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = pct + "%"; }));
    });
  }

  const sm = $("summary"); sm.innerHTML = "";
  fields.forEach((f) => {
    const s = document.createElement("span");
    s.innerHTML = "<b></b>";
    s.querySelector("b").textContent = f.label + ": ";
    s.appendChild(document.createTextNode(shown[f.name]));
    sm.appendChild(s);
  });
  burst();
}

/* ---------------- reset ---------------- */
$("resetBtn").addEventListener("click", () => {
  form.reset(); clearErrors(); updateProgress(); hasLast = false; show("empty");
  toast("info", "All fields cleared.");
});

/* ---------------- analytics: history, counters, CSV ---------------- */
const loadHistory = () => { try { return JSON.parse(sessionStorage.getItem(HISTORY_KEY)) || []; } catch (e) { return []; } };
const saveHistory = (l) => { try { sessionStorage.setItem(HISTORY_KEY, JSON.stringify(l)); } catch (e) {} };

// stat cards (built once, numbers animate)
const statDefs = [{ key: "total", label: "Total predictions", ic: "layers" }]
  .concat(classes.map((c) => ({ key: "c_" + c, label: lab(c), ic: c === POS ? "check" : "target" })))
  .concat([{ key: "rate", label: lab(POS) + " (%)", ic: "percent" }]);
const statEls = {}, statShown = {};
statDefs.forEach((s) => {
  const el = document.createElement("div");
  el.className = "card stat reveal";
  el.innerHTML = '<div class="ico">' + icon(s.ic) + '</div><div class="num">0</div><div class="lbl"></div>';
  el.querySelector(".lbl").textContent = s.label;
  $("stats").appendChild(el);
  statEls[s.key] = el.querySelector(".num"); statShown[s.key] = 0;
});
function tween(key, target, suffix) {
  const el = statEls[key], from = statShown[key], t0 = performance.now(), dur = reduceMotion ? 1 : 900;
  statShown[key] = target;
  (function tick(now) {
    const p = Math.min(1, (now - t0) / dur), v = from + (target - from) * (1 - Math.pow(1 - p, 3));
    el.textContent = (suffix ? v.toFixed(1) : Math.round(v)) + (suffix || "");
    if (p < 1) requestAnimationFrame(tick);
  })(t0);
}

// distribution bars (built once)
const distEls = {};
classes.forEach((c) => {
  const row = document.createElement("div");
  row.className = "dist-row";
  row.innerHTML = '<div class="top"><span></span><b>0 (0%)</b></div><div class="bar"><i></i></div>';
  row.querySelector("span").textContent = lab(c);
  $("dist").appendChild(row);
  distEls[c] = { b: row.querySelector("b"), bar: row.querySelector("i") };
});

// history table header
const histCols = ["#", "Time"].concat(fields.map((f) => f.label), ["Prediction"], hasScore ? ["Score"] : []);
histCols.forEach((h) => { const th = document.createElement("th"); th.textContent = h; $("histHead").appendChild(th); });

function addHistory(inputs, out) {
  const l = loadHistory();
  l.unshift({ time: new Date().toISOString(), prediction: out.prediction, score: out.score, inputs: inputs });
  saveHistory(l.slice(0, 100));
  renderAnalytics();
}

function renderAnalytics() {
  const l = loadHistory(), total = l.length, body = $("historyBody");
  const counts = {}; classes.forEach((c) => { counts[c] = 0; });
  l.forEach((it) => { if (counts[it.prediction] !== undefined) counts[it.prediction]++; });

  tween("total", total);
  classes.forEach((c) => {
    tween("c_" + c, counts[c]);
    const pct = total ? (counts[c] / total) * 100 : 0;
    distEls[c].b.textContent = counts[c] + " (" + pct.toFixed(0) + "%)";
    distEls[c].bar.style.width = pct + "%";
  });
  tween("rate", total ? (counts[POS] / total) * 100 : 0, "%");

  body.innerHTML = "";
  l.forEach((it, i) => {
    const tr = document.createElement("tr");
    tr.style.animationDelay = Math.min(i, 8) * 40 + "ms";
    const cells = [String(total - i), new Date(it.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })]
      .concat(fields.map((f) => it.inputs[f.name]));
    cells.forEach((c) => { const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); });
    const tdp = document.createElement("td");
    tdp.innerHTML = '<span class="tag"></span>';
    const tag = tdp.firstChild; tag.textContent = lab(it.prediction); if (it.prediction !== POS) tag.classList.add("no");
    tr.appendChild(tdp);
    if (hasScore) { const tds = document.createElement("td"); tds.textContent = (it.score === null || it.score === undefined) ? "-" : Number(it.score).toFixed(4); tr.appendChild(tds); }
    body.appendChild(tr);
  });
  $("historyEmpty").hidden = total > 0;
}

$("clearHistory").addEventListener("click", () => {
  if (!loadHistory().length) { toast("info", "History is already empty."); return; }
  saveHistory([]); renderAnalytics(); toast("info", "History cleared.");
});

const csvCell = (v) => Q + String(v === null || v === undefined ? "" : v).split(Q).join(Q + Q) + Q;
$("csvBtn").addEventListener("click", () => {
  const l = loadHistory();
  if (!l.length) { toast("info", "No predictions to download yet."); return; }
  const header = ["Time", "Prediction"].concat(hasScore ? ["Decision_Score"] : [], fields.map((f) => f.name));
  const rows = l.slice().reverse().map((it) => [it.time, it.prediction].concat(hasScore ? [it.score] : [], fields.map((f) => it.inputs[f.name])));
  const csv = [header].concat(rows).map((r) => r.map(csvCell).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8;" }));
  const a = document.createElement("a");
  a.href = url; a.download = "prediction_history.csv";
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("success", "Prediction history downloaded.");
});

/* ---------------- effects ---------------- */
// button ripple
document.addEventListener("pointerdown", (e) => {
  const b = e.target.closest(".btn"); if (!b || reduceMotion) return;
  const r = b.getBoundingClientRect(), s = Math.max(r.width, r.height) / 2;
  const sp = document.createElement("span"); sp.className = "ripple";
  sp.style.cssText = "width:" + s + "px;height:" + s + "px;left:" + (e.clientX - r.left - s / 2) + "px;top:" + (e.clientY - r.top - s / 2) + "px";
  b.appendChild(sp); setTimeout(() => sp.remove(), 700);
});
// cursor spotlight on cards
document.querySelectorAll(".card").forEach((c) => {
  c.addEventListener("pointermove", (e) => {
    const r = c.getBoundingClientRect();
    c.style.setProperty("--cx", (e.clientX - r.left) + "px"); c.style.setProperty("--cy", (e.clientY - r.top) + "px");
  });
});
// parallax for the floating background shapes
if (!reduceMotion) addEventListener("pointermove", (e) => {
  const root = document.documentElement.style;
  root.setProperty("--mx", (e.clientX / innerWidth - .5).toFixed(3));
  root.setProperty("--my", (e.clientY / innerHeight - .5).toFixed(3));
});
// reveal-on-scroll (staggered card entrance)
const revealEls = document.querySelectorAll(".reveal");
if ("IntersectionObserver" in window) {
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
  }, { threshold: .08 });
  revealEls.forEach((el, i) => { if (!el.style.getPropertyValue("--rd")) el.style.setProperty("--rd", (i % 4) * 0.08 + "s"); io.observe(el); });
} else { revealEls.forEach((el) => el.classList.add("in")); }
// nav highlight follows the scroll position
const links = document.querySelectorAll(".nav a");
if ("IntersectionObserver" in window) {
  const so = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.dataset.target === en.target.id)); });
  }, { rootMargin: "-35% 0px -55% 0px" });
  ["home", "predict", "analytics"].forEach((id) => so.observe($(id)));
}

// floating particles with soft connecting lines
(function particles() {
  const cv = $("particles"), ctx = cv.getContext("2d");
  let W, H, d, P2 = [], raf, mx = -9999, my = -9999;
  function size() {
    d = Math.min(devicePixelRatio || 1, 2); W = cv.width = innerWidth * d; H = cv.height = innerHeight * d;
    const n = Math.round(Math.min(70, Math.max(24, innerWidth / 20)));
    P2 = Array.from({ length: n }, () => ({ x: Math.random() * W, y: Math.random() * H, r: (Math.random() * 1.8 + .7) * d,
      vx: (Math.random() - .5) * .3 * d, vy: (Math.random() - .5) * .3 * d, t: Math.random() * 6.28 }));
  }
  addEventListener("pointermove", (e) => { mx = e.clientX * d; my = e.clientY * d; });
  function frame() {
    ctx.clearRect(0, 0, W, H);
    const light = document.documentElement.dataset.theme === "light", rgb = light ? "13,111,101" : "152,232,222";
    for (let i = 0; i < P2.length; i++) {
      const p = P2[i];
      const dx = p.x - mx, dy = p.y - my, dist = Math.hypot(dx, dy);
      if (dist < 130 * d && dist > 0) { p.x += (dx / dist) * 1.2; p.y += (dy / dist) * 1.2; }
      p.x += p.vx; p.y += p.vy; p.t += .02;
      if (p.x < -10) p.x = W + 10; if (p.x > W + 10) p.x = -10;
      if (p.y < -10) p.y = H + 10; if (p.y > H + 10) p.y = -10;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.2832);
      ctx.fillStyle = "rgba(" + rgb + "," + (.25 + .45 * Math.abs(Math.sin(p.t))) + ")";
      ctx.shadowColor = "rgba(" + rgb + ",.9)"; ctx.shadowBlur = 10 * d; ctx.fill();
      for (let j = i + 1; j < P2.length; j++) {
        const q = P2[j], l = Math.hypot(p.x - q.x, p.y - q.y);
        if (l < 120 * d) { ctx.shadowBlur = 0; ctx.strokeStyle = "rgba(" + rgb + "," + (.14 * (1 - l / (120 * d))) + ")"; ctx.lineWidth = d; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke(); }
      }
    }
    raf = requestAnimationFrame(frame);
  }
  size(); addEventListener("resize", size);
  document.addEventListener("visibilitychange", () => { cancelAnimationFrame(raf); if (!document.hidden && !reduceMotion) frame(); });
  if (!reduceMotion) frame();
})();

// mint sparkle burst when a result arrives
function burst() {
  if (reduceMotion) return;
  const cv = $("burst"), ctx = cv.getContext("2d"), d = Math.min(devicePixelRatio || 1, 2);
  cv.width = innerWidth * d; cv.height = innerHeight * d;
  const r = $("outcome").getBoundingClientRect();
  const ox = (r.left + r.width / 2) * d, oy = (r.top + r.height / 2) * d;
  const cols = ["#98E8DE", "#B8FFF3", "#6FCFC1", "#ffffff"];
  const S = Array.from({ length: 64 }, () => { const a = Math.random() * 6.28, s = (Math.random() * 6 + 2) * d;
    return { x: ox, y: oy, vx: Math.cos(a) * s, vy: Math.sin(a) * s - 2 * d, r: (Math.random() * 3 + 1.4) * d, life: 1, c: cols[Math.floor(Math.random() * cols.length)] }; });
  (function loop() {
    ctx.clearRect(0, 0, cv.width, cv.height);
    let alive = false;
    S.forEach((p) => {
      if (p.life <= 0) return;
      alive = true;
      p.x += p.vx; p.y += p.vy; p.vy += .15 * d; p.vx *= .985; p.life -= .017;
      ctx.globalAlpha = Math.max(p.life, 0); ctx.fillStyle = p.c; ctx.shadowColor = p.c; ctx.shadowBlur = 10 * d;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.2832); ctx.fill();
    });
    ctx.globalAlpha = 1;
    if (alive) requestAnimationFrame(loop); else ctx.clearRect(0, 0, cv.width, cv.height);
  })();
}

/* ---------------- start ---------------- */
show("empty");
renderAnalytics();
if (!modelOk) toast("error", "The model file could not be loaded on the server. Check the Render logs.", 9000);
})();
{% endraw %}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    # Render provides the port in the PORT environment variable.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
