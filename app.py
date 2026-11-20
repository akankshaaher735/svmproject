import csv
import io
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm.pkl"

EXPECTED_FEATURES = [
    "Age",
    "Gender",
    "Region",
    "Occupation",
    "Income",
]

FEATURE_DESCRIPTIONS = {
    "Age": "Applicant age (numeric)",
    "Gender": "Applicant gender category",
    "Region": "Applicant region category",
    "Occupation": "Applicant occupation category",
    "Income": "Applicant income (numeric)",
}

CATEGORICAL_OPTIONS = {
    "Gender": [
        {"label": "Category 0 (Female / Group A)", "value": 0},
        {"label": "Category 1 (Male / Group B)", "value": 1},
    ],
    "Region": [
        {"label": "Region 0 (North)", "value": 0},
        {"label": "Region 1 (South)", "value": 1},
        {"label": "Region 2 (East)", "value": 2},
        {"label": "Region 3 (West)", "value": 3},
    ],
    "Occupation": [
        {"label": "Occupation 0 (Salaried)", "value": 0},
        {"label": "Occupation 1 (Self-Employed)", "value": 1},
        {"label": "Occupation 2 (Student)", "value": 2},
        {"label": "Occupation 3 (Other)", "value": 3},
    ],
}


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    with open(path, "rb") as file:
        return pickle.load(file)


MODEL = load_model(MODEL_PATH)

FEATURE_ORDER = (
    list(MODEL.feature_names_in_)
    if hasattr(MODEL, "feature_names_in_") and MODEL.feature_names_in_ is not None
    else EXPECTED_FEATURES
)

CLASSES = (
    [str(c) for c in MODEL.classes_]
    if hasattr(MODEL, "classes_") and MODEL.classes_ is not None
    else ["no", "yes"]
)

SUPPORTS_PROBA = bool(getattr(MODEL, "probability", False) and hasattr(MODEL, "predict_proba"))

SESSION_HISTORY: list[dict[str, Any]] = []
CLASS_COUNTS: dict[str, int] = {cls: 0 for cls in CLASSES}


def build_feature_vector(payload: dict[str, Any]) -> np.ndarray:
    values: list[float] = []
    for feature in FEATURE_ORDER:
        if feature not in payload:
            raise ValueError(f"Missing required feature: '{feature}'.")
        raw_val = payload[feature]
        if raw_val is None or str(raw_val).strip() == "":
            raise ValueError(f"Feature '{feature}' cannot be empty.")
        try:
            num_val = float(raw_val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{feature}' must be a valid number.") from exc
        if not np.isfinite(num_val):
            raise ValueError(f"Feature '{feature}' must be a finite number.")
        values.append(num_val)
    return np.array([values], dtype=np.float64)


HTML_LINES = [
    "<!DOCTYPE html>",
    '<html lang="en" data-theme="dark">',
    "<head>",
    '  <meta charset="UTF-8" />',
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />',
    "  <title>MintTurquoise SVM Studio</title>",
    '  <link rel="preconnect" href="https://fonts.googleapis.com">',
    '  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
    '  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">',
    "  <style>",
    "    :root {",
    "      --mint: #98E8DE;",
    "      --navy: #101827;",
    "      --teal: #163A40;",
    "      --aqua: #B8FFF3;",
    "      --muted-mint: #6FCFC1;",
    "      --text: #F5FAFA;",
    "      --text-sec: #AAB8C2;",
    "      --glass: rgba(22, 58, 64, 0.56);",
    "      --border: rgba(152, 232, 222, 0.28);",
    "      --input-bg: rgba(16, 24, 39, 0.78);",
    "      --shadow: 0 20px 46px rgba(10, 15, 25, 0.65), 0 0 24px rgba(152, 232, 222, 0.14);",
    "    }",
    '    html[data-theme="light"] {',
    "      --mint: #149689;",
    "      --navy: #F5FAFA;",
    "      --teal: #DDF6F2;",
    "      --aqua: #0E7C72;",
    "      --muted-mint: #24A89B;",
    "      --text: #101827;",
    "      --text-sec: #3B5259;",
    "      --glass: rgba(255, 255, 255, 0.78);",
    "      --border: rgba(20, 150, 137, 0.28);",
    "      --input-bg: rgba(255, 255, 255, 0.94);",
    "      --shadow: 0 16px 36px rgba(16, 24, 39, 0.1), 0 0 18px rgba(20, 150, 137, 0.12);",
    "    }",
    "    * { box-sizing: border-box; margin: 0; padding: 0; }",
    "    body {",
    '      font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;',
    "      color: var(--text);",
    "      min-height: 100vh;",
    "      background: linear-gradient(135deg, var(--navy) 0%, var(--teal) 52%, var(--navy) 100%);",
    "      background-size: 220% 220%;",
    "      animation: bgFlow 18s ease infinite;",
    "      overflow-x: hidden;",
    "      transition: background 0.35s ease, color 0.3s ease;",
    "    }",
    "    @keyframes bgFlow {",
    "      0% { background-position: 0% 50%; }",
    "      50% { background-position: 100% 50%; }",
    "      100% { background-position: 0% 50%; }",
    "    }",
    "    #particles-canvas {",
    "      position: fixed; inset: 0; width: 100%; height: 100%;",
    "      z-index: 0; pointer-events: none; opacity: 0.65;",
    "    }",
    "    .floating-shapes {",
    "      position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden;",
    "    }",
    "    .shape {",
    "      position: absolute; border-radius: 999px;",
    "      background: radial-gradient(circle, rgba(152, 232, 222, 0.16), rgba(111, 207, 193, 0.02));",
    "      border: 1px solid rgba(184, 255, 243, 0.14);",
    "      animation: shapeDrift 14s ease-in-out infinite;",
    "    }",
    "    .shape.s1 { width: 240px; height: 240px; top: 8%; left: -60px; animation-delay: 0s; }",
    "    .shape.s2 { width: 170px; height: 170px; top: 58%; right: -40px; border-radius: 36px; animation-delay: -4s; }",
    "    .shape.s3 { width: 120px; height: 120px; bottom: 8%; left: 18%; animation-delay: -8s; }",
    "    @keyframes shapeDrift {",
    "      0%, 100% { transform: translateY(0px) rotate(0deg); }",
    "      50% { transform: translateY(-18px) rotate(8deg); }",
    "    }",
    "    .shell {",
    "      position: relative; z-index: 1; max-width: 1240px;",
    "      margin: 0 auto; padding: 20px 20px 52px;",
    "      animation: pageIn 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;",
    "    }",
    "    @keyframes pageIn {",
    "      from { opacity: 0; transform: translateY(14px); }",
    "      to { opacity: 1; transform: translateY(0); }",
    "    }",
    "    .top-header {",
    "      display: flex; align-items: center; justify-content: space-between; gap: 16px;",
    "      padding: 15px 22px; border-radius: 20px;",
    "      background: var(--glass); backdrop-filter: blur(16px);",
    "      border: 1px solid var(--border); box-shadow: var(--shadow); margin-bottom: 24px;",
    "    }",
    "    .brand { display: flex; align-items: center; gap: 12px; }",
    "    .logo-box {",
    "      width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center;",
    "      background: linear-gradient(135deg, var(--mint), var(--muted-mint));",
    "      color: #101827; box-shadow: 0 0 18px rgba(152, 232, 222, 0.45);",
    "    }",
    "    .app-title { font-size: 1.22rem; font-weight: 800; letter-spacing: -0.02em; color: var(--mint); }",
    "    .tagline { font-size: 0.8rem; color: var(--text-sec); }",
    "    .header-controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }",
    "    .btn {",
    '      font-family: "Inter", sans-serif; font-size: 0.86rem; font-weight: 600;',
    "      padding: 10px 16px; border-radius: 12px; border: 1px solid var(--border);",
    "      background: rgba(152, 232, 222, 0.1); color: var(--text);",
    "      cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; gap: 8px;",
    "      transition: transform 0.2s ease, background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;",
    "    }",
    "    .btn:hover {",
    "      transform: translateY(-2px); border-color: var(--aqua);",
    "      background: rgba(152, 232, 222, 0.2); box-shadow: 0 0 16px rgba(152, 232, 222, 0.25);",
    "    }",
    "    .btn:active { transform: translateY(0) scale(0.98); }",
    "    .btn-primary {",
    "      background: linear-gradient(135deg, var(--mint), var(--muted-mint));",
    "      color: #101827; border-color: var(--aqua);",
    "      box-shadow: 0 8px 22px rgba(111, 207, 193, 0.35);",
    "    }",
    "    .btn-primary:hover { box-shadow: 0 12px 28px rgba(184, 255, 243, 0.45); }",
    "    .glass-card {",
    "      background: var(--glass); backdrop-filter: blur(16px);",
    "      border: 1px solid var(--border); border-radius: 22px;",
    "      padding: 24px; box-shadow: var(--shadow);",
    "      transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;",
    "    }",
    "    .glass-card:hover {",
    "      transform: translateY(-3px); border-color: var(--aqua);",
    "      box-shadow: 0 22px 50px rgba(10, 15, 25, 0.72), 0 0 26px rgba(152, 232, 222, 0.2);",
    "    }",
    "    .welcome-box { margin-bottom: 22px; }",
    "    .welcome-box h2 { font-size: 1.45rem; font-weight: 800; color: var(--aqua); margin-bottom: 6px; }",
    "    .welcome-box p { font-size: 0.92rem; color: var(--text-sec); line-height: 1.55; }",
    "    .stats-row {",
    "      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));",
    "      gap: 16px; margin-bottom: 22px;",
    "    }",
    "    .stat-card {",
    "      background: var(--glass); backdrop-filter: blur(16px);",
    "      border: 1px solid var(--border); border-radius: 18px;",
    "      padding: 18px 20px; box-shadow: var(--shadow);",
    "    }",
    "    .stat-label { font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-sec); margin-bottom: 6px; }",
    '    .stat-val { font-size: 1.8rem; font-weight: 800; font-family: "JetBrains Mono", monospace; color: var(--mint); }',
    "    .grid-main {",
    "      display: grid; grid-template-columns: 1.3fr 0.9fr;",
    "      gap: 22px; align-items: start; margin-bottom: 22px;",
    "    }",
    "    @media (max-width: 940px) { .grid-main { grid-template-columns: 1fr; } }",
    "    .section-heading { font-size: 1.1rem; font-weight: 700; color: var(--mint); margin-bottom: 16px; }",
    "    .form-grid {",
    "      display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));",
    "      gap: 16px; margin-bottom: 20px;",
    "    }",
    "    .field { display: flex; flex-direction: column; gap: 6px; }",
    '    .field label { font-size: 0.84rem; font-weight: 600; font-family: "JetBrains Mono", monospace; color: var(--text); }',
    "    .field small { font-size: 0.74rem; color: var(--text-sec); }",
    "    .control {",
    "      width: 100%; padding: 11px 13px; border-radius: 12px;",
    "      border: 1px solid var(--border); background: var(--input-bg);",
    '      color: var(--text); font-family: "JetBrains Mono", monospace; font-size: 0.86rem;',
    "      outline: none; transition: border-color 0.2s ease, box-shadow 0.2s ease;",
    "    }",
    "    .control:focus {",
    "      border-color: var(--aqua);",
    "      box-shadow: 0 0 0 3px rgba(152, 232, 222, 0.24), 0 0 14px rgba(184, 255, 243, 0.25);",
    "    }",
    "    .btn-group { display: flex; flex-wrap: wrap; gap: 10px; }",
    "    .result-card-glow {",
    "      border: 1px solid var(--mint);",
    "      box-shadow: 0 18px 44px rgba(10, 15, 25, 0.7), 0 0 28px rgba(152, 232, 222, 0.28);",
    "      min-height: 320px;",
    "    }",
    "    .result-panel { transition: opacity 0.3s ease, transform 0.3s ease; }",
    "    .result-badge {",
    "      display: inline-block; padding: 8px 16px; border-radius: 999px;",
    "      background: rgba(152, 232, 222, 0.16); border: 1px solid var(--aqua);",
    '      color: var(--aqua); font-weight: 800; font-size: 1.05rem; font-family: "JetBrains Mono", monospace;',
    "      box-shadow: 0 0 18px rgba(152, 232, 222, 0.3); margin-bottom: 14px;",
    "    }",
    "    .meta-line { font-size: 0.86rem; color: var(--text-sec); margin-bottom: 14px; }",
    "    .bar-wrap { margin-bottom: 12px; }",
    '    .bar-head { display: flex; justify-content: space-between; font-size: 0.8rem; font-family: "JetBrains Mono", monospace; margin-bottom: 5px; }',
    "    .bar-track { height: 10px; border-radius: 999px; background: rgba(152, 232, 222, 0.15); overflow: hidden; }",
    "    .bar-fill {",
    "      height: 100%; width: 0%; border-radius: 999px;",
    "      background: linear-gradient(90deg, var(--muted-mint), var(--aqua));",
    "      transition: width 0.65s cubic-bezier(0.22, 1, 0.36, 1);",
    "    }",
    "    .error-banner {",
    "      display: none; margin-top: 12px; padding: 11px 14px; border-radius: 12px;",
    "      background: rgba(244, 63, 94, 0.16); border: 1px solid rgba(251, 113, 133, 0.45);",
    "      color: #fecdd3; font-size: 0.84rem; font-weight: 500;",
    "    }",
    '    html[data-theme="light"] .error-banner { color: #9f1239; }',
    "    .error-banner.show { display: block; }",
    "    .loading-veil {",
    "      position: fixed; inset: 0; z-index: 50; display: none;",
    "      align-items: center; justify-content: center; flex-direction: column; gap: 14px;",
    "      background: rgba(16, 24, 39, 0.72); backdrop-filter: blur(8px);",
    "    }",
    "    .loading-veil.show { display: flex; }",
    "    .spinner-ring {",
    "      width: 54px; height: 54px; border-radius: 50%;",
    "      border: 4px solid rgba(152, 232, 222, 0.22);",
    "      border-top-color: var(--aqua);",
    "      animation: spin 0.9s linear infinite;",
    "      box-shadow: 0 0 20px rgba(152, 232, 222, 0.35);",
    "    }",
    "@keyframes spin { to { transform: rotate(360deg); } }",
    "    .table-box { overflow-x: auto; border-radius: 14px; border: 1px solid var(--border); background: var(--input-bg); }",
    "    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }",
    '    th, td { padding: 11px 14px; text-align: left; border-bottom: 1px solid rgba(152, 232, 222, 0.12); font-family: "JetBrains Mono", monospace; }',
    "    th { font-size: 0.74rem; text-transform: uppercase; color: var(--mint); background: rgba(152, 232, 222, 0.08); }",
    "    tr:last-child td { border-bottom: none; }",
    "  </style>",
    "</head>",
    "<body>",
    '  <canvas id="particles-canvas"></canvas>',
    '  <div class="floating-shapes" aria-hidden="true">',
    '    <div class="shape s1"></div><div class="shape s2"></div><div class="shape s3"></div>',
    "  </div>",
    '  <div class="loading-veil" id="loadingVeil" aria-hidden="true">',
    '    <div class="spinner-ring"></div>',
    '    <div style="font-weight: 600; color: var(--aqua);">Predicting with SVM...</div>',
    "  </div>",
    '  <div class="shell">',
    '    <header class="top-header">',
    '      <div class="brand">',
    '        <div class="logo-box" aria-hidden="true">',
    '          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">',
    '            <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"></polygon>',
    '            <circle cx="12" cy="12" r="3"></circle>',
    "          </svg>",
    "        </div>",
    "        <div>",
    '          <div class="app-title">MintTurquoise SVM Studio</div>',
    '          <div class="tagline">Support Vector Machine Classification &bull; svm.pkl</div>',
    "        </div>",
    "      </div>",
    '      <div class="header-controls">',
    '        <button type="button" class="btn" id="modeBtn">Mode: Category</button>',
    '        <button type="button" class="btn" id="themeBtn"><span id="themeIcon">☀️</span><span id="themeText">Light</span></button>',
    "      </div>",
    "    </header>",
    '    <section class="glass-card welcome-box">',
    "      <h2>Welcome to Your SVM Inference Dashboard</h2>",
    "      <p>Run real-time predictions using your trained <strong>SVC</strong> model (<code>svm.pkl</code>) across the 5 exact features (<code>Age</code>, <code>Gender</code>, <code>Region</code>, <code>Occupation</code>, <code>Income</code>) with session analytics and CSV export.</p>",
    "    </section>",
    '    <section class="stats-row">',
    '      <div class="stat-card">',
    '        <div class="stat-label">Total Predictions</div>',
    '        <div class="stat-val" id="totalStat">{{ total_predictions }}</div>',
    "      </div>",
    "      {% for cls, cnt in class_counts.items() %}",
    '      <div class="stat-card">',
    '        <div class="stat-label">Class "{{ cls }}" Count</div>',
    '        <div class="stat-val cls-stat" data-cls="{{ cls }}">{{ cnt }}</div>',
    "      </div>",
    "      {% endfor %}",
    "    </section>",
    '    <div class="grid-main">',
    '      <section class="glass-card">',
    '        <div class="section-heading">Model Input Form</div>',
    '        <form id="svmForm" novalidate>',
    '          <div class="form-grid">',
    "            {% for feat in features %}",
    "            {% set opts = categorical_options.get(feat) %}",
    '            <div class="field">',
    '              <label for="{{ feat }}">{{ loop.index }}. {{ feat }}</label>',
    "              {% if opts %}",
    '              <select class="control" id="{{ feat }}" name="{{ feat }}">',
    '                <option value="">Select {{ feat }}</option>',
    "                {% for o in opts %}",
    '                <option value="{{ o.value }}">{{ o.label }}</option>',
    "                {% endfor %}",
    "              </select>",
    '              <input class="control" type="number" step="any" id="{{ feat }}_num" placeholder="Numeric code" style="display:none;" />',
    "              {% else %}",
    '              <input class="control" type="number" step="any" id="{{ feat }}" name="{{ feat }}" placeholder="Enter {{ feat }}" required />',
    "              {% endif %}",
    "              <small>{{ descriptions.get(feat, \"Numeric input\") }}</small>",
    "            </div>",
    "            {% endfor %}",
    "          </div>",
    '          <div class="btn-group">',
    '            <button type="submit" class="btn btn-primary">Predict Class</button>',
    '            <button type="button" class="btn" id="resetBtn">Reset</button>',
    "          </div>",
    '          <div class="error-banner" id="errorBanner" role="alert"></div>',
    "        </form>",
    "      </section>",
    '      <aside class="glass-card result-card-glow">',
    '        <div class="section-heading">Prediction Result</div>',
    '        <div class="result-panel" id="resultPanel">',
    '          <div id="emptyResult" style="color: var(--text-sec); font-size: 0.88rem; padding: 28px 0;">',
    "            Submit valid feature values to inspect the predicted SVM class.",
    "          </div>",
    '          <div id="activeResult" style="display: none;">',
    '            <div class="result-badge" id="predBadge">Predicted: —</div>',
    '            <div class="meta-line" id="confLine">Confidence: Not available (SVC probability=False)</div>',
    '            <div id="probBars"></div>',
    "          </div>",
    "        </div>",
    "      </aside>",
    "    </div>",
    '    <section class="glass-card">',
    '      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 14px;">',
    '        <div class="section-heading" style="margin-bottom: 0;">Session Prediction History</div>',
    '        <div class="btn-group">',
    '          <a href="/download-csv" class="btn">Download CSV</a>',
    '          <button type="button" class="btn" id="clearHistoryBtn">Clear History</button>',
    "        </div>",
    "      </div>",
    '      <div class="table-box">',
    "        <table>",
    "          <thead><tr><th>Time</th><th>Prediction</th><th>Confidence</th><th>Features</th></tr></thead>",
    '          <tbody id="historyBody">',
    "            {% if history %}",
    "              {% for item in history|reverse %}",
    "              <tr>",
    "                <td>{{ item.timestamp }}</td>",
    "                <td>{{ item.prediction }}</td>",
    "                <td>{% if item.confidence is not none %}{{ \"%.2f%%\"|format(item.confidence * 100) }}{% else %}N/A{% endif %}</td>",
    "                <td>{{ item.summary }}</td>",
    "              </tr>",
    "              {% endfor %}",
    "            {% else %}",
    '              <tr id="noHistoryRow"><td colspan="4" style="text-align: center; color: var(--text-sec); padding: 18px;">No predictions yet in this session.</td></tr>',
    "            {% endif %}",
    "          </tbody>",
    "        </table>",
    "      </div>",
    "    </section>",
    "  </div>",
    "  <script>",
    "    const FEATURE_ORDER = {{ features | tojson }};",
    "    const CAT_KEYS = {{ cat_keys | tojson }};",
    "    const htmlRoot = document.documentElement;",
    "    const themeBtn = document.getElementById('themeBtn');",
    "    const themeIcon = document.getElementById('themeIcon');",
    "    const themeText = document.getElementById('themeText');",
    "    function setTheme(t) {",
    "      htmlRoot.setAttribute('data-theme', t);",
    "      localStorage.setItem('svm_theme', t);",
    "      themeIcon.textContent = t === 'light' ? '🌙' : '☀️';",
    "      themeText.textContent = t === 'light' ? 'Dark' : 'Light';",
    "    }",
    "    setTheme(localStorage.getItem('svm_theme') || 'dark');",
    "    themeBtn.addEventListener('click', () => {",
    "      setTheme(htmlRoot.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');",
    "    });",
    "    let numericMode = false;",
    "    const modeBtn = document.getElementById('modeBtn');",
    "    modeBtn.addEventListener('click', () => {",
    "      numericMode = !numericMode;",
    "      modeBtn.textContent = numericMode ? 'Mode: Numeric Code' : 'Mode: Category';",
    "      CAT_KEYS.forEach((k) => {",
    "        const sel = document.getElementById(k);",
    "        const num = document.getElementById(k + '_num');",
    "        if (!sel || !num) return;",
    "        if (numericMode) {",
    "          if (sel.value !== '') num.value = sel.value;",
    "          sel.style.display = 'none';",
    "          num.style.display = 'block';",
    "        } else {",
    "          if (num.value !== '') {",
    "            const found = Array.from(sel.options).find((o) => o.value === String(Number(num.value)));",
    "            sel.value = found ? found.value : '';",
    "          }",
    "          num.style.display = 'none';",
    "          sel.style.display = 'block';",
    "        }",
    "      });",
    "    });",
    "    function animateCount(el, target) {",
    "      const start = Number(el.textContent) || 0;",
    "      const end = Number(target) || 0;",
    "      if (start === end) { el.textContent = end; return; }",
    "      const t0 = performance.now();",
    "      function tick(now) {",
    "        const p = Math.min((now - t0) / 400, 1);",
    "        el.textContent = Math.round(start + (end - start) * (1 - Math.pow(1 - p, 3)));",
    "        if (p < 1) requestAnimationFrame(tick);",
    "      }",
    "      requestAnimationFrame(tick);",
    "    }",
    "    const errorBanner = document.getElementById('errorBanner');",
    "    function showError(msg) { errorBanner.textContent = msg; errorBanner.classList.add('show'); }",
    "    function hideError() { errorBanner.textContent = ''; errorBanner.classList.remove('show'); }",
    "    const loadingVeil = document.getElementById('loadingVeil');",
    "    const svmForm = document.getElementById('svmForm');",
    "    document.getElementById('resetBtn').addEventListener('click', () => {",
    "      svmForm.reset();",
    "      CAT_KEYS.forEach((k) => { const n = document.getElementById(k + '_num'); if (n) n.value = ''; });",
    "      hideError();",
    "    });",
    "    function renderPrediction(res) {",
    "      const panel = document.getElementById('resultPanel');",
    "      panel.style.opacity = '0.3';",
    "      panel.style.transform = 'translateY(6px)';",
    "      setTimeout(() => {",
    "        document.getElementById('emptyResult').style.display = 'none';",
    "        document.getElementById('activeResult').style.display = 'block';",
    "        document.getElementById('predBadge').textContent = 'Predicted Class: ' + res.prediction;",
    "        const confEl = document.getElementById('confLine');",
    "        const barsEl = document.getElementById('probBars');",
    "        barsEl.innerHTML = '';",
    "        if (res.confidence !== null && res.confidence !== undefined) {",
    "          confEl.textContent = 'Confidence: ' + (res.confidence * 100).toFixed(2) + '%';",
    "        } else {",
    "          confEl.textContent = 'Confidence: Not available for this SVM configuration';",
    "        }",
    "        if (Array.isArray(res.probabilities)) {",
    "          res.probabilities.forEach((item) => {",
    "            const pct = (item.probability * 100).toFixed(2);",
    "            const div = document.createElement('div');",
    "            div.className = 'bar-wrap';",
    "            div.innerHTML = '<div class=\"bar-head\"><span>Class ' + item.class + '</span><span>' + pct + '%</span></div>' +",
    "              '<div class=\"bar-track\"><div class=\"bar-fill\" style=\"width:0%\"></div></div>';",
    "            barsEl.appendChild(div);",
    "            requestAnimationFrame(() => {",
    "              div.querySelector('.bar-fill').style.width = Math.max(0, Math.min(100, item.probability * 100)) + '%';",
    "            });",
    "          });",
    "        }",
    "        panel.style.opacity = '1';",
    "        panel.style.transform = 'translateY(0)';",
    "      }, 120);",
    "      animateCount(document.getElementById('totalStat'), res.total_predictions);",
    "      if (res.class_counts) {",
    "        document.querySelectorAll('.cls-stat').forEach((el) => {",
    "          const key = el.getAttribute('data-cls');",
    "          if (key in res.class_counts) animateCount(el, res.class_counts[key]);",
    "        });",
    "      }",
    "      const tbody = document.getElementById('historyBody');",
    "      const emptyRow = document.getElementById('noHistoryRow');",
    "      if (emptyRow) emptyRow.remove();",
    "      const tr = document.createElement('tr');",
    "      const confText = (res.entry.confidence !== null && res.entry.confidence !== undefined) ? (res.entry.confidence * 100).toFixed(2) + '%' : 'N/A';",
    "      tr.innerHTML = '<td>' + res.entry.timestamp + '</td><td>' + res.entry.prediction + '</td><td>' + confText + '</td><td>' + res.entry.summary + '</td>';",
    "      tbody.prepend(tr);",
    "    }",
    "    svmForm.addEventListener('submit', async (e) => {",
    "      e.preventDefault();",
    "      hideError();",
    "      const payload = {};",
    "      for (const feat of FEATURE_ORDER) {",
    "        const el = (CAT_KEYS.includes(feat) && numericMode) ? document.getElementById(feat + '_num') : document.getElementById(feat);",
    "        const raw = el ? el.value.trim() : '';",
    "        if (raw === '' || Number.isNaN(Number(raw)) || !Number.isFinite(Number(raw))) {",
    "          showError('Please enter a valid number or category for ' + feat + '.');",
    "          if (el) el.focus();",
    "          return;",
    "        }",
    "        payload[feat] = Number(raw);",
    "      }",
    "      loadingVeil.classList.add('show');",
    "      try {",
    "        const resp = await fetch('/predict', {",
    "          method: 'POST',",
    "          headers: { 'Content-Type': 'application/json' },",
    "          body: JSON.stringify(payload),",
    "        });",
    "        const data = await resp.json();",
    "        if (!resp.ok || !data.success) { showError(data.error || 'Prediction failed.'); return; }",
    "        renderPrediction(data);",
    "      } catch (err) {",
    "        showError('Network error while contacting the prediction server.');",
    "      } finally {",
    "        loadingVeil.classList.remove('show');",
    "      }",
    "    });",
    "    document.getElementById('clearHistoryBtn').addEventListener('click', async () => {",
    "      try {",
    "        const resp = await fetch('/clear-history', { method: 'POST' });",
    "        const data = await resp.json();",
    "        if (resp.ok && data.success) {",
    "          document.getElementById('historyBody').innerHTML = '<tr id=\"noHistoryRow\"><td colspan=\"4\" style=\"text-align: center; color: var(--text-sec); padding: 18px;\">No predictions yet in this session.</td></tr>';",
    "          animateCount(document.getElementById('totalStat'), 0);",
    "          document.querySelectorAll('.cls-stat').forEach((el) => animateCount(el, 0));",
    "        }",
    "      } catch (err) { showError('Failed to clear session history.'); }",
    "    });",
    "    const cvs = document.getElementById('particles-canvas');",
    "    const ctx = cvs.getContext('2d');",
    "    let pts = [];",
    "    function sizeCvs() { cvs.width = window.innerWidth; cvs.height = window.innerHeight; }",
    "    function seedPts() {",
    "      const n = Math.min(40, Math.floor((window.innerWidth * window.innerHeight) / 36000));",
    "      pts = Array.from({ length: n }, () => ({",
    "        x: Math.random() * cvs.width, y: Math.random() * cvs.height,",
    "        r: Math.random() * 2.2 + 0.8, vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,",
    "      }));",
    "    }",
    "    function stepPts() {",
    "      ctx.clearRect(0, 0, cvs.width, cvs.height);",
    "      ctx.fillStyle = htmlRoot.getAttribute('data-theme') === 'light' ? 'rgba(20, 150, 137, 0.35)' : 'rgba(152, 232, 222, 0.42)';",
    "      for (const p of pts) {",
    "        p.x += p.vx; p.y += p.vy;",
    "        if (p.x < 0) p.x = cvs.width; if (p.x > cvs.width) p.x = 0;",
    "        if (p.y < 0) p.y = cvs.height; if (p.y > cvs.height) p.y = 0;",
    "        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();",
    "      }",
    "      requestAnimationFrame(stepPts);",
    "    }",
    "    sizeCvs(); seedPts(); stepPts();",
    "    window.addEventListener('resize', () => { sizeCvs(); seedPts(); });",
    "  </script>",
    "</body>",
    "</html>",
]

HTML_TEMPLATE = "\n".join(HTML_LINES)


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template_string(
        HTML_TEMPLATE,
        features=FEATURE_ORDER,
        descriptions=FEATURE_DESCRIPTIONS,
        categorical_options=CATEGORICAL_OPTIONS,
        cat_keys=list(CATEGORICAL_OPTIONS.keys()),
        total_predictions=len(SESSION_HISTORY),
        class_counts=CLASS_COUNTS,
        history=SESSION_HISTORY,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"success": False, "error": "Invalid JSON payload."}), 400

        X = build_feature_vector(payload)
        raw_pred = MODEL.predict(X)[0]
        pred_label = str(raw_pred)

        probabilities_data = None
        confidence = None

        if SUPPORTS_PROBA:
            probs = MODEL.predict_proba(X)[0]
            probabilities_data = [
                {"class": str(c), "probability": float(p)}
                for c, p in zip(CLASSES, probs)
            ]
            confidence = float(np.max(probs))

        CLASS_COUNTS[pred_label] = CLASS_COUNTS.get(pred_label, 0) + 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        summary = ", ".join(f"{k}={payload[k]}" for k in FEATURE_ORDER)

        entry = {
            "timestamp": timestamp,
            "prediction": pred_label,
            "confidence": confidence,
            "summary": summary,
            "inputs": {k: payload[k] for k in FEATURE_ORDER},
        }
        SESSION_HISTORY.append(entry)

        return jsonify(
            {
                "success": True,
                "prediction": pred_label,
                "confidence": confidence,
                "probabilities": probabilities_data,
                "total_predictions": len(SESSION_HISTORY),
                "class_counts": CLASS_COUNTS,
                "entry": entry,
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Prediction error: {exc}"}), 500


@app.route("/clear-history", methods=["POST"])
def clear_history():
    SESSION_HISTORY.clear()
    for key in CLASS_COUNTS:
        CLASS_COUNTS[key] = 0
    return jsonify({"success": True})


@app.route("/download-csv", methods=["GET"])
def download_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "prediction", "confidence", *FEATURE_ORDER])
    for row in SESSION_HISTORY:
        inputs = row.get("inputs", {})
        writer.writerow(
            [
                row.get("timestamp", ""),
                row.get("prediction", ""),
                f"{row['confidence']:.6f}" if row.get("confidence") is not None else "",
                *[inputs.get(feat, "") for feat in FEATURE_ORDER],
            ]
        )
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=svm_predictions.csv"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
