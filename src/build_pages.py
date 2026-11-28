"""Build docs/: a static GitHub Pages site that runs app.py fully in the browser with stlite
(Streamlit on Pyodide/WebAssembly; no server). Run after src/run_all.py:  python src/build_pages.py"""
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STLITE = "1.12.2"

if DOCS.exists():
    shutil.rmtree(DOCS)
(DOCS / "data" / "processed").mkdir(parents=True)
(DOCS / "data" / "raw").mkdir(parents=True)
(DOCS / "outputs").mkdir()
(DOCS / "images").mkdir()

files = {}


def add(rel, src=None):
    dst = DOCS / rel
    if src is not None:
        shutil.copy(src, dst)
    files[rel] = {"url": "./" + rel}


add("app.py", ROOT / "app.py")
for name in ["panel_iteration2", "p3_iter1_oof", "p3_iter2_oof", "p3_forecast_holdout"]:
    df = pd.read_parquet(ROOT / "data" / "processed" / f"{name}.parquet")
    df.to_csv(DOCS / "data" / "processed" / f"{name}.csv", index=False, float_format="%.5g")
    add(f"data/processed/{name}.csv")
# raw panel: only the countries the feedback-loop tab can inspect
arts = pd.read_csv(ROOT / "outputs" / "p1_artifact_log.csv")
raw = pd.read_csv(ROOT / "data" / "raw" / "master_dataset.csv")
raw[raw.entity.isin(arts.entity.unique())].to_csv(DOCS / "data" / "raw" / "master_dataset.csv", index=False)
add("data/raw/master_dataset.csv")
for f in ["p3_cv_results.csv", "p3_regional_rmse_same_rows.csv", "p3_psi_drift.csv", "p1_artifact_log.csv",
          "iter1_assignments.csv", "iter2_assignments.csv", "p1_logistic_fits.csv", "p1_spearman_by_year.csv",
          "p3_model_metrics.json", "p1_sigma_convergence.csv", "p1_beta_convergence.csv", "p1_coverage_by_year.csv",
          "iter2_tsne_embedding.csv", "p3_interval_calibration.csv"]:
    add(f"outputs/{f}", ROOT / "outputs" / f)
for f in ["V0_workflow_architecture.png", "V0b_system_architecture.png", "V8_trajectory_archetypes.png", "V14_synthesis_answer.png"]:
    add(f"images/{f}", ROOT / "images" / f)

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
<title>From Snapshots to Trajectories</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE}/build/stlite.css" />
<style>
  body {{ margin: 0; background: #ffffff; font-family: "Segoe UI", system-ui, sans-serif; }}
  #loading {{ position: fixed; inset: 0; display: flex; flex-direction: column; align-items: center;
             justify-content: center; color: #1F2328; gap: 10px; }}
  #loading small {{ color: #6E7781; }}
</style>
</head>
<body>
<div id="loading"><b>Loading the A3 visual analytics demo…</b>
  <small>Python runs in your browser (Pyodide). First load takes 20-60 s.</small></div>
<div id="root"></div>
<script type="module">
import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE}/build/stlite.js";
const files = {json.dumps(files, indent=2)};
mount({{
  requirements: ["plotly"],
  entrypoint: "app.py",
  files,
  streamlitConfig: {{
    "theme.base": "light", "theme.primaryColor": "#2A78D6", "theme.backgroundColor": "#FFFFFF",
    "theme.secondaryBackgroundColor": "#F6F8FA", "theme.textColor": "#1F2328",
    "client.toolbarMode": "viewer"
  }}
}}, document.getElementById("root"));
const obs = new MutationObserver(() => {{
  if (document.querySelector('[data-testid="stAppViewContainer"]')) {{
    document.getElementById("loading").remove(); obs.disconnect();
  }}
}});
obs.observe(document.body, {{ childList: true, subtree: true }});
</script>
</body>
</html>
"""
(DOCS / "index.html").write_text(html, encoding="utf8")
(DOCS / ".nojekyll").write_text("")
size = sum(p.stat().st_size for p in DOCS.rglob("*") if p.is_file())
print(f"docs/ built: {len(files)} files, {size / 1e6:.1f} MB")
