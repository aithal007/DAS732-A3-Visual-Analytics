"""Regenerate the workflow diagrams and compile every TikZ diagram (report/diagrams/*.tex) to vector PDF
for the report and to 300-dpi PNG in images/ for the deck and the app.  Needs pdflatex and pdftoppm.
    python src/build_diagrams.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "report" / "diagrams"

subprocess.run([sys.executable, str(ROOT / "src" / "build_workflow_diagrams.py")], check=True)
for tex in sorted(DIAG.glob("*.tex")):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=DIAG, check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", tex.with_suffix(".pdf").name,
                    str(ROOT / "images" / tex.stem)], cwd=DIAG, check=True)
    for ext in (".aux", ".log"):
        tex.with_suffix(ext).unlink(missing_ok=True)
    print("built", tex.stem)
