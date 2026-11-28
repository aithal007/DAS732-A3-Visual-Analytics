"""Compile the TikZ architecture diagrams (report/diagrams/*.tex) to vector PDF for the report and to
300-dpi PNG in images/ for the deck and the app.  Needs pdflatex and pdftoppm (MiKTeX / TeX Live).
    python src/build_diagrams.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "report" / "diagrams"

for tex in sorted(DIAG.glob("*.tex")):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=DIAG, check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", tex.with_suffix(".pdf").name,
                    str(ROOT / "images" / tex.stem)], cwd=DIAG, check=True)
    for ext in (".aux", ".log"):
        tex.with_suffix(ext).unlink(missing_ok=True)
    print("built", tex.stem)
