"""docs/REPORT.md -> docs/REPORT.pdf (A4) via Markdown -> HTML -> headless Chromium; prints the page count.

Usage: uv run python tools/build_report.py   (needs `npx playwright install chromium` once, done by the e2e setup)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
MD, HTML, PDF = ROOT / "docs/REPORT.md", ROOT / "docs/REPORT.html", ROOT / "docs/REPORT.pdf"
MAX_PAGES = 6

CSS = """
@page { size: A4; margin: 13mm 13mm 13mm 13mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 8.6pt; line-height: 1.33; color: #111; }
h1 { font-size: 14pt; margin: 0 0 2pt; } h2 { font-size: 10.5pt; margin: 9pt 0 3pt; border-bottom: 1px solid #ccc; }
p, li { margin: 2.5pt 0; } ul, ol { padding-left: 15pt; margin: 2pt 0; }
blockquote { margin: 4pt 0; padding: 3pt 8pt; background: #fff4d6; border-left: 3px solid #d9a400; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0; font-size: 7.8pt; }
th, td { border: 1px solid #ccc; padding: 1.5pt 4pt; text-align: left; } th { background: #f2f2f2; }
img { display: block; margin: 4pt auto; max-width: 100%; }
img[alt^="CoronaryLens"] { max-height: 290px; } img[alt^="Share"] { max-height: 145px; }
img[alt^="What-if"] { max-height: 430px; } img[alt^="Reliability"] { max-height: 230px; }
code, pre { font-size: 7.6pt; } pre { background: #f6f6f6; padding: 4pt; margin: 3pt 0; }
"""

JS = """
const { chromium } = require('@playwright/test');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + process.argv[2], { waitUntil: 'networkidle' });
  await p.pdf({ path: process.argv[3], format: 'A4', printBackground: true, preferCSSPageSize: true });
  await b.close();
})();
"""


def main() -> int:
    body = markdown.markdown(MD.read_text("utf-8"), extensions=["tables", "fenced_code"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>CoronaryLens report</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    js = ROOT / "frontend" / ".report_pdf.cjs"
    js.write_text(JS, encoding="utf-8")
    try:
        subprocess.run(["node", str(js), str(HTML), str(PDF)], cwd=ROOT / "frontend", check=True)
    finally:
        js.unlink(missing_ok=True)
    n = len(PdfReader(str(PDF)).pages)
    print(f"{PDF.relative_to(ROOT)}: {n} pages (limit {MAX_PAGES})")
    return 0 if n <= MAX_PAGES else 1


if __name__ == "__main__":
    sys.exit(main())
