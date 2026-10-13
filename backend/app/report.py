"""Printable HTML report (analyze + explain) with the mandatory disclaimer."""

from __future__ import annotations

from html import escape

DISCLAIMER = ("Decision support and educational use only. Not a substitute for formal diagnostic imaging "
              "(e.g., coronary angiography) or clinical judgment.")

STATUS = {"likely": "likely (90% conformal set = stenosis, agrees with threshold)",
          "uncertain": "uncertain (90% conformal set has both outcomes or disagrees with the threshold, "
                       "or the patient is out of distribution / many inputs were imputed)",
          "unlikely": "unlikely (90% conformal set = no stenosis, agrees with threshold)"}
LOCALISATION_NOTE = ("Vessel probabilities mostly reflect overall CAD risk: among patients with CAD the model separates "
                     "LAD only modestly and LCX/RCA barely better than chance (within-CAD AUC 0.69 / 0.52 / 0.57).")


def _pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def render(features: dict, analysis: dict, explanation: dict, fmap: dict) -> str:
    rows = []
    for name, r in [("CAD (any vessel)", analysis["cad"])] + [(v, r) for v, r in analysis["vessels"].items()]:
        lo, hi = r["interval"]
        rows.append(f"<tr><td>{escape(name)}</td><td>{_pct(r['p'])}</td><td>{_pct(lo)}–{_pct(hi)}</td>"
                    f"<td>{escape(STATUS[r['status']])}</td><td>{_pct(r['threshold'])}</td>"
                    f"<td>{'above' if r['positive'] else 'below'}</td></tr>")
    mv = analysis["multivessel"]
    app = analysis["applicability"]
    feat_rows = "".join(
        f"<tr><td>{escape(fmap[k]['label'])}</td><td>{escape(str(v))} {escape(fmap[k].get('unit') or '')}</td>"
        f"<td>{'imputed' if k in analysis['imputed'] else ''}</td></tr>"
        for k, v in features.items() if k in fmap)
    narr = "".join(f"<h3>{escape(t)}</h3><ul>" + "".join(f"<li>{escape(s)}</li>" for s in e["narrative"]) + "</ul>"
                   for t, e in explanation["targets"].items())
    pat = "".join(f"<li>{escape(s)}</li>" for s in explanation["targets"]["CAD"]["narrative_patient"])
    warn = "" if app["in_domain"] else (
        "<p class='warn'>Warning: this input is outside the applicability domain of the training data "
        f"(score {app['score']:.2f} &gt; {app['threshold']:.2f}); estimates are unreliable.</p>")
    oor = "".join(f"<li>{escape(o['feature'])} = {o['value']:g} (training range {o['range'][0]:g}–{o['range'][1]:g})</li>"
                  for o in analysis["out_of_range"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoronaryLens report</title>
<style>
body {{ font-family: system-ui, sans-serif; color: #111; background: #fff; max-width: 860px; margin: 24px auto;
       padding: 0 16px; }}
.disclaimer {{ border: 2px solid #b45309; background: #fff7ed; padding: 10px 14px; font-weight: 600; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-size: 14px; }}
td, th {{ border-bottom: 1px solid #ddd; padding: 4px 6px; text-align: left; }}
.warn {{ color: #b91c1c; font-weight: 600; }}
@media print {{ .disclaimer {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }} }}
</style></head><body>
<p class="disclaimer">{escape(DISCLAIMER)}</p>
<h1>CoronaryLens risk report</h1>
<p>Model {escape(analysis['model_version'])} · threshold mode: {escape(analysis['mode'])}</p>
{warn}
<h2>Estimated probabilities (marginals of one joint vessel-pattern distribution)</h2>
<table><tr><th>Target</th><th>P</th><th>90% interval</th><th>Conformal status</th><th>Threshold</th><th></th></tr>
{''.join(rows)}</table>
<p>Multi-vessel (≥2): {_pct(mv['p_ge2'])} · three-vessel: {_pct(mv['p_3vd'])}</p>
<p><em>{escape(LOCALISATION_NOTE)}</em></p>
<h2>Explanation (SHAP/Owen values, probability scale)</h2>{narr}
<h2>Plain-language summary</h2><ul>{pat}</ul>
<h2>Inputs</h2><table><tr><th>Feature</th><th>Value</th><th></th></tr>{feat_rows}</table>
{('<h3>Values outside the training range</h3><ul>' + oor + '</ul>') if oor else ''}
<p class="disclaimer">{escape(DISCLAIMER)}</p>
</body></html>
"""
