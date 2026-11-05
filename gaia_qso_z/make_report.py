"""Build REPORT.pdf: what was done, results, prior work, caveats, and full source listings."""
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent
FIG, DATA = ROOT / "figures", ROOT / "data"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSerif.ttf"),  # no DejaVu Sans Oblique installed: serif marks emphasis
                ("DVMono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.5, leading=13.2, spaceAfter=5)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="DV-B", fontSize=14, spaceBefore=10, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="DV-B", fontSize=11, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", parent=ss["Title"], fontName="DV-B", fontSize=18, leading=22)
sub = ParagraphStyle("sub", parent=body, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))
cap = ParagraphStyle("cap", parent=body, fontSize=8.3, leading=11, textColor=colors.HexColor("#333333"), spaceAfter=10)
code = ParagraphStyle("code", fontName="DVMono", fontSize=6.3, leading=7.6)
cell = ParagraphStyle("cell", parent=body, fontSize=7.4, leading=9, spaceAfter=0)

W = A4[0] - 3.6 * cm
story = []


def P(text, style=body):
    story.append(Paragraph(text, style))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, bullet, bulletText="•"))


def fig(name, caption, width=W):
    path = FIG / name
    iw, ih = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width, height=width * ih / iw)
    story.append(KeepTogether([img, Paragraph(caption, cap)]))


# ------------------------------------------------------------------ title + summary
story.append(Paragraph("Quasar redshifts from Gaia DR3 XP coefficients", title))
P(f"Project report · {date.today():%d %B %Y} · prepared with Claude Code", sub)
story.append(Spacer(1, 10))

story.append(Paragraph("Summary", h1))
P("<b>Question.</b> Do the coefficients of Gaia DR3 BP/RP (XP) spectra carry information on quasar "
  "redshift? SDSS DR16Q spectroscopic redshifts were matched to Gaia DR3 sources with published XP "
  "coefficients, and redshift was predicted from the coefficients with cross-validated models.")
bullets([
    "<b>Yes, strongly.</b> On 36,402 quasars, a kNN on the first 20 BP + 20 RP coefficients reaches "
    "σ<sub>NMAD</sub> = 0.0040 in Δz/(1+z) with 88% of objects within 0.01(1+z); an MLP reaches 0.50% "
    "catastrophic outliers (|Δz|/(1+z) &gt; 0.2). Gaia colours alone give σ<sub>NMAD</sub> ≈ 0.18 and ≈ 29% outliers.",
    "The signal is visible directly: stacked spectra show Lyα, CIV, CIII], MgII, Hβ+[OIII] and Hα moving "
    "along λ<sub>rest</sub>(1+z).",
    "Gaia's own QSOC redshifts (template χ² fit) remain the most precise (σ<sub>NMAD</sub> = 0.0030) but have more "
    "catastrophic outliers (1.4%) than the best MLP (0.50%).",
    "<b>Not novel.</b> Refereed work already estimates quasar redshifts from XP spectra with SDSS-trained "
    "models, including directly from the Hermite coefficients (Sect. 6). The study was therefore not "
    "developed into a paper.",
])

# ------------------------------------------------------------------ approach
story.append(Paragraph("1  Approach", h1))
P("Quasar broad emission lines cross the 330–1050 nm XP window as redshift changes. Even at XP's very low "
  "resolution, their positions and the continuum shape should be encoded in the 55+55 Hermite "
  "coefficients. The test has two parts: (i) <i>see</i> the signal in the data, and (ii) <i>measure</i> how much "
  "redshift information the coefficients carry beyond broadband Gaia colours, against Gaia's QSOC as "
  "an external benchmark and a shuffled-redshift control.")

# ------------------------------------------------------------------ data
story.append(Paragraph("2  Data", h1))
story.append(Paragraph("2.1  Cross-match (01_crossmatch.py)", h2))
P("SDSS DR16Q (750,414 quasars) lists a Gaia DR2 counterpart for 469,774 of them. DR2 ids of the "
  "301,346 counterparts with G<sub>DR2</sub> &lt; 20.5 were uploaded to the Gaia archive (ESA TAP), mapped to "
  "DR3 with <font face='DVMono'>gaiadr3.dr2_neighbourhood</font>, joined to <font face='DVMono'>gaia_source</font>, "
  "and the DR3 neighbour closest to the SDSS position was kept (&lt; 1″; median separation 0.048″). "
  "QSOC redshifts come from <font face='DVMono'>gaiadr3.qso_candidates</font>.")
P("A direct 1″ positional join was tried first, but the archive aborts anonymous jobs after 60 s, which "
  "made it impractically slow. It did complete for the first 190,000 DR16Q rows, which serves as a "
  "completeness check: the id-based route recovers 6,392 of the 6,395 XP sources found positionally "
  "(99.95%), always the same DR3 source.")
t = [["DR16Q quasars", "750,414"], ["with Gaia DR3 counterpart < 1″ (G_DR2 < 20.5)", "301,344 (40.2%)"],
     ["… with XP coefficients", "36,404 (4.85%)"], ["… with Gaia QSOC redshift", "36,356"],
     ["usable after cleaning", "36,402"], ["G (5 / 50 / 95 percentile)", "17.32 / 18.56 / 18.97"],
     ["z (5 / 50 / 95 percentile)", "0.43 / 1.50 / 2.67"]]
tb = Table(t, colWidths=[W * 0.62, W * 0.3])
tb.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "DV", 8.5), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("ALIGN", (1, 0), (1, -1), "RIGHT")]))
story.append(tb)
story.append(Spacer(1, 6))
P("Unlike the generic G &lt; 17.65 limit for published XP spectra, 92% of the matched quasars with XP are "
  "fainter than that, down to G ≈ 19–20.")
fig("sample.png", "<b>Fig. 1.</b> Matched sample: SDSS redshift and Gaia G distributions.")

story.append(Paragraph("2.2  XP coefficients (02_fetch_xp.py)", h2))
P("ESA's TAP service does not expose the XP coefficient table, and its DataLink service limits anonymous "
  "users to 50 sources per request at ~5 s of server time per source (~1,000 spectra per 25 min). The "
  "coefficients were instead retrieved from the AIP Gaia mirror (gaia.aip.de), whose TAP service has "
  "<font face='DVMono'>gaiadr3.xp_continuous_mean_spectrum</font>: all 36,404 spectra in ~8 minutes. "
  "For 4,000 spectra fetched from both services, the coefficients are bit-identical and the errors and "
  "correlations agree to float32 rounding (≤ 6×10<super>−8</super> relative).")

# ------------------------------------------------------------------ methods
story.append(Paragraph("3  Methods", h1))
bullets([
    "<b>Features (03_features.py).</b> The 110 coefficients divided by the G-band flux. They are linear in flux, so "
    "this leaves a pure shape vector. Calibrated, sampled spectra (336–1020 nm, GaiaXPy) are used only for "
    "visualisation.",
    "<b>Feature sets.</b> A: BP−RP, BP−G, G−RP, G. B: all 110 normalised coefficients + G. "
    "B<sub>low</sub>: first 20 BP + 20 RP coefficients + G. G is included everywhere, since magnitude "
    "carries some redshift information through the luminosity function.",
    "<b>Models (05_regress.py).</b> kNN (k = 15, distance-weighted, standardised + 30-component whitened PCA), "
    "LightGBM (1,500 trees), MLP (256-128-64 units, quantile-to-Gaussian input scaling). Target log(1+z). "
    "5-fold CV with a fixed seed; every prediction is out-of-fold.",
    "<b>Metrics.</b> Δz/(1+z); σ<sub>NMAD</sub> = 1.4826 × median absolute deviation; outlier fractions at "
    "0.1 and 0.2; fraction within 0.01(1+z).",
    "<b>Controls.</b> LightGBM on shuffled redshifts; Gaia QSOC on the same objects.",
    "<b>p(z) (06_pz.py).</b> MLP classifier over 56 bins in log(1+z), 5-fold CV. Gives the mode, median, 68% "
    "interval, number of significant peaks and the probability integral transform (PIT).",
])

# ------------------------------------------------------------------ results
story.append(Paragraph("4  Results", h1))
story.append(Paragraph("4.1  The signal is visible", h2))
fig("redshift_ladder.png", "<b>Fig. 2.</b> Left: calibrated XP spectra stacked in equal-count redshift bins, "
    "divided by the mean observed shape (log ratio). Dashed: λ<sub>rest</sub>(1+z) tracks. Lyα, CIV, CIII], MgII, "
    "Hβ+[OIII] and Hα follow their tracks; Lyα-forest absorption shows blueward of Lyα at high z. Faint "
    "parallel stripes are partly weaker lines and partly basis-function ringing. The vertical seam near "
    "640 nm at low z is the BP/RP join. Right: example stacks.")
fig("coef_vs_z.png", "<b>Fig. 3.</b> |Spearman ρ| and mutual information (MI) of each coefficient with z. "
    "Dotted lines: MI of the colours (0.23, 0.37, 0.24 nats). Several single coefficients carry 0.8–1.0 nats, "
    "mostly low-order RP terms.")
fig("pca_by_z.png", "<b>Fig. 4.</b> First principal components of the normalised coefficients, coloured by "
    "redshift.", width=W * 0.95)

story.append(Paragraph("4.2  How much information: cross-validated regression", h2))
m = pd.read_csv(DATA / "metrics.csv")
hdr = ["Features", "Model", "N", "bias", "σ<sub>NMAD</sub>", "outl. 0.1", "outl. 0.2", "within 0.01(1+z)"]
rows = [hdr] + [[r.features, r.model, f"{r.N:,}", f"{r.bias_median:+.4f}", f"{r.sigma_NMAD:.4f}",
                 f"{r['outlier_0.1']:.1%}", f"{r['outlier_0.2']:.1%}", f"{r['frac_|dz|<0.01']:.1%}"]
                for _, r in m.iterrows()]
rows = [[Paragraph(str(c), cell) for c in row] for row in rows]
tb = Table(rows, colWidths=[W * x for x in (0.25, 0.14, 0.09, 0.1, 0.09, 0.09, 0.09, 0.15)], repeatRows=1)
tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")),
                        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
story.append(tb)
P("<b>Table 1.</b> 5-fold cross-validated metrics (data/metrics.csv). The last two rows rerun the models on the "
  "36,355 objects that also have a QSOC redshift.", cap)
bullets([
    "Coefficients beat colours by a factor 20–40 in σ<sub>NMAD</sub> and up to ~60 in catastrophic outliers. "
    "Colours cannot separate redshifts between z ≈ 1 and 2.5: predictions there pile up near 1.7.",
    "kNN on the low-order coefficients is the sharpest (σ<sub>NMAD</sub> 0.0040), close to QSOC. Dropping the "
    "noisy high-order coefficients matters for kNN, much less for trees.",
    "The MLPs give the fewest catastrophic outliers (0.50–0.59%), about 3× fewer than QSOC. Plain "
    "standardisation broke the MLP (on 3,000 objects it was no better than colours); quantile scaling fixed it.",
    "The shuffled-redshift control has no skill (σ<sub>NMAD</sub> 0.29, 52% outliers).",
    "With 3,000 objects (an early random subset), the scores were much worse (kNN on all coefficients "
    "σ<sub>NMAD</sub> 0.074 vs 0.011 with 36k): these models are data-limited.",
])
fig("z_pred_vs_true.png", "<b>Fig. 5.</b> Predicted vs SDSS redshift: colours only, XP coefficients (MLP), Gaia QSOC. "
    "Red lines: loci where one emission line is mistaken for another, (1+z<sub>pred</sub>)/(1+z<sub>true</sub>) = "
    "λ<sub>a</sub>/λ<sub>b</sub>. QSOC's remaining outliers include a cluster of z ≈ 0.5–1 quasars placed near z ≈ 3.5.")
fig("metrics_binned.png", "<b>Fig. 6.</b> σ<sub>NMAD</sub> and outlier fraction vs SDSS redshift and Gaia G "
    "(LightGBM models and QSOC). All methods degrade smoothly towards G ≈ 19.5.")

story.append(Paragraph("4.3  Redshift probabilities and line confusion", h2))
P("The p(z) classifier gives 0.63% outliers (mode) and σ<sub>NMAD</sub> 0.011 (median). 2.3% of objects get a "
  "p(z) with more than one significant peak. For 43% of the remaining catastrophic outliers, the model "
  "still puts more than 20% probability within 0.05(1+z) of the true redshift, so most of its failures are "
  "flagged ambiguities. The 68% intervals are too wide (81% coverage), probably because the "
  "bins are coarse. A first attempt with a LightGBM multiclass model diverged (one-hot probabilities, "
  "33% outliers) and was replaced.")
fig("pz_examples.png", "<b>Fig. 7.</b> Example p(z): single-peaked correct, multi-peaked, and catastrophic "
    "cases. Red: SDSS redshift.")
fig("pz_calibration.png", "<b>Fig. 8.</b> PIT histogram, number of significant peaks, and p(z) mode vs SDSS z.")

# ------------------------------------------------------------------ prior work
story.append(Paragraph("5  Caveats", h1))
bullets([
    "The sample is DR16Q-selected and relatively bright (G ≲ 19.5). Its performance does not carry over to "
    "fainter or differently selected quasars, or to contamination by stars and galaxies, which was not "
    "modelled here.",
    "Cross-validation is within DR16Q only. No independent test sample (e.g. DESI) was used, and no "
    "spatial-block split was tried.",
    "QSOC's template was built from SDSS DR12Q quasars, so the QSOC comparison is not fully independent "
    "of SDSS. The comparison with published deep-learning redshifts was not done (Sect. 6).",
    "Hyperparameters were not tuned. The p(z) intervals are not calibrated.",
])

story.append(Paragraph("6  Prior work and assessment", h1))
P("A literature search found refereed work that already estimates quasar redshifts from Gaia XP spectra "
  "with models trained on SDSS. Only abstracts and summaries were read, because the full texts were not "
  "reachable from the analysis environment.")
bullets([
    "Delchambre et al. 2023, A&amp;A 674, A31 — Gaia DR3 QSOC: χ² fit of a composite quasar template (from SDSS "
    "DR12Q) to BP/RP spectra.",
    "Storey-Fisher et al. 2024, ApJ — Quaia: kNN on QSOC redshifts plus Gaia/unWISE photometry, trained on SDSS; "
    "reports outliers from line aliasing.",
    "Fu et al. 2024, ApJS 271, 54 (CatNorth) and Fu et al. 2025, ApJS 279, 54 (CatSouth) — a CNN on sampled XP "
    "spectra, trained on SDSS (CatSouth reports σ<sub>NMAD</sub> ≈ 0.030, outliers ≈ 2.5%).",
    "“Unveiling Quasars in Gaia DR3: Multimodal Classification and Redshift Estimation of BP/RP Sources with "
    "Deep Learning”, ApJS, doi:10.3847/1538-4365/ae4003 — a deep network on the Hermite coefficients, sampled "
    "fluxes and catalogue features (SpecNet-Z); catalogue on Zenodo (10.5281/zenodo.17960118).",
])
P("The core idea is therefore established. The only potentially new point, that a simple kNN on "
  "low-order coefficients may be much sharper than the published CNNs (0.004 vs ≈ 0.03), would need a "
  "comparison on identical objects. That comparison must account for the published models having been "
  "trained on SDSS quasars that overlap this sample. It was not pursued.")

# ------------------------------------------------------------------ reproducibility
story.append(Paragraph("7  Reproducibility", h1))
P("All code is in <font face='DVMono'>gaia_qso_z/</font> and listed in full in the Appendix: "
  "<font face='DVMono'>common.py</font>, <font face='DVMono'>01_crossmatch.py</font> … "
  "<font face='DVMono'>06_pz.py</font>, <font face='DVMono'>run_all.sh</font>, "
  "<font face='DVMono'>make_report.py</font> (this report), <font face='DVMono'>requirements.txt</font> and "
  "<font face='DVMono'>README.md</font>. To reproduce: <font face='DVMono'>pip install -r requirements.txt</font> "
  "then <font face='DVMono'>./run_all.sh</font>. Network access is needed to data.sdss.org (DR16Q, 1.7 GB), "
  "gea.esac.esa.int (Gaia TAP) and gaia.aip.de (XP coefficients). The full run takes about an hour on 4 CPUs, "
  "dominated by the cross-match (~10 min) and model training. Raw data products (~3 GB) are not part of the code "
  "and are regenerated by the scripts.")
P("Python 3.13 with: " + ", ".join((ROOT / "requirements.txt").read_text().split()) + ".")

# ------------------------------------------------------------------ appendix: code
story.append(PageBreak())
story.append(Paragraph("Appendix: source code", h1))
for name in ["README.md", "requirements.txt", "run_all.sh", "common.py", "01_crossmatch.py", "02_fetch_xp.py",
             "03_features.py", "04_explore.py", "05_regress.py", "06_pz.py", "make_report.py"]:
    story.append(Paragraph(name, h2))
    story.append(Preformatted((ROOT / name).read_text(), code))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawString(1.8 * cm, 1.1 * cm, "Quasar redshifts from Gaia DR3 XP coefficients — project report")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(ROOT / "REPORT.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        topMargin=1.6 * cm, bottomMargin=1.8 * cm, title="Quasar redshifts from Gaia DR3 XP coefficients",
                        author="prepared with Claude Code")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", ROOT / "REPORT.pdf")
