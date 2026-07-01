"""
Build the Final Biweekly Report (1 page, condensed).
Editing the Fourth report template — all formatting preserved.

Updated: P0 (feature compression) + P1 (Patch Drop, MixStyle, grid search) included.
WBS titles mapped from new-WBS.png project structure.
"""
from docx import Document

SRC = 'docs/Fourth Biweekly Report - Ye Li.docx'
DST = 'docs/Final Biweekly Report - Ye Li.docx'

doc = Document(SRC)
table = doc.tables[0]


def get_font_ref(cell):
    for p in cell.paragraphs:
        for r in p.runs:
            if r.font.size:
                return r
    return None


def clear_old_content(cell, header_keyword):
    paras = list(cell.paragraphs)
    header_idx = None
    for i, p in enumerate(paras):
        if header_keyword in p.text:
            header_idx = i
            break
    if header_idx is None:
        return

    p_header = paras[header_idx]
    full_text = p_header.text
    newline_pos = full_text.find('\n')

    if newline_pos >= 0:
        keep_text = full_text[:newline_pos]
        cumul = ''
        for r in list(p_header.runs):
            if len(cumul) >= len(keep_text):
                r._element.getparent().remove(r._element)
            else:
                cumul += r.text

    for p in paras[header_idx + 1:]:
        p._element.getparent().remove(p._element)


def add_paras(cell, lines, font_ref=None):
    for text in lines:
        p = cell.add_paragraph()
        r = p.add_run(text)
        if font_ref:
            r.font.size = font_ref.font.size
            r.font.name = font_ref.font.name


def fix_date_in_paragraph(p, old_start, new_start, old_end, new_end):
    full = p.text
    if old_start not in full and old_end not in full:
        return
    full = full.replace(old_start, new_start).replace(old_end, new_end)
    if p.runs:
        p.runs[0].text = full
        for r in p.runs[1:]:
            r._element.getparent().remove(r._element)


# =========================================================================
# Row 1: Reporting Period
# =========================================================================
for cell in table.rows[1].cells:
    for p in cell.paragraphs:
        fix_date_in_paragraph(p, '18th May', '1st June', '31th May', '14th June')
print("OK: Date -> 1st June - 14th June")

# =========================================================================
# Row 3: Individual Work Reports (2 paragraphs covering all P0+P1 work)
# =========================================================================
cell3 = table.rows[3].cells[0]
font3 = get_font_ref(cell3)
clear_old_content(cell3, 'Individual Work Reports')

iwr = [
    "First, I fixed a critical JPM bug where local branches received no ID loss "
    "supervision under ArcFace; independent classifiers were created for all five "
    "branches. Next, I replaced Triplet+Center loss with feature-level Circle Loss and "
    "upgraded the optimizer from SGD to AdamW with configurable betas. Finally, I added "
    "Random Patch Drop (ViT token masking for basketball occlusion simulation) and "
    "MixStyle (cross-sample feature statistics mixing for indoor/outdoor court domain "
    "generalization) as training-time augmentations.",

    "Additionally, I fixed hardcoded device='cuda' across all loss modules, created "
    "the vit_all_optimizations.yml combined configuration with ablation variants "
    "(vit_transreid_stride_adamw.yml, vit_transreid_stride_circle_sgd.yml), "
    "developed a checkpoint migration utility, built a focused grid search generator "
    "covering 5 key hyperparameters, and rewrote the project README with "
    "architecture documentation and a 4-phase training-to-deployment workflow.",
]
add_paras(cell3, iwr, font3)
print("OK: Individual Work Reports (2 paras, P0+P1 included)")

# =========================================================================
# Row 4: Key Achievements (WBS titles from new-WBS.png)
# =========================================================================
cell4 = table.rows[4].cells[0]
font4 = get_font_ref(cell4)
clear_old_content(cell4, 'Key Achievements')

achievements = [
    "1. Baseline Model Optimization and Loss Function Design: Fixed JPM multi-branch "
    "ArcFace supervision bug; implemented feature-level Circle Loss with adaptive "
    "per-pair weighting; developed checkpoint migration utility. "
    "10h, 100%, Not done: full convergence verification.",

    "2. Training Pipeline Engineering and Optimization: Upgraded optimizer to AdamW "
    "with configurable betas; fixed hardcoded device='cuda' across all loss modules; "
    "created combined and ablation training configurations; built focused grid search "
    "generator covering 5 key hyperparameters. Investigated feature compression via "
    "shared projection layer — found it caused significant degradation in fine-grained "
    "identity discrimination accuracy (same-jersey players), so the full 3840-dim "
    "features are retained. "
    "8h, 100%, Not done: LR range test and grid search execution.",

    "3. Data Augmentation and Domain Generalization: Added Random Patch Drop (ViT "
    "token-level occlusion simulation) and MixStyle (cross-sample statistics mixing "
    "for court domain robustness); designed 4-phase optimization workflow (ablation "
    "-> grid search -> multi-seed -> ensemble). "
    "6h, 100%, Not done: augmentation probability sensitivity analysis.",
]
add_paras(cell4, achievements, font4)
print("OK: Key Achievements (6 WBS-aligned items with titles)")

# =========================================================================
# Row 5: Issues and Challenges
# =========================================================================
cell5 = table.rows[5].cells[0]
font5 = get_font_ref(cell5)
clear_old_content(cell5, 'Issues and Challenges')

issues = [
    "- The JPM+ArcFace bug was present through all prior training; baseline results "
    "(mAP 91.1%, Rank-1 93.6%) reflect undertrained local branches and are not directly "
    "comparable to the fixed architecture, requiring full retraining to establish "
    "updated metrics.",

    "- Circle Loss, AdamW, Patch Drop, and MixStyle each introduce new hyperparameters "
    "that interact with existing ArcFace settings; the expanded 15-dimension "
    "configuration space demands systematic grid search to avoid suboptimal minima.",

    "- The 3840-dim 5-branch concatenated feature vector poses challenges for the "
    "less-than-40ms inference latency constraint; preliminary investigation of feature "
    "compression (shared projection layer) caused significant degradation in "
    "fine-grained identity discrimination accuracy, so the full-dimensional features "
    "are retained for now.",
]
add_paras(cell5, issues, font5)
print("OK: Issues and Challenges (3 bullets)")

# =========================================================================
# Row 6: Plans for Next Biweekly Period (testing & summarizing focus)
# =========================================================================
cell6 = table.rows[6].cells[0]
font6 = get_font_ref(cell6)
clear_old_content(cell6, 'Plans for Next')

plans = [
    "- Execute the 4-phase optimization pipeline: (a) component ablation studies "
    "quantifying each optimization's contribution, (b) focused hyperparameter grid "
    "search on the 5 most impactful parameters, (c) multi-seed stability validation "
    "(seeds 1234, 42, 99), and (d) multi-model EMA ensemble with full test-time "
    "augmentation (multi-scale + QE + Re-Ranking) for the final score.",

    "- Produce comprehensive results tables reporting mAP, Rank-1, and Rank-5 for "
    "each configuration; identify the optimal hyperparameter combination and report "
    "final metrics with mean and standard deviation across seeds.",

    "- Compile the final technical report summarizing all optimization strategies, "
    "ablation findings, hyperparameter sensitivity analysis, and the complete "
    "development cycle from baseline to deployment-ready model configuration.",
]
add_paras(cell6, plans, font6)
print("OK: Plans (3 bullets, testing/summarizing focus)")

# =========================================================================
# Row 7: Risk Assessment
# =========================================================================
cell7 = table.rows[7].cells[0]
font7 = get_font_ref(cell7)
clear_old_content(cell7, 'Risk Assessment')

risks = [
    "- Combined JPM fix + Circle Loss + AdamW + Patch Drop + MixStyle changes may "
    "exhibit non-linear interaction effects, mitigated by planned ablation studies, "
    "multi-seed stability testing (seeds 1234, 42, 99), and single-variable grid "
    "search to isolate each component's marginal contribution.",

    "- The 3840-dim feature vector may exceed the inference latency budget on "
    "deployment hardware; preliminary feature compression caused significant "
    "accuracy loss, so alternative acceleration strategies (e.g., test-time branch "
    "pruning, FP16 inference, TensorRT optimization) will be explored as "
    "post-training optimizations that do not affect model accuracy.",
]
add_paras(cell7, risks, font7)
print("OK: Risk Assessment (2 bullets)")

# =========================================================================
doc.save(DST)
print(f"\nSaved: {DST}")
print("Done!")
