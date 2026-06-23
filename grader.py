def grade(parsed, ground_truth, category):
    """Compare parsed model answer to ground truth. Returns True/False/None.
    None means: don't include in accuracy calc (unparseable response, or
    confabulation, which is manually reviewed instead).

    count uses EXACT match, not tolerance -- a miscount is a real,
    gradable hallucination, not noise to smooth over. If COCO/SLAKE
    ground truth has its own annotation imperfections, that's a documented
    limitation of the source data, not something the grader should
    compensate for."""
    if parsed == 'unparseable' or ground_truth is None:
        return None

    gt = str(ground_truth).strip().lower()
    p = str(parsed).strip().lower()

    if category in ('existence', 'spatial', 'attribute', 'count'):
        return p == gt

    if category == 'ocr_numeric':
        def to_float(s):
            s = s.replace('%', '').replace('$', '').replace(',', '').strip()
            try:
                return float(s)
            except ValueError:
                return None
        pf, gf = to_float(p), to_float(gt)
        if pf is not None and gf is not None:
            return abs(pf - gf) <= 0.1
        return p == gt

    if category == 'confabulation':
        return None

    return None
