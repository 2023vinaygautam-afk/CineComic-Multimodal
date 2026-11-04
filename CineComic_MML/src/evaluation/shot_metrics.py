from __future__ import annotations


def boundary_prf(pred: list[float], gt: list[float], tol: float = 0.5) -> dict:
    """One-to-one greedy matching of cut times within +/- tol seconds."""
    pred, gt = sorted(pred), sorted(gt)
    used, tp = set(), 0

    for p in pred:
        cands = [(abs(p - g), j) for j, g in enumerate(gt) if j not in used and abs(p - g) <= tol]
        if cands:
            used.add(min(cands)[1])
            tp += 1

    P = tp / len(pred) if pred else (1.0 if not gt else 0.0)
    R = tp / len(gt) if gt else (1.0 if not pred else 0.0)
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"boundary_precision": P, "boundary_recall": R, "boundary_f1": F, "tp": tp}


def box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0
