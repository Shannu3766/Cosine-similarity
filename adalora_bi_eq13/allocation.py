import numpy as np

def bi_allocate_ranks(scores, R, tau=1.0):
    scores = np.array(scores, dtype=float)
    if scores.size == 0:
        return []
    x = scores - np.max(scores)
    exp = np.exp(x / tau)
    weights = exp / np.sum(exp)
    raw = R * weights
    flo = np.floor(raw).astype(int)
    flo = np.maximum(flo, 1)
    residuals = raw - np.floor(raw)
    cur = int(flo.sum())
    diff = R - cur
    ranks = flo.tolist()
    if diff > 0:
        order = list(np.argsort(-residuals))
        for idx in order[:diff]:
            ranks[idx] += 1
    elif diff < 0:
        order = list(np.argsort(residuals))
        i = 0
        while diff < 0 and i < len(order):
            idx = order[i]
            if ranks[idx] > 1:
                ranks[idx] -= 1
                diff += 1
            i += 1
    return ranks
