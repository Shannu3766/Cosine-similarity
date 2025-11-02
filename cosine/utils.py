import numpy as np

def softmax(x, tau=1.0):
    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x / tau)
    return exp_x / np.sum(exp_x)

def allocate_from_raw(raw_ranks, R):
    """
    raw_ranks: numpy array of floats (R * weights).
    Returns integer ranks (rmin fixed=1) summing exactly to R.
    Strategy: floor then greedy assign by residuals; ensure rmin=1.
    """
    rmin = 1
    flo = np.floor(raw_ranks).astype(int)
    flo = np.maximum(flo, rmin)
    residuals = raw_ranks - np.floor(raw_ranks)
    cur = int(flo.sum())
    diff = R - cur
    ranks = flo.copy().tolist()
    if diff > 0:
        order = list(np.argsort(-residuals))
        for idx in order[:diff]:
            ranks[idx] += 1
    elif diff < 0:
        order = list(np.argsort(residuals))
        i = 0
        while diff < 0 and i < len(order):
            idx = order[i]
            if ranks[idx] > rmin:
                ranks[idx] -= 1
                diff += 1
            i += 1
        i = 0
        while diff < 0:
            idx = i % len(ranks)
            if ranks[idx] > rmin:
                ranks[idx] -= 1
                diff += 1
            i += 1
    return ranks
