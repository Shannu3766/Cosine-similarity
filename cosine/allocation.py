# examples/allocation.py
from cosine.utils import allocate_ranks_from_scores

def allocate_by_importance(scores_dict, total_R, tau=0.5, r_min=0):
    """
    scores_dict: dict name->score
    returns dict name->rank (ints) summing to total_R
    """
    names = list(scores_dict.keys())
    scores = [scores_dict[n] for n in names]
    ranks = allocate_ranks_from_scores(scores, total_R=total_R, tau=tau, r_min=r_min)
    return {n: int(r) for n, r in zip(names, ranks)}
