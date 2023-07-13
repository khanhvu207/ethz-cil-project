import math


def cos_anneal(e0, e1, t0, t1, e):
    """ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1]"""
    alpha = max(
        0, min(1, (e - e0) / (e1 - e0))
    )  # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi / 2)  # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0  # interpolate accordingly
    return t
