import numpy as np
from scipy.stats import entropy

def kl_divergence(p, q):
    mask = (p > 0)
    return np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-12)))

def mutual_information(p_xy, p_x, p_y):
    """Compute mutual information I(X;Y) for discrete distributions."""
    joint = p_x[:, None] * p_y[None, :]
    ratio = p_xy / (joint + 1e-12)
    mask = p_xy > 0
    return np.sum(p_xy[mask] * np.log(ratio[mask]))

def information_bottleneck(p_x, p_y_given_x, beta, num_clusters, num_iterations=100):
    num_x, num_y = p_y_given_x.shape
    p_t_given_x = np.random.rand(num_x, num_clusters)
    p_t_given_x /= p_t_given_x.sum(axis=1, keepdims=True)

    for _ in range(num_iterations):
        p_t = (p_x[:, None] * p_t_given_x).sum(axis=0)
        p_y_given_t = np.zeros((num_clusters, num_y))
        for t in range(num_clusters):
            for y in range(num_y):
                p_y_given_t[t, y] = np.sum(p_t_given_x[:, t] * p_x * p_y_given_x[:, y])
            if p_t[t] > 0:
                p_y_given_t[t, :] /= p_t[t]
        for x in range(num_x):
            D_kl = np.array([kl_divergence(p_y_given_x[x, :], p_y_given_t[t, :]) 
                             for t in range(num_clusters)])
            p_t_given_x[x, :] = p_t * np.exp(-beta * D_kl)
            p_t_given_x[x, :] /= np.sum(p_t_given_x[x, :])

    p_xt = p_x[:, None] * p_t_given_x
    p_y = (p_x[:, None] * p_y_given_x).sum(axis=0)
    p_yt = np.zeros((num_clusters, num_y))
    for t in range(num_clusters):
        for y in range(num_y):
            p_yt[t, y] = np.sum(p_x * p_t_given_x[:, t] * p_y_given_x[:, y])

    I_xt = mutual_information(p_xt, p_x, p_t)

    I_ty = mutual_information(p_yt, p_t, p_y)

    return p_t_given_x, I_xt, I_ty
