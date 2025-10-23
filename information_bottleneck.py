import numpy as np

def kl(p, q):
    p, q = np.array(p), np.array(q)
    return np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))

def ib_simple(p_x, p_y_given_x, beta=5.0, n_clusters=2, n_iter=50):
    n_x, n_y = p_y_given_x.shape
    p_t_given_x = np.random.dirichlet(np.ones(n_clusters), size=n_x)

    for _ in range(n_iter):
        p_t = np.sum(p_x[:, None] * p_t_given_x, axis=0)
        p_y_given_t = (p_t_given_x.T @ (p_x[:, None] * p_y_given_x)) / p_t[:, None]

        for i in range(n_x):
            dkl = np.array([kl(p_y_given_x[i], p_y_given_t[j]) for j in range(n_clusters)])
            p_t_given_x[i] = p_t * np.exp(-beta * dkl)
            p_t_given_x[i] /= np.sum(p_t_given_x[i])

    p_xt = p_x[:, None] * p_t_given_x
    I_x_t = np.sum(p_xt * np.log((p_t_given_x + 1e-12) / (p_t[None, :] + 1e-12)))
    I_t_y = np.sum((p_t[:, None] * p_y_given_t) * np.log((p_y_given_t + 1e-12) / np.sum(p_x[:, None]*p_y_given_x, axis=0)[None,:]))

    return p_t_given_x, I_x_t, I_t_y

p_x = np.array([0.5, 0.5])
p_y_given_x = np.array([[0.9, 0.1], [0.2, 0.8]])

p_t_given_x, I_x_t, I_t_y = ib_simple(p_x, p_y_given_x, beta=5.0)
print("p(t|x):\n", p_t_given_x)
print("I(X;T):", round(I_x_t, 4), "| I(T;Y):", round(I_t_y, 4))
