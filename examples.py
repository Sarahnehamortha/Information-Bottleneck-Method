import numpy as np
from information_bottleneck import information_bottleneck

p_x = np.array([0.5, 0.5])
p_y_given_x = np.array([[0.9, 0.1],
                        [0.2, 0.8]])
beta = 5.0
num_clusters = 2
p_t_given_x, I_xt, I_ty = information_bottleneck(p_x, p_y_given_x, beta, num_clusters)
print("p(t|x):\n", p_t_given_x)
print("I(X;T):", I_xt, "I(T;Y):", I_ty)

p_x_multi = np.array([0.1, 0.15, 0.2, 0.15, 0.2, 0.2])
p_y_given_x_multi = np.array([[0.8,0.1,0.1],
                              [0.1,0.7,0.2],
                              [0.2,0.2,0.6],
                              [0.7,0.2,0.1],
                              [0.05,0.85,0.1],
                              [0.1,0.1,0.8]])
beta = 5.0
num_clusters = 3
p_t_given_x, I_xt, I_ty = information_bottleneck(p_x_multi, p_y_given_x_multi, beta, num_clusters)
print("Multi-class p(t|x):\n", p_t_given_x)
print("I(X;T):", I_xt, "I(T;Y):", I_ty)
