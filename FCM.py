import numpy as np


def dist(a, b):
    return np.linalg.norm(a-b)
def FCM(D, k, m):
    epsilon = 1e-5
    data = D.detach().numpy()
    n = data.shape[0]
    p = data.shape[1]
    u = np.random.rand(n, k)
    u = u / np.sum(u, axis=1, keepdims=True)
    centers = np.zeros((k, p))
    for i in range(k):
        centers[i] = np.sum(np.power(u[:, i][:, np.newaxis], m) * data, axis=0) / np.sum(np.power(u[:, i], m))
    iteration = 0
    max_iterations = 100
    while iteration < max_iterations:
        iteration += 1
        u_old = u.copy()

        for i in range(n):
            for j in range(k):
                num = dist(data[i], centers[j]) ** 2
                den = np.sum([dist(data[i], centers[c]) ** 2 for c in range(k)])
                u[i][j] = num / den ** ((m - 1) / 2)

        u = u / np.sum(u, axis=1, keepdims=True)

        for j in range(k):
            centers[j] = np.sum(np.power(u[:, j][:, np.newaxis], m) * data, axis=0) / np.sum(np.power(u[:, j], m))

        if np.linalg.norm(u - u_old) < epsilon:
            break
    return u