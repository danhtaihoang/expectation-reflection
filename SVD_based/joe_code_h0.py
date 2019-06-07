from numpy import matlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix

# number of input variables xi
n = 20
# number of possible states of 1 output variable y
my = 3
# numbers of possible states of inputs xi
mx = np.random.randint(2, 5, size=n)
#mx = np.full(n,3)
# number of samples
l = 2*(60**2)

# model parameters
v = np.random.normal(size=my)
w = np.array([np.random.normal(size=(mi, my)) for mi in mx])
v -= v.mean()
for i in range(n):
    w[i] -= w[i].mean(0)
    w[i] -= w[i].mean(1)[:, np.newaxis]

# sample data
x = np.hstack([np.random.randint(mi, size=(l, 1), dtype=int) for mi in mx])
h = v + np.array(
    [np.sum([w[i][x[t, i]] for i in range(n)], 0) for t in range(l)])
p = np.exp(h)
p /= p.sum(1)[:, np.newaxis]
y = (p.cumsum(1) < np.random.uniform(size=(l, 1))).sum(1)


def fit(x, y, mx, my, max_iter=100):
    l,n = x.shape  # 2018.12.14: Tai
    v = matlib.zeros(my)
    w = matlib.zeros((mx.sum(), my))

    # for convenience
    mx_cumsum = np.insert(mx.cumsum(), 0, 0)
    i1i2 = np.stack([mx_cumsum[:-1], mx_cumsum[1:]]).T

    # one-hot encoding of x
    x_oh = csc_matrix((np.ones(l * n), (np.repeat(range(l), n),
                                        (x + mx_cumsum[:-1]).flatten())))

    # SVD-based solve of x_oh * w = h
    x_oh_svd = svds(x_oh, k=mx.sum() - n + 1)
    x_oh_sv_pinv = x_oh_svd[1].copy()
    zero_sv = np.isclose(x_oh_sv_pinv, 0)
    x_oh_sv_pinv[zero_sv] = 0
    x_oh_sv_pinv[~zero_sv] = 1. / x_oh_sv_pinv[~zero_sv]
    x_oh_pinv = (x_oh_svd[2].T, x_oh_sv_pinv[:, np.newaxis], x_oh_svd[0].T)

    def solve(u):
        w = x_oh_pinv[2].dot(u)
        w = np.multiply(x_oh_pinv[1], w)
        w = x_oh_pinv[0] * w
        return w

    # one-hot encoding of y
    y_oh = csc_matrix((np.ones(l), (range(l), y)))
    # 'cold' states
    y_cold = ~(y_oh.toarray().astype(bool))

    # discrepancy
    d = [(1. / float(my) / float(my)) + 1]

    for it in range(1, max_iter):

        h0 = v
        h1 = x_oh * w
        p = np.exp(h0 + h1)
        p /= p.sum(1)

        # additive update
        dh = y_oh - p
        v = (h0 + dh).mean(0)
        w = solve(h1 + dh)

        v -= v.mean()
        w -= w.mean(1)
        for i1, i2 in i1i2:
            w[i1:i2] -= w[i1:i2].mean(0)

        # discrepancy: avg 2-norm squared of cold entries
        d.append(np.power(p[y_cold], 2).mean())

        print it, d[-1]

    return np.asarray(v), np.array([np.asarray(w[i1:i2])
                                    for i1, i2 in i1i2]), d[1:]


# store true values
v_true = v.copy()
w_true = w.copy()

# inference
v, w, d = fit(x, y, mx, my)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(d, 'k-')
lo = min(v.min(), np.vstack(w).min())
hi = max(v.max(), np.vstack(w).max())
grid = np.linspace(lo, hi)
ax[1].plot(grid, grid, 'k--', alpha=0.5)
ax[1].scatter(v, v_true, c='r', s=10)
#ax[1].scatter(np.vstack(w).flatten(), np.vstack(w_true).flatten(), c='b', s=1)
ax[0].set_xlabel('iteration')
ax[0].set_ylabel('discrepancy')
ax[1].set_xlabel('fitted')
ax[1].set_ylabel('true')
plt.tight_layout()
plt.show()
