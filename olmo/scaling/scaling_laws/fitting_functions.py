import numpy as np
import scipy

# Power Law functions


def openai_fit(x, a, b, c):
    return (a / x + c) ** b


def chinchilla_fit(x, a, b, c):
    return a * x**b + c


def chinchilla_contaminated_fit(x, a, b, c, d):
    return (a * x**b + c) * (1 - x / d)


# Scipy curve_fit with least squares
def get_coefficients(train_xs, train_ys, fitting_func, p0, disp=True):
    if isinstance(train_xs[0], list):
        train_xs = np.array(train_xs).transpose()
    coeffs = scipy.optimize.curve_fit(fitting_func, train_xs, train_ys, p0=p0, maxfev=50000)[0]
    coeffs_string = ", ".join([chr(ord("a") + i) + f" = {coeffs[i]:.2f}" for i in range(len(coeffs))])
    if disp:
        print(f"{fitting_func.__name__}: {coeffs_string}")
    return coeffs


# x = flops
# p[0] = A, p[1] = B, p[2] = E
def chinchilla_flops_fit(x, p):
    # return ax**b + E
    return p[0] * np.pow(x, p[1]) + p[2]


def grad_chinchilla_flops_fit(x, p):
    grad_A = np.pow(x, p[1])
    grad_B = p[0] * np.pow(x, p[1]) * np.log(x)
    grad_E = 1
    return [grad_A, grad_B, grad_E]


# x[0] = d, x[1] = h
# p[0] = b = log100(B), p[1] = beta, p[2] = E, p[3] = F
def chinchilla_d_lr_fit(x, p):
    return 100 ** p[0] / x[0] ** p[1] + p[2] + p[3] * x[1]


def grad_chinchilla_d_lr_fit(x, p):
    grad_b = (1 / x[0] ** p[1]) * (100 ** p[0] * np.log(100))
    grad_beta = -(100 ** p[0]) * np.log(x[0]) / x[0] ** p[1]
    grad_E = 1
    grad_F = x[1]
    return [grad_b, grad_beta, grad_E, grad_F]


def chinchilla_n_d_fit_e(x, p0, p1, p2, p3, p4):
    return np.exp(p0) / x[0] ** p2 + np.exp(p1) / x[1] ** p3 + p4


# x[0] = n, x[1] = d
# p[0] = a = log(A), p[1] = b = log(B), p[2] = alpha, p[3] = beta, p[4] = E
def chinchilla_n_d_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4]


def grad_chinchilla_n_d_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E]


# x[0] = n, x[1] = d, x[2] = h
# p[0] = a = log(A), p[1] = b = log(B), p[2] = alpha, p[3] = beta, p[4] = E, p[5] = F
def chinchilla_n_d_lr_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E + F * x[2]
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] + p[5] * x[2]


def grad_chinchilla_n_d_lr_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = x[2]
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F]


# x[0] = n, x[1] = d, x[2] = h
# p[0] = a = log(A), p[1] = b = log(B), p[2] = alpha, p[3] = beta, p[4] = E, p[5] = F
def chinchilla_n_d_lr_minus_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * (1 - x[2])
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] - p[5] * (1 - x[2])


def grad_chinchilla_n_d_lr_minus_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = -(1 - x[2])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F]


def chinchilla_n_d_lr_log_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E + F * x[2] * np.log(x[0] / e**r + s)
    return (
        np.exp(p[0]) / x[0] ** p[2]
        + np.exp(p[1]) / x[1] ** p[3]
        + p[4]
        + p[5] * x[2] * np.log(x[0] / np.exp(p[6]) + p[7])
    )


def grad_chinchilla_n_d_lr_log_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = x[2] * np.log(x[0] / np.exp(p[6]) + p[7])
    grad_r = p[5] * x[2] * (1 / (x[0] / np.exp(p[6]) + p[7])) * x[0] * (-1 / np.exp(p[6]))
    grad_s = p[5] * x[2] * (1 / (x[0] / np.exp(p[6]) + p[7]))
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r, grad_s]


def chinchilla_n_d_lr_power_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E + F * x[2] * x[0]**r
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] + p[5] * x[2] * x[0] ** p[6]


def grad_chinchilla_n_d_lr_power_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = x[2] * x[0] ** p[6]
    grad_r = p[5] * x[2] * x[0] ** p[6] * np.log(x[0])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r]


def chinchilla_n_d_lr_power_minus_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * (1 - x[2]) * x[0]**r
    return np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] - p[5] * (1 - x[2]) * x[0] ** p[6]


def grad_chinchilla_n_d_lr_power_minus_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = -(1 - x[2]) * x[0] ** p[6]
    grad_r = -p[5] * (1 - x[2]) * x[0] ** p[6] * np.log(x[0])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r]


def chinchilla_n_d_lr_power_minus_powerd_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * (1 - x[2]) * x[0]**r * x[1]**s
    return (
        np.exp(p[0]) / x[0] ** p[2]
        + np.exp(p[1]) / x[1] ** p[3]
        + p[4]
        - p[5] * (1 - x[2]) * x[0] ** p[6] * x[1] ** p[7]
    )


def grad_chinchilla_n_d_lr_power_minus_powerd_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = -(1 - x[2]) * x[0] ** p[6] * x[1] ** p[7]
    grad_r = -p[5] * (1 - x[2]) * x[1] ** p[7] * x[0] ** p[6] * np.log(x[0])
    grad_s = -p[5] * (1 - x[2]) * x[0] ** p[6] * x[1] ** p[7] * np.log(x[1])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r, grad_s]


def chinchilla_n_d_lr_power_minus_powertd_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * (1 - x[2]) * x[0]**r * (x[1]**s + t)
    return (
        np.exp(p[0]) / x[0] ** p[2]
        + np.exp(p[1]) / x[1] ** p[3]
        + p[4]
        - p[5] * (1 - x[2]) * x[0] ** p[6] * (x[1] ** p[7] + np.exp(p[8]))
    )


def grad_chinchilla_n_d_lr_power_minus_powertd_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = -(1 - x[2]) * x[0] ** p[6] * (x[1] ** p[7] + np.exp(p[8]))
    grad_r = -p[5] * (1 - x[2]) * (x[1] ** p[7] + np.exp(p[8])) * x[0] ** p[6] * np.log(x[0])
    grad_s = -p[5] * (1 - x[2]) * x[0] ** p[6] * x[1] ** p[7] * np.log(x[1])
    grad_t = -p[5] * (1 - x[2]) * x[0] ** p[6] * np.exp(p[8])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r, grad_s, grad_t]


def chinchilla_n_d_lr_power_minus_logtd_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * (1 - x[2]) * x[0]**r * (log(x[1]) + s)
    return (
        np.exp(p[0]) / x[0] ** p[2]
        + np.exp(p[1]) / x[1] ** p[3]
        + p[4]
        - p[5] * (1 - x[2]) * x[0] ** p[6] * (np.log(x[1]) + p[7])
    )


def grad_chinchilla_n_d_lr_power_minus_logtd_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = -(1 - x[2]) * x[0] ** p[6] * (np.log(x[1]) + p[7])
    grad_r = -p[5] * (1 - x[2]) * (np.log(x[1]) + p[7]) * x[0] ** p[6] * np.log(x[0])
    grad_s = -p[5] * (1 - x[2]) * x[0] ** p[6]
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r, grad_s]


def chinchilla_n_d_lr_logt_minus_logtd_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * (1 - x[2]) / (log(x[0]) + r) * (log(x[1]) + s)
    return (
        np.exp(p[0]) / x[0] ** p[2]
        + np.exp(p[1]) / x[1] ** p[3]
        + p[4]
        - p[5] * (1 - x[2]) / (np.log(x[0]) + p[6]) * (np.log(x[1]) + p[7])
    )


def grad_chinchilla_n_d_lr_logt_minus_logtd_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = -(1 - x[2]) / (np.log(x[0]) + p[6]) * (np.log(x[1]) + p[7])
    grad_r = -p[5] * (1 - x[2]) * (np.log(x[1]) + p[7]) * (-1 / (np.log(x[0]) + p[6]) ** 2)
    grad_s = -p[5] * (1 - x[2]) / (np.log(x[0]) + p[6])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r, grad_s]


def tissue_fit(x, p):
    # return e**a / x[0]**alpha + e**b / x[1]**beta + E - F * x[2] * x[0]**r
    return max(1e-8, np.exp(p[0]) / x[0] ** p[2] + np.exp(p[1]) / x[1] ** p[3] + p[4] - p[5] * x[2] * x[0] ** p[6])


def grad_tissue_fit(x, p):
    grad_a = np.exp(p[0]) / x[0] ** p[2]
    grad_b = np.exp(p[1]) / x[1] ** p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0] ** p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1] ** p[3]
    grad_E = 1
    grad_F = -x[2] * x[0] ** p[6]
    grad_r = -p[5] * x[2] * x[0] ** p[6] * np.log(x[0])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r]


def sigmoid(x, L, x0, k, b):
    o = L / (1 + np.exp(-k * (x - x0))) + b
    return o


# Scipy minimize w/ Huber loss
def get_coefficients_huber(
    train_xs, train_ys, fitting_func, grad_func, p0, bounds, disp: bool = True, max_iter: int = 10000
):
    def huber_loss(x, delta):
        if np.abs(x) < delta:
            return 0.5 * x**2
        else:
            return delta * (np.abs(x) - 0.5 * delta)

    def loss_fn(p, train_xs, train_ys, delta):
        actuals = train_ys
        preds = [fitting_func(x, p) for x in train_xs]
        loss = np.sum(
            [huber_loss(np.log(pred) - np.log(actual), delta=delta) for actual, pred in zip(actuals, preds)]
        )
        return loss

    def jac_fn(p, train_xs, train_ys, delta):
        actuals = train_ys
        preds = [fitting_func(x, p) for x in train_xs]
        grads = [grad_func(x, p) for x in train_xs]
        us = [np.log(pred) - np.log(actual) for actual, pred in zip(actuals, preds)]
        grad_us = [u if np.abs(u) < delta else (delta * np.abs(u) / u) for u in us]
        results = [
            np.sum([grad_u * (1 / pred) * grad[i] for grad_u, pred, grad in zip(grad_us, preds, grads)])
            for i in range(len(grads[0]))
        ]
        return results

    assert len(train_xs) == len(train_ys)
    delta = 1e-3
    res = scipy.optimize.minimize(
        loss_fn,
        p0,
        args=(train_xs, train_ys, delta),
        jac=jac_fn,
        bounds=bounds,
        tol=0.0,
        method="L-BFGS-B",
        options={"ftol": 0.0, "gtol": 1e-10, "maxiter": max_iter, "disp": disp},
    )
    # res = scipy.optimize.minimize(loss_fn, p0, args=(train_xs, train_ys, delta), jac=jac_fn, tol=0.0, method='BFGS', options={'gtol': 1e-10, 'maxiter': 10000, 'disp': True})
    # print(res.message)
    coeffs = res.x
    if disp:
        print(f"coeffs: {coeffs}")
    return coeffs


def get_coefficients_huber_nolog(
    train_xs, train_ys, fitting_func, grad_func, p0, bounds, disp: bool = True, max_iter: int = 10000
):
    def huber_loss(x, delta):
        if np.abs(x) < delta:
            return 0.5 * x**2
        else:
            return delta * (np.abs(x) - 0.5 * delta)

    def loss_fn(p, train_xs, train_ys, delta):
        actuals = train_ys
        preds = [fitting_func(x, p) for x in train_xs]
        loss = np.sum([huber_loss(pred - actual, delta=delta) for actual, pred in zip(actuals, preds)])
        return loss

    def jac_fn(p, train_xs, train_ys, delta):
        actuals = train_ys
        preds = [fitting_func(x, p) for x in train_xs]
        grads = [grad_func(x, p) for x in train_xs]
        us = [pred - actual for actual, pred in zip(actuals, preds)]
        grad_us = [u if np.abs(u) < delta else (delta * np.abs(u) / u) for u in us]
        results = [
            np.sum([grad_u * grad[i] for grad_u, pred, grad in zip(grad_us, preds, grads)])
            for i in range(len(grads[0]))
        ]
        return results

    assert len(train_xs) == len(train_ys)
    delta = 1e-3
    res = scipy.optimize.minimize(
        loss_fn,
        p0,
        args=(train_xs, train_ys, delta),
        jac=jac_fn,
        bounds=bounds,
        tol=0.0,
        method="L-BFGS-B",
        options={"ftol": 0.0, "gtol": 1e-10, "maxiter": max_iter, "disp": disp},
    )
    # res = scipy.optimize.minimize(loss_fn, p0, args=(train_xs, train_ys, delta), jac=jac_fn, tol=0.0, method='BFGS', options={'gtol': 1e-10, 'maxiter': 10000, 'disp': True})
    # print(res.message)
    coeffs = res.x
    loss = res.fun
    if disp:
        print(f"coeffs: {coeffs}")
    return coeffs, loss
