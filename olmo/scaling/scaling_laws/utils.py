import scipy
import numpy as np

validation = [
    "c4_en-validation",
    "dolma_books-validation",
    "dolma_common-crawl-validation",
    "dolma_pes2o-validation",
    "dolma_reddit-validation",
    "dolma_stack-validation",
    "dolma_wiki-validation",
    "ice-validation",
    "m2d2_s2orc-validation",
    "pile-validation",
    "wikitext_103-validation",
]

downstream_bpb = [
    "piqa_rc_0shot_bpb",
    "hellaswag_rc_0shot_bpb",
    "winogrande_rc_0shot_bpb",
    "openbookqa_rc_0shot_bpb",
    "boolq_rc_0shot_bpb",
    # "sciq_rc_0shot_bpb",
    "arc_easy_rc_0shot_bpb",
    "arc_challenge_rc_0shot_bpb",
    "copa_rc_0shot_bpb",
    # "csqa_rc_0shot_bpb",
    # "socialiqa_rc_0shot_bpb",
    "mmlu_stem_var_bpb",
    "mmlu_humanities_var_bpb",
    "mmlu_social_sciences_var_bpb",
    "mmlu_other_var_bpb",
    "mmlu_stem_bpb",
    "mmlu_humanities_bpb",
    "mmlu_social_sciences_bpb",
    "mmlu_other_bpb",
]

v3_validation = [
    "v3-small-c4_en-validation",
    "v3-small-dolma_books-validation",
    "v3-small-dolma_common-crawl-validation",
    "v3-small-dolma_pes2o-validation",
    "v3-small-dolma_reddit-validation",
    "v3-small-dolma_stack-validation",
    "v3-small-dolma_wiki-validation",
    "v3-small-ice-validation",
    "v3-small-m2d2_s2orc-validation",
    #'v3-small-pile-validation',
    "v3-small-wikitext_103-validation",
]

downstream = [
    "hellaswag_len_norm",
    "winogrande_acc",
    "piqa_len_norm",
    "social_iqa_len_norm",
    "openbook_qa_len_norm",
    "commonsense_qa_len_norm",
    "boolq_acc",
    "copa_acc",
    "arc_easy_acc",
    "arc_challenge_len_norm",
    "sciq_acc",
    "mmlu_social_sciences_var_len_norm",
    "mmlu_humanities_var_len_norm",
    "mmlu_other_var_len_norm",
    "mmlu_stem_mc_5shot_test_len_norm",
    "mmlu_humanities_mc_5shot_len_norm",
    "mmlu_social_sciences_mc_5shot_len_norm",
    "mmlu_stem_var_len_norm",
    "mmlu_other_mc_5shot_test_len_norm",
    "mmlu_humanities_mc_5shot_test_len_norm",
    "mmlu_stem_mc_5shot_len_norm",
    "mmlu_social_sciences_mc_5shot_test_len_norm",
    "mmlu_other_mc_5shot_len_norm",
]


# Power Law functions


def openai_fit(x, a, b, c):
    return (a / x + c) ** b


def chinchilla_fit(x, a, b, c):
    return a * x**b + c


def chinchilla_contaminated_fit(x, a, b, c, d):
    return (a * x**b + c) * (1 - x / d)


# Scipy curve_fit with least squares
def get_coefficients(train_xs, train_ys, fitting_func, p0):
    if type(train_xs[0]) == list:
        train_xs = np.array(train_xs).transpose()
    coeffs = scipy.optimize.curve_fit(fitting_func, train_xs, train_ys, p0=p0, maxfev=50000)[0]
    coeffs_string = ", ".join([chr(ord("a") + i) + f" = {coeffs[i]:.2f}" for i in range(len(coeffs))])
    print(f"{fitting_func.__name__}: {coeffs_string}")
    return coeffs


# x[0] = d, x[1] = h
# p[0] = b = log100(B), p[1] = beta, p[2] = E, p[3] = F
def chinchilla_d_lr_fit(x, p):
    return 100**p[0] / x[0]**p[1] + p[2] + p[3] * x[1]
def grad_chinchilla_d_lr_fit(x, p):
    grad_b = (1 / x[0]**p[1]) * (100**p[0] * np.log(100))
    grad_beta = -(100**p[0]) * np.log(x[0]) / x[0]**p[1]
    grad_E = 1
    grad_F = x[1]
    return [grad_b, grad_beta, grad_E, grad_F]


# x[0] = n, x[1] = d
# p[0] = a = log100(A), p[1] = b = log100(B), p[2] = alpha, p[3] = beta, p[4] = E
def chinchilla_n_d_fit(x, p):
    # return 100**a / x[0]**alpha + 100**b / x[1]**beta + E
    return 100**p[0] / x[0]**p[2] + 100**p[1] / x[1]**p[3] + p[4]
def grad_chinchilla_n_d_fit(x, p):
    grad_a = (1 / x[0]**p[2]) * (100**p[0] * np.log(100))
    grad_b = (1 / x[1]**p[3]) * (100**p[1] * np.log(100))
    grad_alpha = - (100**p[0]) * np.log(x[0]) / x[0]**p[2]
    grad_beta = - (100**p[1]) * np.log(x[1]) / x[1]**p[3]
    grad_E = 1
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E]


# x[0] = n, x[1] = d, x[2] = h
# p[0] = a = log100(A), p[1] = b = log100(B), p[2] = alpha, p[3] = beta, p[4] = E, p[5] = F
def chinchilla_n_d_lr_fit(x, p):
    # return 100**a / x[0]**alpha + 100**b / x[1]**beta + E + F * x[2]
    return 100**p[0] / x[0]**p[2] + 100**p[1] / x[1]**p[3] + p[4] + p[5] * x[2]
def grad_chinchilla_n_d_lr_fit(x, p):
    grad_a = (1 / x[0]**p[2]) * (100**p[0] * np.log(100))
    grad_b = (1 / x[1]**p[3]) * (100**p[1] * np.log(100))
    grad_alpha = - (100**p[0]) * np.log(x[0]) / x[0]**p[2]
    grad_beta = - (100**p[1]) * np.log(x[1]) / x[1]**p[3]
    grad_E = 1
    grad_F = x[2]
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F]

def chinchilla_n_d_lr_log_fit(x, p):
    # return 100**a / x[0]**alpha + 100**b / x[1]**beta + E + F * x[2] * np.log(x[0] / 100**r + s)
    return np.exp(p[0]) / x[0]**p[2] + np.exp(p[1]) / x[1]**p[3] + p[4] + p[5] * x[2] * np.log(x[0] / np.exp(p[6]) + p[7])
def grad_chinchilla_n_d_lr_log_fit(x, p):
    grad_a = np.exp(p[0]) / x[0]**p[2]
    grad_b = np.exp(p[1]) / x[1]**p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0]**p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1]**p[3]
    grad_E = 1
    grad_F = x[2] * np.log(x[0] / np.exp(p[6]) + p[7])
    grad_r = p[5] * x[2] * (1 / (x[0] / np.exp(p[6]) + p[7])) * x[0] * (-1 / np.exp(p[6]))
    grad_s = p[5] * x[2] * (1 / (x[0] / np.exp(p[6]) + p[7]))
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r, grad_s]

def chinchilla_n_d_lr_power_fit(x, p):
    # return 100**a / x[0]**alpha + 100**b / x[1]**beta + E + F * x[2] * x[0]**r
    return np.exp(p[0]) / x[0]**p[2] + np.exp(p[1]) / x[1]**p[3] + p[4] + p[5] * x[2] * x[0]**p[6]
def grad_chinchilla_n_d_lr_power_fit(x, p):
    grad_a = np.exp(p[0]) / x[0]**p[2]
    grad_b = np.exp(p[1]) / x[1]**p[3]
    grad_alpha = np.exp(p[0]) * (-np.log(x[0])) / x[0]**p[2]
    grad_beta = np.exp(p[1]) * (-np.log(x[1])) / x[1]**p[3]
    grad_E = 1
    grad_F = x[2] * x[0]**p[6]
    grad_r = p[5] * x[2] * x[0]**p[6] * np.log(x[0])
    return [grad_a, grad_b, grad_alpha, grad_beta, grad_E, grad_F, grad_r]


# Scipy minimize w/ Huber loss
def get_coefficients_huber(train_xs, train_ys, fitting_func, grad_func, p0, bounds):

    def huber_loss(x, delta):
        if np.abs(x) < delta:
            return 0.5 * x**2
        else:
            return delta * (np.abs(x) - 0.5 * delta)

    def loss_fn(p, train_xs, train_ys, delta):
        actuals = train_ys
        preds = [fitting_func(x, p) for x in train_xs]
        loss = np.sum([huber_loss(np.log(pred) - np.log(actual), delta=delta) for actual, pred in zip(actuals, preds)])
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
    res = scipy.optimize.minimize(loss_fn, p0, args=(train_xs, train_ys, delta), jac=jac_fn, bounds=bounds, tol=0.0, method='L-BFGS-B', options={'ftol': 0.0, 'gtol': 1e-10, 'maxiter': 10000, 'disp': True})
    # res = scipy.optimize.minimize(loss_fn, p0, args=(train_xs, train_ys, delta), jac=jac_fn, tol=0.0, method='BFGS', options={'gtol': 1e-10, 'maxiter': 10000, 'disp': True})
    # print(res.message)
    coeffs = res.x
    print(f'coeffs: {coeffs}')
    return coeffs
