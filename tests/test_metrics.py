import numpy as np

from omari_us.evaluation import metrics as M


def test_report_known_values():
    y = np.array([0, 0, 1, 1, 1, 2])
    p = np.array([0, 1, 1, 1, 2, 2])
    r = M.classification_report(y, p, k=3)
    assert abs(r["accuracy"] - 4 / 6) < 1e-12
    assert abs(r["per_class"][0]["recall"] - 0.5) < 1e-12
    assert abs(r["per_class"][1]["precision"] - 2 / 3) < 1e-12


def test_ece_zero_for_perfect_onehot():
    probs = np.eye(3)[[0, 1, 2, 1]]
    assert M.reliability(probs, np.array([0, 1, 2, 1]))["ece"] < 1e-12


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 4000, 6
    y = rng.integers(0, k, n)
    z = rng.normal(0, 1, (n, k))
    z[np.arange(n), y] += 1.5
    true_p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    # sample labels from the calibrated distribution, then sharpen the reported probabilities (T = 0.5)
    y = np.array([rng.choice(k, p=pp) for pp in true_p])
    sharp = M.apply_temperature(true_p, 0.5)
    t = M.fit_temperature(sharp, y)
    assert 1.6 < t < 2.4  # ≈ 2 undoes the sharpening
    assert M.reliability(M.apply_temperature(sharp, t), y)["ece"] < M.reliability(sharp, y)["ece"]


def test_cluster_bootstrap_wider_than_iid():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(60), 10)
    good = rng.random(60) < 0.7  # patient-level effect → strong clustering
    y = np.zeros(600, int)
    p = np.where(np.repeat(good, 10), 0, 1)
    iid = M.bootstrap_ci(y, p, None, n_boot=300)
    clu = M.bootstrap_ci(y, p, groups, n_boot=300)
    w = lambda ci: ci["accuracy"][1] - ci["accuracy"][0]  # noqa: E731
    assert w(clu) > w(iid)


def test_mcnemar_and_holm():
    y = np.zeros(100, int)
    a = np.zeros(100, int)
    b = np.r_[np.ones(20, int), np.zeros(80, int)]
    t = M.mcnemar(y, a, b)
    assert t["b_only_a_correct"] == 20 and t["p_value"] < 1e-4
    assert M.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
