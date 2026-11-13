"""
Fit and honestly evaluate a learned wildfire hazard model against the hand-weighted formula in
priority_surface.py and against "where it burned before".

Reads  data/hazard_training_500m.npz (do not modify)
Writes data/hazard_model.json, data/hazard_validation.json, data/HAZARD_VALIDATION.md

Needs scikit-learn (requirements-data.txt only; the runtime predictor hazard_model.py does not).
Protocol is fixed in the module constants below and was set before looking at any result.
Run from the repo root:  MPLBACKEND=Agg ../.venv/bin/python data/fit_hazard_model.py   (about 2 minutes)
"""

import json
import os
import sys
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import hazard_model as hm            # noqa: E402
import priority_surface as ps        # noqa: E402

SEED = 20261113
N_BOOT = 500
BLOCK = 20                           # cells per spatial block side (10 km)
C_GRID = np.logspace(-4, 1, 11)
Y0 = 2012
TRAIN = (2012, 2019)
TEST = (2020, 2024)                  # 2025 is a partial year and is not scored
TEST_EX2020 = (2021, 2024)
HIST_FIT = (2012, 2015)              # model e: history window ...
LABEL_FIT = (2016, 2019)             # ... and the label it is fitted against
HIST_TEST = (2016, 2019)             # history used when scoring the test period
DEPLOY_HIST = (2012, 2018)
DEPLOY_LABEL = (2019, 2025)
DEPLOY_PROD_WINDOW = (2012, 2025)

METHODS = ["random", "hand", "persist", "persist4", "lr_static", "lr_hist", "lr_hist_alt", "hgb"]
LABELS = {
    "random": "Random (expected)",
    "hand": "Hand formula (fires 2012-2019 only)",
    "persist": "Persistence (smoothed 2012-2019 count, 8 years)",
    "persist4": "Persistence, 2016-2019 only (4 years)",
    "lr_static": "Logistic, static features",
    "lr_hist": "Logistic + fire history, 2016-2019 (primary)",
    "lr_hist_alt": "Logistic + fire history, 2012-2019 as 4-year rate",
    "hgb": "Gradient boosting + history (check only)",
}
METRICS = ["auc", "ap", "cap10", "cap20"]
METRIC_LABELS = {"auc": "ROC AUC", "ap": "Avg precision", "cap10": "Capture 10%", "cap20": "Capture 20%"}

# (phrase when the standardised coefficient is positive, phrase when negative)
READING = {
    "elevation_m": ("higher ground", "lower ground"),
    "slope_deg": ("steeper slopes", "flatter ground"),
    "northness_x_slope": ("north-facing slopes", "south-facing slopes"),
    "eastness_x_slope": ("east-facing slopes", "west-facing slopes"),
    "lc_tree": ("more tree cover", "less tree cover"),
    "lc_shrub": ("more shrub", "less shrub"),
    "lc_grass": ("more grass", "less grass"),
    "lc_crop": ("more cropland in the cell", "less cropland in the cell"),
    "lc_built": ("more built-up land", "less built-up land"),
    "lc_bare": ("more bare ground", "less bare ground"),
    "lc_water": ("more open water", "less open water"),
    "inside_park": ("being inside the park", "being outside the park"),
    "signed_dist_park_edge_km": ("being farther outside the park boundary (or less deep inside)",
                                 "being deeper inside the park (or closer to its boundary from outside)"),
    "log1p_dist_road": ("being farther from any road or track", "being closer to a road or track"),
    "log1p_dist_major_road": ("being farther from a major road", "being closer to a major road"),
    "log1p_dist_settlement": ("being farther from a settlement", "being closer to a settlement"),
    "crop_share_1p5km": ("more cropland within about 1.5 km", "less cropland within about 1.5 km"),
    "fire_history": ("more detections in the history window", "fewer detections in the history window"),
}


# --------------------------------------------------------------------------- data

def load():
    z = np.load(os.path.join(HERE, "hazard_training_500m.npz"))
    arrays = {k: z[k] for k in z.files}
    years = arrays["years"].astype(int)
    assert list(years) == list(range(2012, 2026))
    return arrays


def counts(arrays, y0, y1):
    f = arrays["fire_by_year"].astype(np.float64)
    return f[y0 - Y0: y1 - Y0 + 1].sum(axis=0)


def subsets(arrays):
    rows, cols = arrays["dem_m"].shape
    rr, cc = np.indices((rows, cols))
    inside = arrays["inside_park"] > 0.5
    reg = np.zeros((rows, cols), bool)
    reg[80:120, 80:120] = True
    dash = np.zeros((rows, cols), bool)
    dash[80 + 15:80 + 35, 80 + 3:80 + 23] = True
    m = {
        "whole": np.ones((rows, cols), bool),
        "park": inside,
        "band5km": arrays["dist_park_edge_m"] <= 5000.0,
        "region40": reg,
        "dashboard20": dash,
    }
    block = (rr // BLOCK) * (cols // BLOCK) + (cc // BLOCK)
    return {k: v.ravel() for k, v in m.items()}, block.ravel(), (cc.ravel())


SUBSET_LABELS = {
    "whole": "Whole grid (200 x 200 cells)",
    "park": "Park cells (inside_park > 0.5)",
    "band5km": "Within 5 km of the park boundary, either side",
    "region40": "Existing 40 x 40 region",
    "dashboard20": "Dashboard 20 x 20 window (small, noisy)",
}


# --------------------------------------------------------------------------- metrics

_trap = getattr(np, "trapezoid", None) or np.trapz


def metrics(s, c):
    """[AUC, average precision, capture at 10%, capture at 20%]; ties handled exactly.
    Capture counts detections (not cells); tied scores at the cut share the remaining slots."""
    y = c > 0
    n = len(s)
    P = int(y.sum())
    if P == 0 or P == n:
        return np.full(4, np.nan)
    o = np.argsort(-s, kind="stable")
    ss, yy, cc = s[o], y[o], c[o]
    tp = np.cumsum(yy)
    idx = np.r_[np.nonzero(ss[1:] != ss[:-1])[0], n - 1]
    tpi = tp[idx]
    fpi = idx + 1 - tpi
    rec = tpi / P
    ap = float(np.sum(np.diff(np.r_[0.0, rec]) * (tpi / (idx + 1))))
    auc = float(_trap(np.r_[0.0, rec], np.r_[0.0, fpi / (n - P)]))
    ccs = np.cumsum(cc)
    total = ccs[-1]
    out = [auc, ap]
    neg = -ss
    for q in (0.10, 0.20):
        k = int(round(q * n))
        tv = neg[k - 1]
        lo = np.searchsorted(neg, tv, "left")
        hi = np.searchsorted(neg, tv, "right")
        above = ccs[lo - 1] if lo > 0 else 0.0
        grp = ccs[hi - 1] - above
        out.append((above + grp * (k - lo) / (hi - lo)) / total)
    return np.array(out)


def method_metrics(name, s, c):
    if name == "random":
        y = c > 0
        if y.sum() == 0:
            return np.full(4, np.nan)
        return np.array([0.5, y.mean(), 0.10, 0.20])
    return metrics(s, c)


# --------------------------------------------------------------------------- models

def block_folds(block_ids, rng, k=5):
    ub = np.unique(block_ids)
    perm = rng.permutation(ub)
    fold_of = {b: i % k for i, b in enumerate(perm)}
    return np.array([fold_of[b] for b in block_ids])


def fit_lr(X, y, block_ids, seed):
    """L2 logistic regression on standardised features; C by 5-fold spatial-block CV (log loss),
    using only the rows passed in (the training period)."""
    folds = block_folds(block_ids, np.random.default_rng(seed))
    losses = []
    for C in C_GRID:
        tot = 0.0
        for f in range(5):
            tr, va = folds != f, folds == f
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
            tot += log_loss(y[va], m.predict_proba(sc.transform(X[va]))[:, 1], labels=[0, 1]) * va.sum()
        losses.append(tot / len(y))
    C = float(C_GRID[int(np.argmin(losses))])
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return dict(sc=sc, m=m, C=C, cv_logloss=[float(v) for v in losses])


def lr_score(fit, X):
    return fit["m"].predict_proba(fit["sc"].transform(X))[:, 1]


def fit_hgb(X, y):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=100, l2_regularization=1.0,
                                       early_stopping=False, random_state=0)
    return m.fit(X, y)


def export_lr(fit, names, train_period, label_period, history, samples, note):
    return dict(
        note=note,
        features=names,
        mean=[float(v) for v in fit["sc"].mean_],
        scale=[float(v) for v in fit["sc"].scale_],
        coef=[float(v) for v in fit["m"].coef_[0]],
        intercept=float(fit["m"].intercept_[0]),
        C=fit["C"],
        training_period=dict(history_years=list(train_period) if train_period else None,
                             label_years=list(label_period)),
        history=history,
        samples=samples,
    )


# --------------------------------------------------------------------------- main

def main():
    t0 = time.time()
    arrays = load()
    shape = arrays["dem_m"].shape
    N = shape[0] * shape[1]
    sub, block, colidx = subsets(arrays)

    Xs, names = hm.build_features(arrays)
    assert Xs.shape == (N, len(names)) and np.isfinite(Xs).all()
    names_h = names + [hm.HISTORY_FEATURE]

    def with_hist(counts2d):
        return np.concatenate([Xs, hm.history_feature(counts2d).reshape(-1, 1)], axis=1)

    c_train = counts(arrays, *TRAIN)
    c_label = c_train.ravel()
    y_train = (c_label > 0).astype(int)
    lab_fit = counts(arrays, *LABEL_FIT).ravel()
    y_fit = (lab_fit > 0).astype(int)
    test = {"main": counts(arrays, *TEST).ravel(), "ex2020": counts(arrays, *TEST_EX2020).ravel()}

    # ---- score sources
    scores = {}
    data = dict(arrays)
    data["fire_count"] = c_train
    scores["hand"] = ps._real_layers(data, dict(ps.REAL_DEFAULT_WEIGHTS), 500.0)["hazard"].ravel()
    scores["persist"] = hm.history_feature(c_train).ravel()
    scores["persist4"] = hm.history_feature(counts(arrays, *HIST_TEST)).ravel()

    X_hist_fit = with_hist(counts(arrays, *HIST_FIT))
    X_hist_test = with_hist(counts(arrays, *HIST_TEST))
    X_hist_alt = with_hist(c_train * (4.0 / 8.0))

    fit_d = fit_lr(Xs, y_train, block, SEED + 1)
    fit_e = fit_lr(X_hist_fit, y_fit, block, SEED + 2)
    hgb = fit_hgb(X_hist_fit, y_fit)
    scores["lr_static"] = lr_score(fit_d, Xs)
    scores["lr_hist"] = lr_score(fit_e, X_hist_test)
    scores["lr_hist_alt"] = lr_score(fit_e, X_hist_alt)
    scores["hgb"] = hgb.predict_proba(X_hist_test)[:, 1]
    scores["random"] = None
    print(f"fits done {time.time()-t0:.0f}s; C static={fit_d['C']:.4g}, hist={fit_e['C']:.4g}")

    # ---- sanity: own metrics agree with scikit-learn
    yt = (test["main"] > 0).astype(int)
    for k in ["hand", "lr_hist"]:
        mm = metrics(scores[k], test["main"])
        assert abs(mm[0] - roc_auc_score(yt, scores[k])) < 1e-9, k
        assert abs(mm[1] - average_precision_score(yt, scores[k])) < 1e-9, k

    # ---- point estimates
    def table(score_map, cnt, mask):
        return {m: dict(zip(METRICS, [None if np.isnan(v) else float(v) for v in
                                      method_metrics(m, None if score_map.get(m) is None else score_map[m][mask], cnt[mask])]))
                for m in METHODS if m in score_map}

    point = {per: {sk: table(scores, test[per], sm) for sk, sm in sub.items()} for per in test}

    # ---- bootstrap over 20 x 20 spatial blocks
    rng = np.random.default_rng(SEED + 3)
    boot = {}
    boot_methods = METHODS
    for sk in ["whole", "park"]:
        cells = np.nonzero(sub[sk])[0]
        by_block = {b: cells[block[cells] == b] for b in np.unique(block[cells])}
        keys = list(by_block)
        res = {per: np.full((N_BOOT, len(boot_methods), 4), np.nan) for per in test}
        for r in range(N_BOOT):
            pick = rng.integers(0, len(keys), len(keys))
            idx = np.concatenate([by_block[keys[i]] for i in pick])
            for per in test:
                for j, m in enumerate(boot_methods):
                    s = None if scores[m] is None else scores[m][idx]
                    res[per][r, j] = method_metrics(m, s, test[per][idx])
        boot[sk] = res
        print(f"bootstrap {sk} done {time.time()-t0:.0f}s")

    def ci(a):
        a = a[~np.isnan(a)]
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    intervals = {per: {sk: {m: {met: ci(boot[sk][per][:, j, i]) for i, met in enumerate(METRICS)}
                            for j, m in enumerate(boot_methods)} for sk in boot} for per in test}
    pairs = [("lr_hist", "persist"), ("lr_hist", "hand"), ("lr_hist", "lr_static"),
             ("lr_hist", "lr_hist_alt"), ("persist", "hand"), ("hgb", "lr_hist"),
             ("lr_hist", "persist4"), ("lr_hist_alt", "persist"), ("hgb", "persist")]
    diffs = {}
    for per in test:
        diffs[per] = {}
        for sk in boot:
            diffs[per][sk] = {}
            for a, b in pairs:
                ja, jb = boot_methods.index(a), boot_methods.index(b)
                d = {}
                for i, met in enumerate(METRICS):
                    pt = point[per][sk][a][met] - point[per][sk][b][met]
                    arr = boot[sk][per][:, ja, i] - boot[sk][per][:, jb, i]
                    lo, hi = ci(arr)
                    d[met] = dict(point=float(pt), lo=lo, hi=hi,
                                  share_positive=float(np.mean(arr[~np.isnan(arr)] > 0)))
                diffs[per][sk][f"{a}_minus_{b}"] = d

    # ---- spatial check: fit on one half, score the other half's test period
    spatial = {}
    for tr_name, te_name, tr_mask, te_mask in [
            ("west", "east", colidx < 100, colidx >= 100),
            ("east", "west", colidx >= 100, colidx < 100)]:
        fd = fit_lr(Xs[tr_mask], y_train[tr_mask], block[tr_mask], SEED + 11)
        fe = fit_lr(X_hist_fit[tr_mask], y_fit[tr_mask], block[tr_mask], SEED + 12)
        sc = dict(scores)
        sc["lr_static"] = lr_score(fd, Xs)
        sc["lr_hist"] = lr_score(fe, X_hist_test)
        sc.pop("lr_hist_alt"), sc.pop("hgb")
        out = {}
        for sk in ["whole", "park"]:
            mask = te_mask & sub[sk]
            out[sk] = table(sc, test["main"], mask)
            out[sk]["_n_cells"] = int(mask.sum())
            out[sk]["_n_detections"] = int(test["main"][mask].sum())
        spatial[f"train_{tr_name}_test_{te_name}"] = out

    def spatial_verdict():
        v = {}
        for k, o in spatial.items():
            for sk in ["whole", "park"]:
                t = o[sk]
                v[f"{k}/{sk}"] = dict(
                    e_beats_persistence_ap=t["lr_hist"]["ap"] > t["persist"]["ap"],
                    e_beats_persistence_cap20=t["lr_hist"]["cap20"] > t["persist"]["cap20"],
                    e_beats_hand_ap=t["lr_hist"]["ap"] > t["hand"]["ap"],
                    e_beats_hand_cap20=t["lr_hist"]["cap20"] > t["hand"]["cap20"],
                    order_by_ap=[m for m in sorted(["hand", "persist", "lr_static", "lr_hist"],
                                                   key=lambda m: -t[m]["ap"])])
        return v
    spatial_v = spatial_verdict()

    # ---- hand formula normalised over the 40 x 40 region (as in production) instead of the 200 x 200 grid
    r40 = slice(80, 120)
    reg = {k: v[r40, r40] for k, v in arrays.items() if np.ndim(v) == 2}
    reg["fire_count"] = c_train.reshape(shape)[r40, r40]
    hand_reg = ps._real_layers(reg, dict(ps.REAL_DEFAULT_WEIGHTS), 500.0)["hazard"].ravel()
    cnt40 = test["main"].reshape(shape)[r40, r40].ravel()
    m_reg = metrics(hand_reg, cnt40)
    hand_region = dict(
        grid_normalised_ap=point["main"]["region40"]["hand"]["ap"],
        region_normalised_ap=float(m_reg[1]),
        grid_normalised_cap20=point["main"]["region40"]["hand"]["cap20"],
        region_normalised_cap20=float(m_reg[3]),
        persistence_ap=point["main"]["region40"]["persist"]["ap"],
        persistence_cap20=point["main"]["region40"]["persist"]["cap20"])

    # ---- coefficients
    coefs = {
        "static": dict(zip(names, map(float, fit_d["m"].coef_[0]))),
        "history": dict(zip(names_h, map(float, fit_e["m"].coef_[0]))),
    }

    # ---- deployment model
    X_dep = with_hist(counts(arrays, *DEPLOY_HIST))
    y_dep = (counts(arrays, *DEPLOY_LABEL).ravel() > 0).astype(int)
    fit_dep = fit_lr(X_dep, y_dep, block, SEED + 4)
    coefs["deployment"] = dict(zip(names_h, map(float, fit_dep["m"].coef_[0])))

    # ---- export models with regression samples (numpy predictor, checked against sklearn)
    srng = np.random.default_rng(SEED + 5)
    park_cells = np.nonzero(sub["park"])[0]
    sample_idx = np.r_[srng.choice(park_cells, 5, replace=False),
                       srng.choice(N, 7, replace=False)]
    hist_validated = dict(first_year=HIST_TEST[0], last_year=HIST_TEST[1],
                          rescale_to_years=HIST_FIT[1] - HIST_FIT[0] + 1)
    hist_deploy = dict(first_year=DEPLOY_PROD_WINDOW[0], last_year=DEPLOY_PROD_WINDOW[1],
                       rescale_to_years=DEPLOY_HIST[1] - DEPLOY_HIST[0] + 1)
    models = {
        "validated": (fit_e, names_h, HIST_FIT, LABEL_FIT, hist_validated, X_hist_test,
                      "Model e as validated: fitted with history 2012-2015 against label 2016-2019; "
                      "score with history 2016-2019. Output ranks cells by the chance of at least "
                      "one detection in a period of about four years."),
        "deployment": (fit_dep, names_h, DEPLOY_HIST, DEPLOY_LABEL, hist_deploy, None,
                       "Same specification refitted with history 2012-2018 against label 2019-2025 "
                       "(seven years). Give predict the whole 2012-2025 record rescaled to seven "
                       "years. Not itself validated out of sample."),
        "static": (fit_d, names, None, TRAIN, None, Xs,
                   "Model d: static features only, fitted against the 2012-2019 label."),
    }
    exported = {}
    for key, (fit, nm, hper, lper, hist, _, note) in models.items():
        entry = export_lr(fit, nm, hper, lper, hist, [], note)
        p = hm.predict(arrays, entry)
        # reference: scikit-learn on the same matrix
        if key == "static":
            ref = lr_score(fit, Xs)
        else:
            hc = hm.history_counts(arrays, entry)
            ref = lr_score(fit, with_hist(hc))
        assert np.abs(p.ravel() - ref).max() < 1e-9, key
        entry["samples"] = [dict(row=int(i // shape[1]), col=int(i % shape[1]),
                                 p=float(p.ravel()[i])) for i in sample_idx]
        exported[key] = entry
    # the validated export must reproduce the score used in the test
    assert np.abs(hm.predict(arrays, exported["validated"]).ravel() - scores["lr_hist"]).max() < 1e-9

    model_doc = dict(
        format=1,
        description="Logistic hazard models for the 500 m Khao Yai grid. See data/HAZARD_VALIDATION.md.",
        grid=dict(shape=list(shape), cell_size_m=500.0, built_by="data/fit_hazard_model.py"),
        models=exported)
    with open(os.path.join(HERE, "hazard_model.json"), "w") as f:
        json.dump(model_doc, f, indent=1)

    # ---- HGB check on non-linearity is in point/boot; collect prevalence facts
    facts = dict(
        cells=N, park_cells=int(sub["park"].sum()),
        train_detections=int(c_train.sum()), test_detections=int(test["main"].sum()),
        test_detections_ex2020=int(test["ex2020"].sum()),
        detections_2020=int(arrays["fire_by_year"][2020 - Y0].sum()),
        test_detections_park=int(test["main"][sub["park"]].sum()),
        test_prevalence=float((test["main"] > 0).mean()),
        test_prevalence_park=float((test["main"][sub["park"]] > 0).mean()),
        train_prevalence=float(y_train.mean()),
        detections_2025=int(arrays["fire_by_year"][-1].sum()),
    )
    doc = dict(
        protocol=dict(
            unit="one 500 m cell; label = at least one VIIRS detection in the period",
            train=list(TRAIN), test=list(TEST), test_excluding_2020=list(TEST_EX2020),
            model_e_history_fit=list(HIST_FIT), model_e_label_fit=list(LABEL_FIT),
            model_e_history_test=list(HIST_TEST), deployment_history=list(DEPLOY_HIST),
            deployment_label=list(DEPLOY_LABEL), n_features_static=len(names),
            bootstrap=dict(resamples=N_BOOT, unit=f"{BLOCK}x{BLOCK}-cell blocks", seed=SEED + 3),
            cv=dict(folds=5, unit="20x20-cell blocks", selection="log loss", c_grid=[float(c) for c in C_GRID]),
            sklearn_version=sklearn.__version__),
        facts=facts, methods=LABELS, chosen_C=dict(static=fit_d["C"], history=fit_e["C"], deployment=fit_dep["C"]),
        hand_formula_region_normalisation=hand_region,
        point=point, intervals=intervals, paired_differences=diffs, spatial_check=spatial,
        spatial_verdict=spatial_v, coefficients=coefs,
        subset_cells={k: int(v.sum()) for k, v in sub.items()},
    )
    with open(os.path.join(HERE, "hazard_validation.json"), "w") as f:
        json.dump(doc, f, indent=1)
    write_md(doc)
    print(f"done in {time.time()-t0:.0f}s")


# --------------------------------------------------------------------------- write-up

def fnum(v, pct=False, d=3):
    if v is None:
        return "n/a"
    return f"{v*100:.1f}%" if pct else f"{v:.{d}f}"


def cell(doc, per, sk, m, met):
    p = doc["point"][per][sk][m][met]
    pct = met.startswith("cap")
    if sk in doc["intervals"][per] and m != "random":
        lo, hi = doc["intervals"][per][sk][m][met]
        return f"{fnum(p, pct)} ({fnum(lo, pct)} to {fnum(hi, pct)})"
    return fnum(p, pct)


def headline_table(doc, per, sk):
    head = "| Method | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |\n"
    head += "|---|" + "---|" * len(METRICS) + "\n"
    rows = ""
    for m in METHODS:
        rows += f"| {LABELS[m]} | " + " | ".join(cell(doc, per, sk, m, k) for k in METRICS) + " |\n"
    return head + rows


def diff_text(d, met):
    pct = met.startswith("cap")
    s = 100 if pct else 1
    unit = " points" if pct else ""
    fmt = lambda v: f"{v*s:+.1f}" if pct else f"{v:+.3f}"
    return f"{fmt(d['point'])}{unit} ({fmt(d['lo'])} to {fmt(d['hi'])})"


MEASURES = {"ap": "average precision", "cap20": "capture at 20%"}


def verdict(d):
    """'clear' if both AP and capture-at-20% intervals sit above 0, 'worse' if both below 0,
    'none' if both include 0, else 'mixed'."""
    pos = [d[m]["lo"] > 0 for m in MEASURES]
    neg = [d[m]["hi"] < 0 for m in MEASURES]
    if all(pos):
        return "clear"
    if all(neg):
        return "worse"
    if not any(pos) and not any(neg):
        return "none"
    return "mixed"


def describe(d, measures=None):
    """Plain description of one paired difference (positive = first-named method ahead)."""
    measures = measures or MEASURES
    better = [measures[m] for m in measures if d[m]["lo"] > 0]
    worse = [measures[m] for m in measures if d[m]["hi"] < 0]
    tied = [measures[m] for m in measures if d[m]["lo"] <= 0 <= d[m]["hi"]]
    parts = []
    if better:
        parts.append("clearly better on " + " and ".join(better))
    if worse:
        parts.append("clearly worse on " + " and ".join(worse))
    if tied:
        parts.append("not clearly different on " + " and ".join(tied))
    return "; ".join(parts)


def coef_lines(coefs, k=6):
    out = []
    for n, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:k]:
        pos, _ = READING[n]
        out.append(f"- `{n}` {v:+.2f}: {'more' if v > 0 else 'fewer'} detections with {pos}.")
    return "\n".join(out)


MEASURES3 = {"auc": "ROC AUC", "ap": "average precision", "cap20": "capture at 20%"}


def pair_line(label, d, measures=MEASURES):
    nums = ", ".join(f"{name} {diff_text(d[m], m)}" for m, name in measures.items())
    desc = describe(d, measures)
    return f"{label}: {nums}. {desc[0].upper() + desc[1:]}."


def write_md(doc):
    P, W = "park", "whole"
    PD = doc["paired_differences"]
    F = doc["facts"]
    PERIODS = [("main", "2020 to 2024"), ("ex2020", "2021 to 2024, with 2020 excluded")]
    ep, ep4, eh = "lr_hist_minus_persist", "lr_hist_minus_persist4", "lr_hist_minus_hand"
    lines = []
    a = lines.append
    a("# Learned hazard model: validation against the hand formula and persistence\n")
    a("Generated by `data/fit_hazard_model.py` on 13 November 2026 (rerun to reproduce; seeds are fixed). "
      "Machine-readable numbers: `data/hazard_validation.json`.\n")
    a("## What was tested\n")
    a("Can a fitted model rank 500 m cells by where satellites will detect fires next better than the "
      "project's hand-weighted hazard formula, or better than simply using where it burned before? "
      "Every method saw fire detections from 2012 to 2019 only and was scored on 2020 to 2024.\n")
    a("- **Persistence** is the where-it-burned-before score: the smoothed count of earlier detections "
      "in and around each cell.\n"
      "- **Capture at 20%** is the share of later detections that fall in the 20% of cells a method "
      "ranks highest; a random ranking captures 20%.\n"
      "- **Average precision** and **ROC AUC** are standard ranking scores, higher is better; "
      "a random ranking scores 0.5 on ROC AUC.\n"
      "- \"Clearly\" means the 95% interval of the difference excludes zero.\n")
    a("## Answer\n")

    D = PD["main"]
    v = {sk: verdict(D[sk][ep]) for sk in (P, W)}
    if all(x == "clear" for x in v.values()):
        lead = ("**Yes: the logistic model with fire history beats persistence over the whole grid and in the park, "
                "outside the bootstrap intervals (2020 to 2024).**")
    else:
        whole = "tie it" if v[W] == "none" else f"are {describe(D[W][ep])}"
        lead = (f"**The logistic models do not beat persistence (2020 to 2024). Over the whole grid they {whole}; "
                f"in the park they are {describe(D[P][ep]).replace('; ', ' and ')}.** Gradient boosting against persistence: whole grid "
                f"{describe(PD['main'][W]['hgb_minus_persist'], MEASURES3)}; park "
                f"{describe(PD['main'][P]['hgb_minus_persist'], MEASURES3)}.")
    a(lead + "\n")
    X = PD["ex2020"]
    a("With 2020 excluded (2021 to 2024): logistic with history against persistence, park "
      f"{describe(X[P][ep])}, whole grid {describe(X[W][ep])}; boosting against persistence, park "
      f"{describe(X[P]['hgb_minus_persist'], MEASURES3)}, whole grid {describe(X[W]['hgb_minus_persist'], MEASURES3)}.\n")
    a("Persistence against the hand formula, which is scoped by period: in the park, 2020 to 2024, it is "
      f"{describe(PD['main'][P]['persist_minus_hand'])}; 2021 to 2024, {describe(X[P]['persist_minus_hand'])}. "
      f"On the whole grid, 2020 to 2024, {describe(PD['main'][W]['persist_minus_hand'])}; 2021 to 2024, "
      f"{describe(X[W]['persist_minus_hand'])}.\n")
    a("The primary learned model sees four years of fire history (2016 to 2019); persistence and the hand formula "
      "see eight (2012 to 2019). The like-for-like comparison is therefore also shown against persistence built from "
      "2016 to 2019 only, and the eight-year variant of the learned model against ordinary persistence. "
      "In every line below the first-named method is the one subtracted from; positive favours it. Capture "
      "differences are in percentage points.\n")
    for per, pname in PERIODS:
        D = PD[per]
        a(f"### Test {pname}\n")
        for sk, nm in [(P, "Inside the park"), (W, "Whole grid")]:
            a(f"**{nm}**\n")
            a("- " + pair_line("Learned minus persistence (8 years)", D[sk][ep]))
            a("- " + pair_line("Learned minus persistence (2016 to 2019 only, same 4 years)", D[sk][ep4]))
            a("- " + pair_line("Learned, 8-year variant, minus persistence (8 years)", D[sk]["lr_hist_alt_minus_persist"]))
            a("- " + pair_line("Learned minus hand formula", D[sk][eh]))
            a("- " + pair_line("Persistence (8 years) minus hand formula", D[sk]["persist_minus_hand"]))
            a("- " + pair_line("Gradient boosting minus persistence (8 years)", D[sk]["hgb_minus_persist"], MEASURES3))
            a("")
    if not all(x == "clear" for x in v.values()):
        j = ("Judgement, not a measured result: a fitted model that only ties persistence adds machinery (a "
             "coefficient file, a history window to keep current, a feature builder) for no demonstrated gain, "
             "which is hard to justify")
        if PD["main"][P][ep]["ap"]["hi"] < 0:
            j += "; where it trails persistence, as in the park on average precision, it is not worth adopting on accuracy grounds"
        a(j + ". Boosting is a check only and is not exported; its section shows what it does and does not add.\n")
    a(f"Setup: fitted on 2012 to 2019 satellite detections, scored on 2020 to 2024 (2025 is a partial year and "
      f"was not scored). The unit is one 500 m cell; the label is at least one detection in the period. "
      f"Intervals are 95% bootstrap intervals over 20 x 20-cell blocks ({N_BOOT} resamples); differences are "
      f"computed on the same resamples.\n")
    a("## Methods compared\n")
    a("- Random: expected values, for reference.\n"
      "- Hand formula: `priority_surface._real_layers` (0.40 fire density, 0.25 fuel, 0.20 dryness, 0.15 terrain "
      "spread, lines 147 to 177), run on this grid. Its fire term uses 2012 to 2019 detections only. The production "
      "pipeline uses all years, which would include the test period, so this is the fair version, not the one shipped.\n"
      "- Persistence (8 years): Gaussian-smoothed (sigma 1 cell) log of 1 plus the 2012 to 2019 count.\n"
      "- Persistence, 2016 to 2019 only: the same smoothing on four years, the history length the primary learned "
      "model gets.\n"
      "- Logistic, static features: 17 terrain, land-cover, park and distance features, fitted on the 2012 to 2019 label.\n"
      "- Logistic plus fire history (primary): the same features plus smoothed log count of detections over a "
      "history window. Fitted with history 2012 to 2015 against label 2016 to 2019, scored with history 2016 to 2019. "
      "This is the primary specification; a variant scored with 2012 to 2019 rescaled to a four-year rate is reported "
      "beside it.\n"
      "- Gradient boosting: same inputs as the primary model, only to check whether non-linearity matters. Not exported.\n"
      "- Regularisation strength was chosen by 5-fold cross-validation on 20 x 20-cell blocks inside the training "
      "period only. No fire detections after 2019 enter any feature or score (see Limits for the land-cover and "
      "road data).\n")
    for per, title in [("main", "Test 2020 to 2024"), ("ex2020", "Test 2021 to 2024 (2020, the largest year, excluded)")]:
        for sk, nm in [(W, "whole grid"), (P, "park cells")]:
            a(f"## {title}: {nm}\n")
            n = doc["subset_cells"][sk]
            a(f"{n:,} cells. Cells show the value and the 95% interval. Capture is the share of test-period detections "
              f"(by count) falling in the top 10% or 20% of cells by score; random is 10% and 20% by construction.\n")
            a(headline_table(doc, per, sk))
    a("## Paired differences\n")
    a("First-named method minus the second; positive favours the first. Capture differences are in percentage points.\n")
    a("| Subset | Test | Difference | Avg precision | Capture 20% | Capture 10% | ROC AUC |\n|---|---|---|---|---|---|---|")
    for per, pn in [("main", "2020-2024"), ("ex2020", "2021-2024")]:
        for sk in (P, W):
            for pair, vn in [("lr_hist_minus_persist", "Learned minus persistence (8 years)"),
                             ("lr_hist_minus_persist4", "Learned minus persistence (2016-2019 only)"),
                             ("lr_hist_alt_minus_persist", "Learned 8-year variant minus persistence (8 years)"),
                             ("lr_hist_minus_hand", "Learned minus hand formula"),
                             ("persist_minus_hand", "Persistence (8 years) minus hand formula"),
                             ("lr_hist_minus_lr_static", "Learned minus logistic, static"),
                             ("lr_hist_minus_lr_hist_alt", "Learned minus its 8-year variant"),
                             ("hgb_minus_lr_hist", "Boosting minus learned"),
                             ("hgb_minus_persist", "Boosting minus persistence (8 years)")]:
                d = PD[per][sk][pair]
                a(f"| {sk} | {pn} | {vn} | {diff_text(d['ap'], 'ap')} | {diff_text(d['cap20'], 'cap20')} | "
                  f"{diff_text(d['cap10'], 'cap10')} | {diff_text(d['auc'], 'auc')} |")
    a("")
    a("## Other subsets (2020 to 2024, point estimates only)\n")
    a("Spatial blocks give no usable intervals on windows this small. The dashboard window (400 cells) holds "
      f"{int(doc['point']['main']['dashboard20']['random']['ap']*400 + 0.5)} cells with a test-period detection and is noisy; do not read rankings from it.\n")
    for met in ("ap", "cap20"):
        a(f"**{METRIC_LABELS[met]}**\n")
        ks = ["whole", "park", "band5km", "region40", "dashboard20"]
        a("| Method | " + " | ".join(SUBSET_LABELS[k] for k in ks) + " |\n|---|" + "---|" * len(ks))
        for m in METHODS:
            a(f"| {LABELS[m]} | " + " | ".join(fnum(doc["point"]["main"][k][m][met], met.startswith("cap")) for k in ks) + " |")
        a("")
    sv = doc["spatial_verdict"]
    a("## Spatial check\n")
    a("Models d and e refitted on one half of the grid only (west: columns 0 to 99; east: 100 to 199) and scored on "
      "the other half's 2020 to 2024 detections. The hand formula and persistence need no fitting.\n")
    sm = ("hand", "persist", "persist4", "lr_static", "lr_hist")
    a("| Fitted on | Scored on | Subset | Hand AP | Persist AP | Persist 2016-19 AP | Static AP | Hist AP | Hand cap20 | Persist cap20 | Persist 2016-19 cap20 | Static cap20 | Hist cap20 |\n|" + "---|" * 13)
    for k, o in doc["spatial_check"].items():
        tr, te = k.replace("train_", "").split("_test_")
        for sk in (W, P):
            t = o[sk]
            a(f"| {tr} | {te} | {sk} | " + " | ".join(fnum(t[m]["ap"]) for m in sm)
              + " | " + " | ".join(fnum(t[m]["cap20"], True) for m in sm) + " |")
    a("")
    holds_p = all(x["e_beats_persistence_ap"] and x["e_beats_persistence_cap20"] for x in sv.values())
    holds_h = all(x["e_beats_hand_ap"] and x["e_beats_hand_cap20"] for x in sv.values())
    orders = "; ".join(f"{k.replace('train_', 'fit ').replace('_test_', ', score ')} / {sk}: "
                       + " > ".join(m.replace("lr_hist", "learned+history").replace("lr_static", "static")
                                    for m in x["order_by_ap"])
                       for k in doc["spatial_check"] for sk in (W, P) for x in [sv[f"{k}/{sk}"]])
    a(f"Ranking by average precision in each case (hand, persistence, static and learned only): {orders}.\n")
    a(f"Learned model above persistence on both measures in every cell of this table: **{'yes' if holds_p else 'no'}**. "
      f"Above the hand formula on both measures in every cell: **{'yes' if holds_h else 'no'}**. "
      f"Per-row detail is in `hazard_validation.json` under `spatial_verdict`.\n")
    a("## Coefficients\n")
    a("Standardised logistic coefficients (change in log-odds per one standard deviation). Largest six by size. "
      "These are associations in this grid and period, not causes. Roads, cropland and settlements probably stand in "
      "for agricultural burning, access and where satellites can see fires, but that was not tested.\n")
    a("**Model d (static features only):**\n")
    a(coef_lines(doc["coefficients"]["static"]) + "\n")
    a("**Model e (static plus fire history):**\n")
    a(coef_lines(doc["coefficients"]["history"]) + "\n")
    a("Full lists are in `hazard_validation.json`.\n")
    a("## Gradient boosting check\n")
    a("Boosting uses the same inputs as the primary logistic model and is not exported. Differences are boosting "
      "minus the other method.\n")
    for per, pname in PERIODS:
        a(f"**Test {pname}**\n")
        for sk, nm in [(W, "Whole grid"), (P, "Park")]:
            a("- " + pair_line(f"{nm}, boosting minus logistic with history", PD[per][sk]["hgb_minus_lr_hist"], MEASURES3))
            a("- " + pair_line(f"{nm}, boosting minus persistence (8 years)", PD[per][sk]["hgb_minus_persist"], MEASURES3))
        a("")
    a("## Exported models\n")
    a("`validated` is model e exactly as tested. `deployment` has the same specification but is refitted with history "
      "2012 to 2018 against label 2019 to 2025, so it uses the later fires; it is given the whole 2012 to 2025 record "
      "rescaled to seven years when predicting. The split in the test cannot be repeated for 2019 to 2025 because "
      "there is nothing left to test on, so this model is not itself validated out of sample; it inherits the "
      "validation of the specification. `static` is model d.\n")
    hr = doc["hand_formula_region_normalisation"]
    a("## Limits\n")
    a("- Detections are not ignitions. VIIRS misses small and under-canopy fires and sees an overpass a few times a "
      "day. Outside the forest it probably sees mostly agricultural burning, so the models likely learn where "
      "satellites see fires, a mixture of farm burning and wildfire that was not separated here.\n"
      f"- Inside the park there are few events: {F['test_detections_park']:,} of {F['test_detections']:,} test detections "
      f"fall on park cells, and {fnum(F['test_prevalence_park'], True, 1)} of park cells have any (grid: "
      f"{fnum(F['test_prevalence'], True, 1)}). Park intervals are wide.\n"
      f"- One test period. 2020 alone holds {F['detections_2020']:,} of the {F['test_detections']:,} test detections; "
      f"results with it excluded are shown above. {F['test_detections_ex2020']:,} detections remain.\n"
      "- Possible leakage from non-fire data. Land cover is ESA WorldCover 2021, inside the test window, and burn scars "
      "can change a cell's class; roads and settlements are an OpenStreetMap snapshot from November 2026. This could "
      "leak a little test-period information into the static features. That can only help the hand formula and the "
      "learned models, never persistence, which uses fire counts alone, so the comparison is conservative against "
      "persistence.\n"
      "- The hand formula is normalised here over the 200 x 200 grid; production normalises over the 40 x 40 region. "
      f"On the 40 x 40 region its average precision is {fnum(hr['grid_normalised_ap'])} with grid normalisation and "
      f"{fnum(hr['region_normalised_ap'])} with region normalisation (capture at 20%: "
      f"{fnum(hr['grid_normalised_cap20'], True)} and {fnum(hr['region_normalised_cap20'], True)}), against "
      f"{fnum(hr['persistence_ap'])} ({fnum(hr['persistence_cap20'], True)}) for persistence, so the order is unchanged.\n"
      "- The bootstrap intervals resample test cells only. The models were fitted once, so the intervals do not "
      "include model-fitting uncertainty.\n"
      "- Cells are spatially correlated; the block bootstrap accounts for that at 10 km, not beyond. "
      "Results come from one area (Khao Yai and surroundings) and may not transfer.\n"
      f"- 2025 ({F['detections_2025']:,} detections in the grid) is a partial year: excluded from testing, included "
      "in the deployment model's label.\n"
      "- The output ranks cells by the chance of at least one detection over a few years. It is not a calibrated "
      "ignition probability, and it says nothing about fire spread or damage.\n"
      "- Coefficients are associations, not causes.\n"
      "- No quantum content or advantage is involved in this analysis.\n")
    with open(os.path.join(HERE, "HAZARD_VALIDATION.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
