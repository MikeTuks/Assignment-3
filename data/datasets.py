"""Loads and pre-processes the eight problems.

Each loader returns (X, T, kind): float64 2-D arrays, and kind is
"classification" (one-hot T) or "regression" (one standardised column).

Measured sets are downloaded from UCI or copied from sklearn on first use;
synthetic sets are written from SEED. All are stored in this folder.
"""

import argparse
import io
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0  # all generated sets; also in the CSV filenames

CACHE = GENERATED = Path(__file__).parent  # all data files live beside this script

UCI = {
    "ionosphere.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
    "sonar.all-data": "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
    "Concrete_Data.xls": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
    "communities.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
}


def _fetch(name):
    """Raw bytes of a UCI file, downloaded into cache/ on first use."""
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(UCI[name], timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def _sklearn_file(name):
    """Path to a copy in cache/ of a data file bundled with sklearn."""
    path = CACHE / name
    if not path.exists():
        import sklearn.datasets

        CACHE.mkdir(parents=True, exist_ok=True)
        src = Path(sklearn.datasets.__file__).parent / "data" / name
        shutil.copy(src, path)
    return path


def _standardise(X):
    """Zero mean, unit variance per column (constant columns left unscaled)."""
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def _drop_constant(X):
    return X[:, X.std(axis=0) > 0]


def _one_hot(y):
    classes = np.unique(y)
    return (y[:, None] == classes[None, :]).astype(np.float64)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def iris():
    """150x4, 3 classes."""
    # Skip sklearn's header row.
    a = np.loadtxt(_sklearn_file("iris.csv"), delimiter=",", skiprows=1)

    # Standardised so raw centimetre values don't saturate tanh.
    return _standardise(a[:, :4]), _one_hot(a[:, 4].astype(int)), "classification"


def ionosphere():
    """351x33 after dropping a constant column, 2 classes."""
    df = pd.read_csv(io.BytesIO(_fetch("ionosphere.data")), header=None)
    X = df.iloc[:, :34].to_numpy(np.float64)
    y = (df.iloc[:, 34].to_numpy() == "g").astype(int)

    # Column 1 is all zeros; its weights would never affect the output.
    X = _drop_constant(X)

    return _standardise(X), _one_hot(y), "classification"


def sonar():
    """208x60, rock vs metal."""
    df = pd.read_csv(io.BytesIO(_fetch("sonar.all-data")), header=None)
    X = df.iloc[:, :60].to_numpy(np.float64)
    y = (df.iloc[:, 60].to_numpy() == "M").astype(int)

    # Already in [0, 1], so not rescaled.
    return X, _one_hot(y), "classification"


def digits():
    """1797x61 after dropping constant pixels, 10 classes."""
    a = np.loadtxt(_sklearn_file("digits.csv.gz"), delimiter=",")
    X, y = a[:, :64], a[:, 64].astype(int)

    # Pixels 0, 32 and 39 are always 0.
    X = _drop_constant(X)

    # Map 0..16 to [-1, 1] rather than standardising: near-constant border
    # pixels have tiny standard deviations that would blow up.
    X = X / 8.0 - 1.0

    return X, _one_hot(y), "classification"


# ---------------------------------------------------------------------------
# Function approximation
# ---------------------------------------------------------------------------


def _generated_path(name):
    """generated/<name>_seed<SEED>.csv"""
    return GENERATED / f"{name}_seed{SEED}.csv"


def _materialise(name):
    """Load a generated set from its CSV, writing the CSV on first use."""
    path = _generated_path(name)
    if path.exists():
        a = np.loadtxt(path, delimiter=",", skiprows=1)
        return a[:, :-1], a[:, -1:]

    X, T = GENERATORS[name]()
    _write(path, X, T)
    return X, T


def _write(path, X, T):
    GENERATED.mkdir(parents=True, exist_ok=True)
    header = ",".join([f"x{i + 1}" for i in range(X.shape[1])] + ["t"])
    np.savetxt(path, np.hstack([X, T]), delimiter=",", header=header, comments="")


def _sine_from_seed(n, noise, seed):
    """sin(x) + noise, x uniform on [-pi, pi]."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, size=(n, 1))
    T = np.sin(X) + rng.normal(0.0, noise, size=(n, 1))
    return X, _standardise(T)


def sine():
    """300x1 noisy sine."""
    return (*_materialise("sine"), "regression")


def concrete():
    """1030x8, compressive strength (MPa)."""
    df = pd.read_excel(io.BytesIO(_fetch("Concrete_Data.xls")))
    X = df.iloc[:, :8].to_numpy(np.float64)
    T = df.iloc[:, 8].to_numpy(np.float64)[:, None]

    # Inputs mix kg/m^3 and days. The target is standardised too, so lambda
    # is on the same scale as the error.
    return _standardise(X), _standardise(T), "regression"


def communities():
    """1994x99, violent crimes per capita."""
    df = pd.read_csv(io.BytesIO(_fetch("communities.data")), header=None,
                     na_values=["?"])

    # Columns 0-4 are identifiers; the last column is the target.
    X = df.iloc[:, 5:-1]
    T = df.iloc[:, -1].to_numpy(np.float64)[:, None]

    # Drop the 23 columns with missing values (22 are over half missing).
    # The target is complete, so no rows are lost.
    X = X.loc[:, X.isna().sum() == 0].to_numpy(np.float64)

    # Inputs already in [0, 1]; target standardised as for concrete.
    return X, _standardise(T), "regression"


def _friedman(n, n_features, noise, seed):
    """Friedman #1: only the first five inputs matter."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, n_features))
    T = (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    )[:, None]
    T = T + rng.normal(0.0, noise, size=T.shape)
    return X, _standardise(T)


def friedman():
    """500x10; inputs 6-10 are noise."""
    return (*_materialise("friedman"), "regression")


# The synthetic sets, defined once. The CSVs in generated/ are saved copies.
GENERATORS = {
    "sine": lambda: _sine_from_seed(300, 0.1, SEED),
    "friedman": lambda: _friedman(500, 10, 1.0, SEED),
}


LOADERS = {
    "iris": iris,
    "ionosphere": ionosphere,
    "sonar": sonar,
    "digits": digits,
    "sine": sine,
    "concrete": concrete,
    "friedman": friedman,
    "communities": communities,
}


def load(name):
    return LOADERS[name]()


def n_weights(n_in, n_hid, n_out):
    return n_hid * (n_in + 1) + n_out * (n_hid + 1)


def regenerate():
    """Rewrite every generated CSV from its seed."""
    for name, make in GENERATORS.items():
        path = _generated_path(name)
        X, T = make()
        _write(path, X, T)
        print(f"wrote {path.name}  {X.shape[0]}x{X.shape[1]}")


def _check_generated_match_seed():
    """Each generated CSV must match a fresh run of its generator."""
    for name, make in GENERATORS.items():
        saved = np.loadtxt(_generated_path(name), delimiter=",", skiprows=1)
        X, T = make()
        fresh = np.hstack([X, T])
        assert saved.shape == fresh.shape, f"{name}.csv has the wrong shape"
        assert np.allclose(saved, fresh), (
            f"{name}.csv no longer matches its seed, rerun with --regenerate"
        )


def _selfcheck():
    expected = {
        "iris": (150, 4, 3),
        "ionosphere": (351, 33, 2),
        "sonar": (208, 60, 2),
        "digits": (1797, 61, 10),
        "sine": (300, 1, 1),
        "concrete": (1030, 8, 1),
        "friedman": (500, 10, 1),
        "communities": (1994, 99, 1),
    }

    print(f"{'problem':<14}{'samples':>8}{'in':>5}{'out':>5}   weights nh=4/10/20/32")
    for name, (n, n_in, n_out) in expected.items():
        X, T, kind = load(name)
        assert X.shape == (n, n_in), f"{name}: X is {X.shape}, expected {(n, n_in)}"
        assert T.shape == (n, n_out), f"{name}: T is {T.shape}, expected {(n, n_out)}"
        assert np.isfinite(X).all() and np.isfinite(T).all(), f"{name}: non-finite"
        assert X.std(axis=0).min() > 0, f"{name}: a constant input column survived"

        if kind == "classification":
            assert set(np.unique(T)) == {0.0, 1.0}, f"{name}: targets not one-hot"
            assert (T.sum(axis=1) == 1).all(), f"{name}: not exactly one class per row"
        else:
            assert abs(T.mean()) < 1e-9, f"{name}: target not centred"

        w = [n_weights(n_in, h, n_out) for h in (4, 10, 20, 32)]
        print(f"{name:<14}{n:>8}{n_in:>5}{n_out:>5}   {w}")

    # Sonar and digits must cross the 630-1260 weight band; iris stays below.
    assert n_weights(60, 10, 2) < 1000 < n_weights(60, 20, 2), "sonar lost its crossing"
    assert n_weights(61, 10, 10) < 1000 < n_weights(61, 20, 10), "digits lost its crossing"
    assert n_weights(4, 32, 3) < 630, "iris is no longer the easy end"

    _check_generated_match_seed()
    print("\nall checks passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--regenerate",
        action="store_true",
        help="rewrite the generated CSVs from their seeds, then self-check",
    )
    if p.parse_args().regenerate:
        regenerate()
    _selfcheck()
