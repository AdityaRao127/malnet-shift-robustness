# import smoke tests, run with pytest tests/ or python tests/test_imports.py
# checks core deps and our own src modules

import os
import sys

# add src to path so the src.* imports work from pytest cwd
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)


def test_core_deps():
    # third party deps that the notebook needs
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401
    import sklearn  # noqa: F401
    import pandas  # noqa: F401
    import matplotlib  # noqa: F401
    import tqdm  # noqa: F401
    import numpy  # noqa: F401
    assert True


def test_data_loader():
    from data_loader import load_malnet_splits, make_loaders, make_pre_transform
    assert callable(load_malnet_splits)
    assert callable(make_loaders)
    assert callable(make_pre_transform)


def test_features():
    from features import compute_degree_features, compute_ldp_features, make_pre_transform
    assert callable(compute_degree_features)
    assert callable(compute_ldp_features)
    assert callable(make_pre_transform)


def test_model():
    from model import BaselineGCN
    m = BaselineGCN(in_channels=5, hidden_channels=8, num_classes=3)
    total = sum(p.numel() for p in m.parameters())
    assert total > 0


def test_train_helpers():
    from train import train_one_epoch, run_eval, train_model, build_metrics, save_metrics
    assert callable(train_one_epoch)
    assert callable(run_eval)
    assert callable(train_model)
    assert callable(build_metrics)
    assert callable(save_metrics)


if __name__ == "__main__":
    # cli mode for quick local check
    tests = [test_core_deps, test_data_loader, test_features, test_model, test_train_helpers]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  [OK] {t.__name__}")
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} test(s) failed")
        sys.exit(1)
    print("\nall tests passed")
