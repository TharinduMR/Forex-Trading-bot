"""
Unit tests for train.py verifying Purged K-Fold overlap purging and training pipeline execution.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import train
import config


def test_purged_kfold_overlap_removal():
    """Test that PurgedKFold removes training samples whose holding horizon overlaps the test fold."""
    n = 100
    df = pd.DataFrame({
        'target': np.random.choice([-1.0, 0.0, 1.0], size=n),
        'holding_bars': np.full(n, 10)  # Each sample holds for 10 bars
    })
    
    pkf = train.PurgedKFold(n_splits=5, embargo_pct=0.0)
    
    for trn_idx, test_idx in pkf.split(df):
        test_start = min(test_idx)
        test_end = max(test_idx)
        
        # Check all training indices BEFORE test_start
        # None of them should have trn_idx + 10 >= test_start!
        pre_test_trn = trn_idx[trn_idx < test_start]
        for idx in pre_test_trn:
            horizon_end = idx + df.loc[idx, 'holding_bars']
            assert horizon_end < test_start, f"Data leakage! Train index {idx} with horizon ending at {horizon_end} overlaps test fold starting at {test_start}!"


def test_train_pipeline_smoke(tmp_path):
    """Smoke test for train_pipeline on synthetic data."""
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    df = pd.DataFrame({
        'close': 2000.0 + np.random.randn(n) * 5.0,
        'high': 2005.0 + np.random.randn(n) * 5.0,
        'low': 1995.0 + np.random.randn(n) * 5.0,
        'open': 2000.0 + np.random.randn(n) * 5.0,
        'volume': 1000.0,
        'fvg_active': np.random.choice([-1.0, 0.0, 1.0], size=n),
        'rsi': np.random.uniform(0.2, 0.8, size=n),
        'target': np.random.choice([-1.0, 0.0, 1.0], size=n),
        'holding_bars': np.random.randint(1, 15, size=n),
        'ret_horizon': np.random.randn(n) * 0.005
    }, index=dates)
    
    # Temporarily override model save paths to tmp_path
    config.XGB_MODEL_PATH = os.path.join(tmp_path, "xgb.joblib")
    config.LGB_MODEL_PATH = os.path.join(tmp_path, "lgb.joblib")
    config.FEATURE_NAMES_PATH = os.path.join(tmp_path, "features.joblib")
    
    feature_cols = ['fvg_active', 'rsi']
    xgb_model, lgb_model, best_params = train.train_pipeline(df, feature_cols, use_optuna=False)
    
    assert os.path.exists(config.XGB_MODEL_PATH)
    assert os.path.exists(config.LGB_MODEL_PATH)
    loaded_features = joblib.load(config.FEATURE_NAMES_PATH)
    assert loaded_features == feature_cols


def test_meta_labeling_training(tmp_path):
    """Test that train_meta_model executes without error and filters signals correctly."""
    from meta_labeling import train_meta_model
    n = 150
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        'fvg_active': np.random.choice([-1.0, 0.0, 1.0], size=n),
        'rsi': np.random.uniform(0.2, 0.8, size=n),
        'target': np.random.choice([-1.0, 0.0, 1.0], size=n),
    }, index=dates)
    
    oos_df = pd.DataFrame({
        'pred': np.random.choice([-1.0, 0.0, 1.0], size=n),
        'prob': np.random.uniform(0.5, 0.9, size=n),
    }, index=dates)
    
    config.META_MODEL_PATH = os.path.join(tmp_path, "meta.joblib")
    config.OOS_PREDS_PATH = os.path.join(tmp_path, "oos.joblib")
    
    feature_cols = ['fvg_active', 'rsi']
    meta_m = train_meta_model(df, feature_cols, oos_preds_df=oos_df, target_col='target')
    assert meta_m is not None
    assert os.path.exists(config.META_MODEL_PATH)
    assert 'meta_prob' in oos_df.columns


def test_combine_probabilities_uses_weighted_blend():
    """Test that ensemble blending uses the configured XGBoost weighting."""
    prob_xgb = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]])
    prob_lgb = np.array([[0.3, 0.2, 0.5], [0.1, 0.1, 0.8]])

    blended = train.combine_probabilities(prob_xgb, prob_lgb, xgb_weight=0.7)

    assert np.allclose(blended[0], np.array([0.58, 0.2, 0.22]))
    assert np.allclose(blended[1], np.array([0.17, 0.45, 0.38]))


def test_meta_labeling_handles_no_trade_signals(tmp_path):
    """Test that meta-labeling falls back gracefully when the primary model emits no trades."""
    from meta_labeling import train_meta_model

    df = pd.DataFrame({
        'fvg_active': np.zeros(40),
        'rsi': np.linspace(0.2, 0.8, 40),
        'target': np.zeros(40),
    })

    config.META_MODEL_PATH = os.path.join(tmp_path, "meta_fallback.joblib")
    meta_model = train_meta_model(df, ['fvg_active', 'rsi'], oos_preds_df=None, target_col='target')

    assert meta_model is not None
    probs = meta_model.predict_proba(np.zeros((3, 2)))
    assert probs.shape[1] == 2


def test_holdout_split_and_nested_cv(tmp_path):
    """Test train_pipeline with 20% holdout split and nested CV."""
    n = 150
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    df = pd.DataFrame({
        'close': 2000.0 + np.random.randn(n) * 5.0,
        'high': 2005.0 + np.random.randn(n) * 5.0,
        'low': 1995.0 + np.random.randn(n) * 5.0,
        'open': 2000.0 + np.random.randn(n) * 5.0,
        'volume': 1000.0,
        'fvg_active': np.random.choice([-1.0, 0.0, 1.0], size=n),
        'rsi': np.random.uniform(0.2, 0.8, size=n),
        'target': np.random.choice([-1.0, 0.0, 1.0], size=n),
        'holding_bars': np.random.randint(1, 15, size=n),
        'ret_horizon': np.random.randn(n) * 0.005
    }, index=dates)
    
    config.XGB_MODEL_PATH = os.path.join(tmp_path, "xgb_h.joblib")
    config.LGB_MODEL_PATH = os.path.join(tmp_path, "lgb_h.joblib")
    config.FEATURE_NAMES_PATH = os.path.join(tmp_path, "features_h.joblib")
    config.OOS_PREDS_PATH = os.path.join(tmp_path, "oos_h.joblib")
    config.META_MODEL_PATH = os.path.join(tmp_path, "meta_h.joblib")
    
    feature_cols = ['fvg_active', 'rsi']
    xgb_m, lgb_m, best_p = train.train_pipeline(df, feature_cols, use_optuna=False, holdout_pct=0.20)
    
    holdout_path = config.OOS_PREDS_PATH.replace("oos_predictions", "holdout_predictions").replace("oos_h.joblib", "holdout_h.joblib")
    assert os.path.exists(config.XGB_MODEL_PATH)
    assert os.path.exists(config.LGB_MODEL_PATH)

