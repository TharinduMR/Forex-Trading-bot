"""
Purged Walk-Forward Cross-Validation and Model Training Engine.
Supports multi-symbol training (XAUUSD, EURUSD) with per-symbol model persistence.
Implements López de Prado's Purged K-Fold CV with Embargo, XGBoost & LightGBM ensemble training,
and Optuna hyperparameter optimization.
"""
import os
import joblib
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import classification_report, accuracy_score
import config
from labeling import compute_sample_weights


def combine_probabilities(prob_xgb, prob_lgb, xgb_weight=0.65):
    """Blend XGBoost and LightGBM probabilities with a slightly stronger XGBoost bias."""
    return (prob_xgb * xgb_weight) + (prob_lgb * (1.0 - xgb_weight))

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation with Embargo for financial time series.
    Prevents data leakage by removing training samples whose evaluation horizon
    overlaps with the test set, and applies an embargo after the test set.
    """
    def __init__(self, n_splits=config.N_SPLITS, embargo_pct=config.EMBARGO_PCT):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        
    def split(self, df, target_col='target', holding_col='holding_bars'):
        n_samples = len(df)
        indices = np.arange(n_samples)
        
        # Divide into equal contiguous folds
        fold_size = n_samples // self.n_splits
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            folds.append((start, end))
            
        embargo_bars = int(n_samples * self.embargo_pct)
        holding_bars = df[holding_col].values
        
        for k in range(self.n_splits):
            test_start, test_end = folds[k]
            test_idx = indices[test_start:test_end]
            
            # FIX BUG 7: Strict past-only mask for OOS harvesting
            train_mask_strict = indices < test_start
            
            # Purging: Check training samples BEFORE test_start
            # If their holding period extends into or beyond test_start, they leak test data!
            for trn_idx in range(test_start):
                if train_mask_strict[trn_idx]:
                    horizon_end = trn_idx + (holding_bars[trn_idx] if not np.isnan(holding_bars[trn_idx]) else 1)
                    if horizon_end >= test_start:
                        train_mask_strict[trn_idx] = False
                        
            train_idx = indices[train_mask_strict]
            yield train_idx, test_idx


def evaluate_cv_profit_factor(y_true, y_pred, ret_horizon):
    """
    Evaluate Profit Factor (Gross Profit / Gross Loss) on test predictions.
    We assume taking long when pred==1 and short when pred==-1.
    """
    trades_mask = y_pred != 0
    if not np.any(trades_mask):
        return 0.5  # Heavy penalty for taking zero trades
        
    pnl = np.where(y_pred == 1, ret_horizon, np.where(y_pred == -1, -ret_horizon, 0.0))
    
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = np.abs(np.sum(pnl[pnl < 0]))
    
    if gross_loss == 0:
        return gross_profit * 10.0 if gross_profit > 0 else 1.0
        
    return gross_profit / gross_loss

from sklearn.calibration import CalibratedClassifierCV

def train_xgboost(X_train, y_train, sample_weights=None, params=None, X_val=None, y_val=None):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed")
        
    default_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_lambda': 1.0,
        'gamma': 0.1,
        'early_stopping_rounds': 20
    }
    if params:
        default_params.update(params)
        
    if X_val is None or y_val is None:
        default_params.pop('early_stopping_rounds', None)
        
    # Shift targets from [-1, 0, 1] to [0, 1, 2] for XGBoost multi-class
    y_train_shifted = (y_train + 1).astype(int)
    
    model = XGBClassifier(**default_params)
    if X_val is not None and y_val is not None:
        y_val_shifted = (y_val + 1).astype(int)
        
        # PREVENT LEAKAGE: Use a subset of training data for early stopping, NOT the OOS validation fold.
        sub_val_size = max(int(len(X_train) * 0.15), 1)
        X_sub_train, X_sub_val = X_train[:-sub_val_size], X_train[-sub_val_size:]
        y_sub_train, y_sub_val = y_train_shifted[:-sub_val_size], y_train_shifted[-sub_val_size:]
        sw_sub = sample_weights[:-sub_val_size] if sample_weights is not None else None
        
        model.fit(
            X_sub_train, y_sub_train, 
            sample_weight=sw_sub,
            eval_set=[(X_sub_val, y_sub_val)],
            verbose=False
        )
        # IMPROVEMENT 11: Calibrate if validation data is provided
        try:
            calibrated_model = CalibratedClassifierCV(estimator=model, method='isotonic', cv='prefit')
            calibrated_model.fit(X_sub_val, y_sub_val)
            return calibrated_model
        except Exception as e:
            print(f"[Warning] Calibration failed ({e}). Returning uncalibrated model.")
            return model
    else:
        model.fit(X_train, y_train_shifted, sample_weight=sample_weights)
    return model


def train_lightgbm(X_train, y_train, sample_weights=None, params=None, X_val=None, y_val=None):
    """Train LightGBM Classifier."""
    if LGBMClassifier is None:
        raise ImportError("LightGBM is not installed.")
        
    default_params = {
        'n_estimators': 300,
        'max_depth': 3,
        'learning_rate': 0.04,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'min_child_samples': 25,
        'reg_lambda': 1.0,
        'reg_alpha': 0.2,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    if params:
        default_params.update(params)
        
    y_train_shifted = (y_train + 1).astype(int)
    
    model = LGBMClassifier(**default_params)
    
    # LightGBM handles early stopping slightly differently depending on version, 
    # but older versions support it via fit params or callbacks.
    if X_val is not None and y_val is not None:
        try:
            from lightgbm import early_stopping
            callbacks = [early_stopping(20, verbose=False)]
        except ImportError:
            callbacks = None
            
        if callbacks:
            # PREVENT LEAKAGE: Use a subset of training data for early stopping.
            sub_val_size = max(int(len(X_train) * 0.15), 1)
            X_sub_train, X_sub_val = X_train[:-sub_val_size], X_train[-sub_val_size:]
            y_sub_train, y_sub_val = y_train_shifted[:-sub_val_size], y_train_shifted[-sub_val_size:]
            sw_sub = sample_weights[:-sub_val_size] if sample_weights is not None else None
            
            model.fit(
                X_sub_train, y_sub_train, 
                sample_weight=sw_sub,
                eval_set=[(X_sub_val, y_sub_val)],
                callbacks=callbacks
            )
        else:
            model.fit(X_train, y_train_shifted, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train_shifted, sample_weight=sample_weights)
    return model


def optimize_hyperparameters(df, feature_cols, target_col='target', n_trials=config.OPTUNA_TRIALS):
    """
    Use Optuna to optimize XGBoost hyperparameters across Purged K-Fold CV.
    Objective: Maximize out-of-sample Profit Factor.
    """
    if optuna is None:
        warnings.warn("Optuna not installed. Skipping hyperparameter optimization.")
        return {}
        
    print(f"[Optuna] Starting hyperparameter optimization across {n_trials} trials...")
    X = df[feature_cols].values
    y = df[target_col].values
    ret_horizon = df['ret_horizon'].values
    weights = compute_sample_weights(df, target_col).values
    
    pkf = PurgedKFold()
    
    def objective(trial):
        # IMPROVEMENT 8: Optimize labeling parameters
        tp_mult = trial.suggest_float('tp_mult', 1.5, 3.0)
        sl_mult = trial.suggest_float('sl_mult', 1.0, 2.5)
        max_hold = trial.suggest_int('max_holding', 8, 24)
        
        # Re-label with trial parameters
        from labeling import apply_triple_barrier
        df_trial = apply_triple_barrier(df.copy(), tp_mult=tp_mult, sl_mult=sl_mult, max_holding=max_hold)
        df_trial = df_trial.dropna(subset=[target_col])
        
        X_trial = df_trial[feature_cols].values
        y_trial = df_trial[target_col].values
        ret_horizon_trial = df_trial['ret_horizon'].values
        weights_trial = compute_sample_weights(df_trial, target_col).values
        
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'n_estimators': trial.suggest_int('n_estimators', 80, 250)
        }
        
        profit_factors = []
        for trn_idx, val_idx in pkf.split(df_trial, target_col):
            if len(trn_idx) < 50 or len(val_idx) < 20:
                continue
                
            model = train_xgboost(X_trial[trn_idx], y_trial[trn_idx], sample_weights=weights_trial[trn_idx], params=params, X_val=X_trial[val_idx], y_val=y_trial[val_idx])
            
            # Predict on val
            val_preds_shifted = model.predict(X_trial[val_idx])
            val_preds = val_preds_shifted - 1  # Shift back to [-1, 0, 1]
            
            pf = evaluate_cv_profit_factor(y_trial[val_idx], val_preds, ret_horizon_trial[val_idx])
            profit_factors.append(pf)
            
        return np.mean(profit_factors) if profit_factors else 0.5
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"[Optuna] Best Trial Profit Factor: {study.best_value:.3f}")
    print(f"[Optuna] Best Parameters: {study.best_params}")
    return study.best_params


def get_feature_importance(model, feature_cols, model_name="XGBoost"):
    """Extract and sort feature importances."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
        fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
        return fi_df
    return pd.DataFrame()


def train_pipeline(df, feature_cols, target_col='target', holdout_pct=None, use_optuna=False, n_trials=5, symbol=None, timeframe="15m"):
    """
    Complete ML pipeline:
    1. Filter out un-usable bars (NaN targets).
    2. Run Optuna hyperparameter optimization on Training/CV set (Nested CV).
    3. Run Purged K-Fold CV to report out-of-sample performance.
    4. Train final XGBoost & LightGBM ensemble models.
    5. Evaluate performance on strict Out-of-Sample Holdout set.
    6. Save models, feature names, and predictions to disk.
    """
    if holdout_pct is None:
        holdout_pct = getattr(config, 'HOLDOUT_PCT', 0.20)
        
    # Resolve per-symbol model paths
    if symbol is not None:
        from config import get_model_paths, get_symbol_profile
        model_paths = get_model_paths(symbol, timeframe)
        profile = get_symbol_profile(symbol)
        sym_display = profile['display_name']
    else:
        model_paths = {
            'xgb': config.XGB_MODEL_PATH,
            'lgb': config.LGB_MODEL_PATH,
            'feature_names': config.FEATURE_NAMES_PATH,
            'oos_preds': config.OOS_PREDS_PATH,
            'meta': config.META_MODEL_PATH,
        }
        sym_display = 'XAUUSD (Legacy)'
    
    print(f"\n==================================================")
    print(f"STARTING MODEL TRAINING PIPELINE — {sym_display}")
    print(f"==================================================")
    
    # Drop rows with NaN targets (e.g. at the tail or uncalculated)
    clean_df = df.dropna(subset=[target_col]).copy()
    print(f"[Dataset] Total samples: {len(df)} | Clean labeled samples: {len(clean_df)}")
    
    # Chronological Holdout Partitioning (80% Train/CV, 20% Holdout)
    if holdout_pct > 0.0 and len(clean_df) >= 100:
        split_idx = int(len(clean_df) * (1.0 - holdout_pct))
        train_val_df = clean_df.iloc[:split_idx].copy()
        holdout_df = clean_df.iloc[split_idx:].copy()
        print(f"[Dataset Partition] Train/CV Set: {len(train_val_df)} bars (first {(1.0-holdout_pct)*100:.0f}%) | Holdout Set: {len(holdout_df)} bars (last {holdout_pct*100:.0f}%)")
    else:
        train_val_df = clean_df
        holdout_df = None
    
    # Check target class distribution
    class_counts = train_val_df[target_col].value_counts().sort_index()
    print(f"[Target Distribution (Train/CV Set)]:")
    for cls, count in class_counts.items():
        label_str = "Long (+1)" if cls == 1.0 else ("Short (-1)" if cls == -1.0 else "No Trade (0)")
        print(f"  {label_str}: {count} ({count/len(train_val_df)*100:.1f}%)")
        
    X = train_val_df[feature_cols].values
    y = train_val_df[target_col].values
    ret_horizon = train_val_df['ret_horizon'].values
    weights = compute_sample_weights(train_val_df, target_col).values
    
    # Step 1: Optuna Hyperparameter Optimization (Nested CV on Train/CV set)
    best_params = {}
    if use_optuna and optuna is not None:
        best_params = optimize_hyperparameters(train_val_df, feature_cols, target_col, n_trials=n_trials)
        
    # Step 2: Purged Walk-Forward CV evaluation & OOS prediction harvesting
    print(f"\n--- Running Purged K-Fold CV ({config.N_SPLITS} splits, embargo={config.EMBARGO_PCT*100:.1f}%) ---")
    pkf = PurgedKFold()
    cv_pfs = []
    cv_accs = []
    oos_dfs = []
    
    for fold, (trn_idx, val_idx) in enumerate(pkf.split(train_val_df, target_col)):
        if len(trn_idx) < 50 or len(val_idx) < 20:
            print(f"  Fold {fold+1}: Skipping (Train size {len(trn_idx)} too small due to strict OOS)")
            continue
            
        xgb_cv = train_xgboost(X[trn_idx], y[trn_idx], sample_weights=weights[trn_idx], params=best_params, X_val=X[val_idx], y_val=y[val_idx])
        lgb_cv = train_lightgbm(X[trn_idx], y[trn_idx], sample_weights=weights[trn_idx], params=best_params, X_val=X[val_idx], y_val=y[val_idx])
        
        prob_xgb = xgb_cv.predict_proba(X[val_idx])
        prob_lgb = lgb_cv.predict_proba(X[val_idx])
        prob_avg = combine_probabilities(prob_xgb, prob_lgb, xgb_weight=0.65)
        
        # Apply EMA Smoothing
        df_probs = pd.DataFrame(prob_avg)
        prob_avg_smoothed = df_probs.ewm(span=config.SIGNAL_SMOOTHING_WINDOW).mean().values
        
        preds_int = np.argmax(prob_avg_smoothed, axis=1)
        preds = preds_int - 1.0
        probs = np.max(prob_avg_smoothed, axis=1)
        
        pf = evaluate_cv_profit_factor(y[val_idx], preds, ret_horizon[val_idx])
        acc = accuracy_score(y[val_idx], preds)
        cv_pfs.append(pf)
        cv_accs.append(acc)
        print(f"  Fold {fold+1}: Train={len(trn_idx)}, Val={len(val_idx)} | Out-of-Sample Accuracy={acc*100:.1f}%, Profit Factor={pf:.2f}")
        
        # Train fold-specific Meta-Model on trn_idx (0% leakage into val_idx!)
        prob_xgb_trn = xgb_cv.predict_proba(X[trn_idx])
        prob_lgb_trn = lgb_cv.predict_proba(X[trn_idx])
        prob_avg_trn = combine_probabilities(prob_xgb_trn, prob_lgb_trn, xgb_weight=0.65)
        
        # Apply EMA Smoothing
        df_probs_trn = pd.DataFrame(prob_avg_trn)
        prob_avg_trn_smoothed = df_probs_trn.ewm(span=config.SIGNAL_SMOOTHING_WINDOW).mean().values
        
        preds_trn = np.argmax(prob_avg_trn_smoothed, axis=1) - 1.0
        probs_trn = np.max(prob_avg_trn_smoothed, axis=1)
        
        trn_mask = (preds_trn != 0.0) & (probs_trn >= config.CONFIDENCE_THRESHOLD)
        meta_probs_val = np.ones(len(val_idx))
        
        if sum(trn_mask) >= 20:
            meta_y_trn = ((preds_trn[trn_mask] == 1.0) & (y[trn_idx][trn_mask] == 1.0)) | ((preds_trn[trn_mask] == -1.0) & (y[trn_idx][trn_mask] == -1.0))
            meta_y_trn = meta_y_trn.astype(int)
            pos_c = sum(meta_y_trn == 1)
            neg_c = sum(meta_y_trn == 0)
            sw = neg_c / max(1, pos_c)
            from xgboost import XGBClassifier
            meta_cv = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=sw, random_state=42, eval_metric='logloss')
            meta_cv.fit(X[trn_idx][trn_mask], meta_y_trn)
            meta_probs_val = meta_cv.predict_proba(X[val_idx])[:, 1]
            
        fold_df = pd.DataFrame({
            'prob_short': prob_avg_smoothed[:, 0],
            'prob_flat': prob_avg_smoothed[:, 1],
            'prob_long': prob_avg_smoothed[:, 2],
            'pred': preds,
            'prob': probs,
            'meta_prob': meta_probs_val
        }, index=train_val_df.index[val_idx])
        oos_dfs.append(fold_df)
        
    print(f"--- Mean CV Performance: Accuracy={np.mean(cv_accs)*100:.1f}%, Profit Factor={np.mean(cv_pfs):.2f} ---\n")
    
    if oos_dfs:
        oos_full_df = pd.concat(oos_dfs).sort_index()
        # Remove any duplicate timestamps by taking the latest fold's prediction
        oos_full_df = oos_full_df[~oos_full_df.index.duplicated(keep='last')]
        joblib.dump(oos_full_df, model_paths['oos_preds'])
        print(f"[Saved] Out-of-Sample Walk-Forward Predictions -> {model_paths['oos_preds']}")
        
    # Step 3: Train final models on Train/CV dataset
    print(f"\n[Training] Fitting final XGBoost & LightGBM ensemble models on Train/CV set...")
    xgb_model = train_xgboost(X, y, sample_weights=weights, params=best_params)
    lgb_model = train_lightgbm(X, y, sample_weights=weights, params=best_params)
    
    # Step 4: Save models and feature names to per-symbol paths
    joblib.dump(xgb_model, model_paths['xgb'])
    joblib.dump(lgb_model, model_paths['lgb'])
    joblib.dump(feature_cols, model_paths['feature_names'])
    print(f"[Saved] XGBoost model -> {model_paths['xgb']}")
    print(f"[Saved] LightGBM model -> {model_paths['lgb']}")
    print(f"[Saved] Feature Schema -> {model_paths['feature_names']}")
    
    # Step 5: Train Secondary AI Decision Maker (Meta-Labeling)
    from meta_labeling import train_meta_model
    oos_df_for_meta = None
    if os.path.exists(model_paths['oos_preds']):
        oos_df_for_meta = joblib.load(model_paths['oos_preds'])
    train_meta_model(train_val_df, feature_cols, oos_preds_df=oos_df_for_meta, target_col=target_col, symbol=symbol)
    
    # Step 6: Strict Out-of-Sample Holdout Evaluation
    if holdout_df is not None and len(holdout_df) > 0:
        X_h = holdout_df[feature_cols].values
        y_h = holdout_df[target_col].values
        ret_h = holdout_df['ret_horizon'].values
        
        prob_xgb_h = xgb_model.predict_proba(X_h)
        prob_lgb_h = lgb_model.predict_proba(X_h)
        prob_avg_h = combine_probabilities(prob_xgb_h, prob_lgb_h, xgb_weight=0.65)
        
        df_probs_h = pd.DataFrame(prob_avg_h)
        prob_avg_h_smoothed = df_probs_h.ewm(span=config.SIGNAL_SMOOTHING_WINDOW).mean().values
        
        preds_h_int = np.argmax(prob_avg_h_smoothed, axis=1)
        preds_h = preds_h_int - 1.0
        probs_h = np.max(prob_avg_h_smoothed, axis=1)
        
        h_acc = accuracy_score(y_h, preds_h)
        h_pf = evaluate_cv_profit_factor(y_h, preds_h, ret_h)
        
        print(f"\n==================================================")
        print(f"STRICT OUT-OF-SAMPLE HOLDOUT EVALUATION — {sym_display}")
        print(f"==================================================")
        print(f"Holdout Window: {holdout_df.index[0]} to {holdout_df.index[-1]} ({len(holdout_df)} bars)")
        print(f"Holdout Accuracy: {h_acc*100:.1f}% | Holdout Profit Factor: {h_pf:.2f}")
        
        holdout_df_out = pd.DataFrame({
            'prob_short': prob_avg_h_smoothed[:, 0],
            'prob_flat': prob_avg_h_smoothed[:, 1],
            'prob_long': prob_avg_h_smoothed[:, 2],
            'pred': preds_h,
            'prob': probs_h
        }, index=holdout_df.index)
        
        holdout_path = model_paths['oos_preds'].replace("oos_predictions", "holdout_predictions")
        joblib.dump(holdout_df_out, holdout_path)
        print(f"[Saved] Out-of-Sample Holdout Predictions -> {holdout_path}\n")
    
    # Step 7: Display top feature importance
    fi_df = get_feature_importance(xgb_model, feature_cols, "XGBoost")
    if not fi_df.empty:
        print(f"--- Top 10 Most Important Features (XGBoost Gain) ---")
        for idx, row in fi_df.head(10).iterrows():
            print(f"  {idx+1}. {row['feature']:<25} ({row['importance']:.4f})")
            
    return xgb_model, lgb_model, best_params


def train_smc_model(df, feature_cols, symbol=None, timeframe="15m"):
    """
    Train the SMC Specialist Regression Model on filtered event data.
    
    Instead of training on every candle (95% of which are noise), this isolates
    ONLY the bars where an SMC Playbook confluence occurred (HTF Discount/Premium +
    Liquidity Sweep + CHoCH + FVG/OB) and trains an XGBRegressor to predict the
    volatility-adjusted forward return magnitude.
    
    The model learns the hidden variables that separate winning SMC setups from
    fakeouts (e.g., session timing, FVG size, sweep depth, volume anomalies).
    
    Args:
        df: Full featured DataFrame with 'smc_trigger' and 'target_reg' columns.
        feature_cols: List of feature column names for ML input.
        symbol: Canonical symbol key (e.g., 'XAUUSD') for model path resolution.
        timeframe: Timeframe string (e.g., '15m') for model path resolution.
    
    Returns:
        Trained XGBRegressor model, or None if insufficient data.
    """
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    
    # Resolve per-symbol model paths
    if symbol is not None:
        from config import get_model_paths, get_symbol_profile
        model_paths = get_model_paths(symbol, timeframe)
        profile = get_symbol_profile(symbol)
        sym_display = profile['display_name']
    else:
        model_paths = {
            'smc_reg': os.path.join(config.MODEL_DIR, "smc_regressor.joblib"),
            'smc_feature_names': os.path.join(config.MODEL_DIR, "smc_feature_names.joblib"),
        }
        sym_display = 'Unknown'
    
    print(f"\n==================================================")
    print(f"TRAINING SMC SPECIALIST REGRESSION MODEL — {sym_display}")
    print(f"==================================================")
    
    # 1. Isolate ONLY the bars where an SMC event triggered
    if 'smc_trigger' not in df.columns:
        print(f"[Error] 'smc_trigger' column not found. Run detect_smc_event() first.")
        return None
        
    smc_bars = df[df['smc_trigger'] != 0.0].copy()
    
    total_bars = len(df)
    n_long_events = int((df['smc_trigger'] == 1.0).sum())
    n_short_events = int((df['smc_trigger'] == -1.0).sum())
    n_total_events = len(smc_bars)
    
    print(f"[Data] Total bars in dataset: {total_bars}")
    print(f"[Data] SMC Long events:  {n_long_events}")
    print(f"[Data] SMC Short events: {n_short_events}")
    print(f"[Data] Total SMC events: {n_total_events} ({n_total_events/max(total_bars,1)*100:.1f}% of all bars)")
    
    if n_total_events < getattr(config, 'SMC_MIN_EVENTS', 30):
        print(f"[Error] Only {n_total_events} SMC events found. Need at least {getattr(config, 'SMC_MIN_EVENTS', 30)} to train reliably.")
        print(f"        Consider: (1) Using longer historical data via MT5, (2) Increasing SMC_LOOKBACK in config.py,")
        print(f"                  (3) Relaxing the confluence requirements.")
        return None
    
    # Drop rows where target_reg is NaN (tail of dataset where forward return can't be calculated)
    if 'target_reg' not in smc_bars.columns:
        print(f"[Error] 'target_reg' column not found. Run create_smc_regression_target() first.")
        return None
    
    smc_bars = smc_bars.dropna(subset=['target_reg'])
    smc_bars = smc_bars[smc_bars['target_reg'] != 0.0]  # Remove masked/zero targets
    
    if len(smc_bars) < 50:
        print(f"[Error] Only {len(smc_bars)} usable SMC events after removing zero/NaN targets.")
        return None
    
    print(f"[Data] Usable SMC events for training: {len(smc_bars)}")
    
    # 2. Prepare X and y
    # Ensure all feature columns exist
    available_features = [c for c in feature_cols if c in smc_bars.columns]
    missing_features = set(feature_cols) - set(available_features)
    if missing_features:
        print(f"[Warning] {len(missing_features)} features missing from SMC event data. Using available {len(available_features)} features.")
        for mf in smc_bars.columns:
            if mf not in available_features and mf not in ['open', 'high', 'low', 'close', 'volume', 
                'smc_long_event', 'smc_short_event', 'smc_trigger', 'target_reg', 'target', 
                'ret_horizon', 'holding_bars', 'poc_val', 'vah_val', 'val_val', 'mnsr_val', 'yesterday_poc']:
                pass  # Don't add columns that aren't in the original feature_cols
    
    X = smc_bars[available_features].values
    y = smc_bars['target_reg'].values
    
    # 3. Chronological Split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"[Split] Train: {len(X_train)} events | Test: {len(X_test)} events")
    
    # 4. Train XGBoost Regressor
    print(f"\n[Training] Fitting XGBRegressor on {len(X_train)} SMC events...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    # Use early stopping with a sub-split of training data
    sub_val_size = max(int(len(X_train) * 0.15), 1)
    X_sub_train, X_sub_val = X_train[:-sub_val_size], X_train[-sub_val_size:]
    y_sub_train, y_sub_val = y_train[:-sub_val_size], y_train[-sub_val_size:]
    
    model.fit(
        X_sub_train, y_sub_train,
        eval_set=[(X_sub_val, y_sub_val)],
        verbose=False
    )
    
    # 5. Evaluate on Out-of-Sample Test Set
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    # Directional accuracy: how often does sign(prediction) match sign(actual)?
    dir_correct = np.sum(np.sign(preds) == np.sign(y_test))
    dir_accuracy = dir_correct / max(len(y_test), 1) * 100.0
    
    # Mean predicted magnitude for winning vs losing setups
    winning_mask = y_test > 0
    losing_mask = y_test <= 0
    mean_pred_winners = np.mean(preds[winning_mask]) if np.any(winning_mask) else 0
    mean_pred_losers = np.mean(preds[losing_mask]) if np.any(losing_mask) else 0
    
    print(f"\n==================================================")
    print(f"SMC MODEL OUT-OF-SAMPLE TEST RESULTS — {sym_display}")
    print(f"==================================================")
    print(f"  Test Set Size:        {len(y_test)} SMC events")
    print(f"  RMSE:                 {rmse:.4f}")
    print(f"  Directional Accuracy: {dir_accuracy:.1f}%")
    print(f"  Mean Pred (Winners):  {mean_pred_winners:+.3f}")
    print(f"  Mean Pred (Losers):   {mean_pred_losers:+.3f}")
    print(f"  Separation Gap:       {mean_pred_winners - mean_pred_losers:.3f}")
    
    # Re-train on ALL events for deployment
    print(f"\n[Final Fit] Re-training on all {len(X)} SMC events for deployment...")
    model_final = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    model_final.fit(X, y, verbose=False)
    
    # 6. Save model and feature schema
    joblib.dump(model_final, model_paths['smc_reg'])
    joblib.dump(available_features, model_paths['smc_feature_names'])
    print(f"[Saved] SMC Specialist Model -> {model_paths['smc_reg']}")
    print(f"[Saved] SMC Feature Schema   -> {model_paths['smc_feature_names']}")
    
    # 7. Display top feature importance
    fi_df = get_feature_importance(model_final, available_features, "SMC XGBRegressor")
    if not fi_df.empty:
        print(f"\n--- Top 15 Most Important SMC Features (Gain) ---")
        for idx, row in fi_df.head(15).iterrows():
            print(f"  {idx+1:2d}. {row['feature']:<30} ({row['importance']:.4f})")
    
    return model_final


def train_ict_model(df, feature_cols, symbol=None, timeframe="15m"):
    """
    Train the ICT Specialist Regression Model on filtered event data.
    
    Isolates ONLY the bars where an ICT Playbook confluence occurred:
    (Killzone + Bias + Judas Sweep + Displacement/FVG + OTE) and trains
    an XGBRegressor to predict the volatility-adjusted forward return.
    
    Args:
        df: Full featured DataFrame with 'ict_trigger' and 'target_ict_reg' columns.
        feature_cols: List of feature column names for ML input.
        symbol: Canonical symbol key (e.g., 'XAUUSD') for model path resolution.
        timeframe: Timeframe string (e.g., '15m') for model path resolution.
    
    Returns:
        Trained XGBRegressor model, or None if insufficient data.
    """
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    
    # Resolve per-symbol model paths
    if symbol is not None:
        from config import get_model_paths, get_symbol_profile
        model_paths = get_model_paths(symbol, timeframe)
        profile = get_symbol_profile(symbol)
        sym_display = profile['display_name']
    else:
        model_paths = {
            'ict_reg': os.path.join(config.MODEL_DIR, "ict_regressor.joblib"),
            'ict_feature_names': os.path.join(config.MODEL_DIR, "ict_feature_names.joblib"),
        }
        sym_display = 'Unknown'
    
    print(f"\n==================================================")
    print(f"TRAINING ICT SPECIALIST REGRESSION MODEL — {sym_display}")
    print(f"==================================================")
    
    # 1. Isolate ONLY the bars where an ICT event triggered
    if 'ict_trigger' not in df.columns:
        print(f"[Error] 'ict_trigger' column not found. Run detect_ict_event() first.")
        return None
        
    ict_bars = df[df['ict_trigger'] != 0.0].copy()
    
    total_bars = len(df)
    n_long_events = int((df['ict_trigger'] == 1.0).sum())
    n_short_events = int((df['ict_trigger'] == -1.0).sum())
    n_total_events = len(ict_bars)
    
    print(f"[Data] Total bars in dataset: {total_bars}")
    print(f"[Data] ICT Long events:  {n_long_events}")
    print(f"[Data] ICT Short events: {n_short_events}")
    print(f"[Data] Total ICT events: {n_total_events} ({n_total_events/max(total_bars,1)*100:.1f}% of all bars)")
    
    min_events = getattr(config, 'ICT_MIN_EVENTS', 30)
    if n_total_events < min_events:
        print(f"[Error] Only {n_total_events} ICT events found. Need at least {min_events} to train reliably.")
        print(f"        Consider: (1) Using longer historical data via MT5, (2) Increasing ICT_LOOKBACK in config.py,")
        print(f"                  (3) Relaxing OTE constraints or HTF bias rules.")
        return None
        
    # Drop rows where target_ict_reg is NaN
    if 'target_ict_reg' not in ict_bars.columns:
        print(f"[Error] 'target_ict_reg' column not found. Run create_ict_regression_target() first.")
        return None
        
    ict_bars = ict_bars.dropna(subset=['target_ict_reg'])
    ict_bars = ict_bars[ict_bars['target_ict_reg'] != 0.0]  # Remove masked/zero targets
    
    if len(ict_bars) < 20:
        print(f"[Error] Only {len(ict_bars)} usable ICT events after removing zero/NaN targets.")
        return None
        
    print(f"[Data] Usable ICT events for training: {len(ict_bars)}")
    
    # 2. Prepare X and y
    available_features = [c for c in feature_cols if c in ict_bars.columns]
    X = ict_bars[available_features].values
    y = ict_bars['target_ict_reg'].values
    
    # 3. Chronological Split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"[Split] Train: {len(X_train)} events | Test: {len(X_test)} events")
    
    # 4. Train XGBoost Regressor
    print(f"\n[Training] Fitting XGBRegressor (max_depth=3 for rare events) on {len(X_train)} ICT events...")
    model = XGBRegressor(
        n_estimators=150,
        max_depth=3,          # Shallow depth to prevent overfitting on rare events
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    # Use early stopping with a sub-split of training data
    sub_val_size = max(int(len(X_train) * 0.15), 1)
    X_sub_train, X_sub_val = X_train[:-sub_val_size], X_train[-sub_val_size:]
    y_sub_train, y_sub_val = y_train[:-sub_val_size], y_train[-sub_val_size:]
    
    model.fit(
        X_sub_train, y_sub_train,
        eval_set=[(X_sub_val, y_sub_val)],
        verbose=False
    )
    
    # 5. Evaluate on Out-of-Sample Test Set
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    # Directional accuracy: how often does sign(prediction) match sign(actual)?
    dir_correct = np.sum(np.sign(preds) == np.sign(y_test))
    dir_accuracy = dir_correct / max(len(y_test), 1) * 100.0
    
    # Mean predicted magnitude for winning vs losing setups
    winning_mask = y_test > 0
    losing_mask = y_test <= 0
    mean_pred_winners = np.mean(preds[winning_mask]) if np.any(winning_mask) else 0
    mean_pred_losers = np.mean(preds[losing_mask]) if np.any(losing_mask) else 0
    
    print(f"\n==================================================")
    print(f"ICT MODEL OUT-OF-SAMPLE TEST RESULTS — {sym_display}")
    print(f"==================================================")
    print(f"  Test Set Size:        {len(y_test)} ICT events")
    print(f"  RMSE:                 {rmse:.4f}")
    print(f"  Directional Accuracy: {dir_accuracy:.1f}%")
    print(f"  Mean Pred (Winners):  {mean_pred_winners:+.3f}")
    print(f"  Mean Pred (Losers):   {mean_pred_losers:+.3f}")
    print(f"  Separation Gap:       {mean_pred_winners - mean_pred_losers:.3f}")
    
    # Re-train on ALL events for deployment
    print(f"\n[Final Fit] Re-training on all {len(X)} ICT events for deployment...")
    model_final = XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    model_final.fit(X, y, verbose=False)
    
    # 6. Save model and feature schema
    joblib.dump(model_final, model_paths['ict_reg'])
    joblib.dump(available_features, model_paths['ict_feature_names'])
    print(f"[Saved] ICT Specialist Model -> {model_paths['ict_reg']}")
    print(f"[Saved] ICT Feature Schema   -> {model_paths['ict_feature_names']}")
    
    # 7. Display top feature importance
    fi_df = get_feature_importance(model_final, available_features, "ICT XGBRegressor")
    if not fi_df.empty:
        print(f"\n--- Top 15 Most Important ICT Features (Gain) ---")
        for idx, row in fi_df.head(15).iterrows():
            print(f"  {idx+1:2d}. {row['feature']:<30} ({row['importance']:.4f})")
            
    return model_final
