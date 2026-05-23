#!/usr/bin/env python3
"""
IndiaLend - Comprehensive Model Training Pipeline
==================================================

Trains multiple ML models on large-scale synthetic Indian credit data,
leveraging Apple Silicon Metal GPU for neural network training and
all CPU cores for tree-based models.

Models trained:
1. HistGradientBoosting (sklearn) - Fast gradient boosting
2. Random Forest (sklearn) - Bagging ensemble
3. Extra Trees (sklearn) - Randomized splits
4. Logistic Regression (sklearn) - Interpretable baseline
5. Neural Network (PyTorch Metal GPU) - Deep learning
6. Weighted Ensemble - Combines all models
7. LightGBM / XGBoost (if libomp available)

Usage:
    python train_models.py                    # Full training (200K data)
    python train_models.py --samples 500000   # Extra large dataset
    python train_models.py --epochs 200       # More neural network epochs
    python train_models.py --quick            # Quick test run (10K data)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def print_banner():
    """Print startup banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   IndiaLend - Multi-Model Credit Scoring Training Pipeline           ║
║   ═══════════════════════════════════════════════════════════         ║
║                                                                      ║
║   Apple Silicon M4 Optimized | Metal GPU + 10-core CPU               ║
║   Models: HistGBM, RF, ExtraTrees, LR, Neural Net, Ensemble         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def detect_hardware():
    """Detect and display hardware capabilities."""
    import platform

    print("=" * 70)
    print("  HARDWARE DETECTION")
    print("=" * 70)

    # CPU
    machine = platform.machine()
    cpu_count = os.cpu_count() or 1
    print(f"  Platform:  {platform.system()} {platform.release()}")
    print(f"  Arch:      {machine}")
    print(f"  CPU Cores: {cpu_count}")

    # Memory
    try:
        import subprocess
        mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        mem_gb = mem_bytes / (1024 ** 3)
        print(f"  RAM:       {mem_gb:.0f} GB")
    except Exception:
        mem_gb = 16
        print(f"  RAM:       (unable to detect)")

    # Metal GPU
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
        print(f"  PyTorch:   {torch.__version__}")
        print(f"  Metal GPU: {'Available (MPS)' if mps_available else 'Not available'}")
        if mps_available:
            # Quick benchmark
            device = torch.device("mps")
            x = torch.randn(5000, 5000, device=device)
            start = time.time()
            _ = torch.mm(x, x)
            torch.mps.synchronize()
            gpu_time = time.time() - start
            print(f"  GPU Bench: 5000x5000 matmul in {gpu_time*1000:.1f}ms")
    except ImportError:
        mps_available = False
        print(f"  PyTorch:   Not installed")
        print(f"  Metal GPU: Unavailable (install torch)")

    # sklearn
    import sklearn
    print(f"  sklearn:   {sklearn.__version__}")

    # Optional: LightGBM, XGBoost
    for lib_name in ["lightgbm", "xgboost"]:
        try:
            lib = __import__(lib_name)
            version = lib.__version__
            # Test if it actually works (libomp issue on macOS)
            try:
                if lib_name == "lightgbm":
                    m = lib.LGBMClassifier(n_estimators=2, verbose=-1)
                    m.fit([[1, 2], [3, 4]], [0, 1])
                    status = f"{version} (working)"
                else:
                    m = lib.XGBClassifier(n_estimators=2, verbosity=0)
                    m.fit([[1, 2], [3, 4]], [0, 1])
                    status = f"{version} (working)"
            except Exception:
                status = f"{version} (installed but libomp missing)"
        except ImportError:
            status = "not installed"
        print(f"  {lib_name:10s}: {status}")

    print("=" * 70)
    print()

    return {
        "cpu_count": cpu_count,
        "mem_gb": mem_gb,
        "mps_available": mps_available,
        "machine": machine,
    }


def generate_data(n_samples: int, seed: int = 42) -> tuple:
    """Generate large-scale synthetic data and split into train/val/test."""
    from indialend.utils import generate_synthetic_data
    from sklearn.model_selection import train_test_split

    print(f"\n{'='*70}")
    print(f"  STEP 1: DATA GENERATION ({n_samples:,} samples)")
    print(f"{'='*70}")

    start = time.time()
    df = generate_synthetic_data(
        n_samples=n_samples,
        seed=seed,
        default_rate=0.08,
        include_alternative=True,
        include_bureau_history=True,
    )
    gen_time = time.time() - start
    print(f"  Generated {len(df):,} rows x {len(df.columns)} columns in {gen_time:.1f}s")
    print(f"  Default rate: {df['default_flag'].mean():.2%}")
    print(f"  CAT segment: {(df['cat_segment'] == 'CAT').mean():.1%}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Feature columns (all numeric features relevant for scoring)
    feature_cols = [
        "age", "education_numeric", "dependents", "years_employed",
        "years_current_job", "annual_income", "monthly_income",
        "existing_emi", "num_existing_loans", "credit_card_outstanding",
        "credit_card_limit", "credit_utilization", "loan_amount",
        "tenure_months", "ltv_ratio", "foir", "cibil_score",
        "dpd_30_12m", "dpd_60_12m", "dpd_90_12m",
        "enquiries_3m", "enquiries_6m", "enquiries_12m",
        "oldest_account_months", "avg_account_age_months",
        "written_off_accounts", "settled_accounts",
    ]

    # Add alternative data features if available
    alt_features = [
        "upi_txn_count_3m", "upi_txn_value_3m", "upi_unique_merchants",
        "upi_inflow_ratio", "phone_age_months", "avg_recharge_amount",
        "data_usage_gb", "app_install_count", "ecom_orders_6m",
        "ecom_spend_6m", "ecom_return_rate",
    ]
    feature_cols += [f for f in alt_features if f in df.columns]

    # Add bureau history features if available
    bureau_features = [
        "total_accounts", "active_accounts", "closed_accounts",
        "secured_accounts", "unsecured_accounts", "total_outstanding",
        "total_sanctioned", "max_dpd_ever", "suit_filed",
    ]
    feature_cols += [f for f in bureau_features if f in df.columns]

    available = [f for f in feature_cols if f in df.columns]
    print(f"  Features: {len(available)} selected")

    X = df[available]
    y = df["default_flag"]

    # Split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp,
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"  Default rates: train={y_train.mean():.2%} val={y_val.mean():.2%} test={y_test.mean():.2%}")

    # Save data
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_credit_data.csv", index=False)
    print(f"  Saved to data/synthetic_credit_data.csv")

    return X_train, X_val, X_test, y_train, y_val, y_test, available, df


def train_ensemble(X_train, X_val, y_train, y_val, features, n_jobs=-1):
    """Train the multi-model ensemble."""
    from indialend.ensemble_scorer import EnsembleCreditScorer

    print(f"\n{'='*70}")
    print(f"  STEP 2: ENSEMBLE TRAINING (sklearn models on {os.cpu_count()} CPU cores)")
    print(f"{'='*70}")

    ensemble = EnsembleCreditScorer(
        ensemble_method="weighted",
        calibrate=True,
        features=features,
        n_jobs=n_jobs,
        verbose=True,
    )
    ensemble.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    return ensemble


def train_neural(X_train, X_val, y_train, y_val, features, epochs=100, batch_size=2048):
    """Train the neural network on Metal GPU."""
    from indialend.neural_scorer import NeuralCreditScorer

    print(f"\n{'='*70}")
    print(f"  STEP 3: NEURAL NETWORK TRAINING (PyTorch + Apple Metal GPU)")
    print(f"{'='*70}")

    neural = NeuralCreditScorer(
        hidden_dims=[512, 256, 128, 64],
        dropout_rates=[0.3, 0.25, 0.2, 0.1],
        use_residual=True,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        weight_decay=0.01,
        patience=20,
        focal_alpha=0.25,
        focal_gamma=2.0,
        label_smoothing=0.02,
        device="auto",
        verbose=True,
    )
    neural.fit(X_train, y_train, X_val=X_val, y_val=y_val, features=features)

    return neural


def train_original_scorer(X_train, X_val, y_train, y_val, features):
    """Train the original CreditScorer (HistGBM with CAT/NonCAT segments)."""
    from indialend.credit_scoring import CreditScorer

    print(f"\n{'='*70}")
    print(f"  STEP 4: SEGMENTED SCORER (CAT/NonCAT - RBI compliant)")
    print(f"{'='*70}")

    scorer = CreditScorer(
        model_type="lightgbm",
        cat_type="auto",
        features=features,
        n_estimators=800,
        learning_rate=0.03,
        max_depth=7,
        min_child_samples=40,
    )

    start = time.time()
    scorer.fit(X_train, y_train, calibrate=True)
    train_time = time.time() - start

    print(f"  Model type (after fallback): {scorer.model_type}")
    print(f"  Training time: {train_time:.1f}s")

    metrics = scorer.evaluate(X_val, y_val)
    print(f"  Val AUC: {metrics['auc_roc']:.4f} | KS: {metrics['ks_statistic']:.4f} | Gini: {metrics['gini_coefficient']:.4f}")

    return scorer


def cross_validate_best(X_train, y_train, features, n_folds=5):
    """Run cross-validation on the best model configuration."""
    from indialend.credit_scoring import CreditScorer

    print(f"\n{'='*70}")
    print(f"  STEP 5: CROSS-VALIDATION ({n_folds}-fold stratified)")
    print(f"{'='*70}")

    scorer = CreditScorer(
        model_type="lightgbm",
        cat_type="combined",
        features=features,
        n_estimators=800,
        learning_rate=0.03,
        max_depth=7,
    )

    cv_results = scorer.cross_validate(X_train, y_train, n_folds=n_folds)

    print(f"  AUC:  {cv_results['mean_auc']:.4f} +/- {cv_results['std_auc']:.4f}")
    print(f"  KS:   {cv_results['mean_ks']:.4f} +/- {cv_results['std_ks']:.4f}")
    print(f"  Gini: {cv_results['mean_gini']:.4f} +/- {cv_results['std_gini']:.4f}")

    for fold_result in cv_results["fold_results"]:
        print(f"    Fold {fold_result['fold']}: AUC={fold_result['auc_roc']:.4f} KS={fold_result['ks_statistic']:.4f}")

    return cv_results


def evaluate_all(
    ensemble, neural, segmented,
    X_test, y_test, features,
):
    """Evaluate all models on the held-out test set."""
    from scipy.stats import ks_2samp
    from sklearn.metrics import roc_auc_score, f1_score

    print(f"\n{'='*70}")
    print(f"  STEP 6: FINAL EVALUATION (held-out test set, n={len(X_test):,})")
    print(f"{'='*70}")

    results = []

    # Individual ensemble models
    for name in ensemble._fitted_models:
        try:
            probs = ensemble._predict_single(name, X_test)
            auc = roc_auc_score(y_test, probs)
            ks, _ = ks_2samp(probs[y_test == 0], probs[y_test == 1])
            gini = 2 * auc - 1
            preds = (probs >= 0.5).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            results.append({
                "Model": name,
                "AUC": round(auc, 4),
                "KS": round(ks, 4),
                "Gini": round(gini, 4),
                "F1": round(f1, 4),
            })
        except Exception:
            pass

    # Ensemble
    ens_probs = ensemble.predict_proba(X_test)
    ens_auc = roc_auc_score(y_test, ens_probs)
    ens_ks, _ = ks_2samp(ens_probs[y_test == 0], ens_probs[y_test == 1])
    results.append({
        "Model": "Ensemble (weighted)",
        "AUC": round(ens_auc, 4),
        "KS": round(ens_ks, 4),
        "Gini": round(2 * ens_auc - 1, 4),
        "F1": round(f1_score(y_test, (ens_probs >= 0.5).astype(int), zero_division=0), 4),
    })

    # Neural network
    if neural is not None:
        nn_probs = neural.predict_proba(X_test)
        nn_auc = roc_auc_score(y_test, nn_probs)
        nn_ks, _ = ks_2samp(nn_probs[y_test == 0], nn_probs[y_test == 1])
        results.append({
            "Model": "Neural Network (Metal)",
            "AUC": round(nn_auc, 4),
            "KS": round(nn_ks, 4),
            "Gini": round(2 * nn_auc - 1, 4),
            "F1": round(f1_score(y_test, (nn_probs >= 0.5).astype(int), zero_division=0), 4),
        })

    # Segmented scorer
    seg_probs = segmented.predict_proba(X_test)
    seg_auc = roc_auc_score(y_test, seg_probs)
    seg_ks, _ = ks_2samp(seg_probs[y_test == 0], seg_probs[y_test == 1])
    results.append({
        "Model": "Segmented (CAT/NonCAT)",
        "AUC": round(seg_auc, 4),
        "KS": round(seg_ks, 4),
        "Gini": round(2 * seg_auc - 1, 4),
        "F1": round(f1_score(y_test, (seg_probs >= 0.5).astype(int), zero_division=0), 4),
    })

    # MEGA ENSEMBLE: combine all models
    all_probs = [ens_probs]
    if neural is not None:
        all_probs.append(nn_probs)
    all_probs.append(seg_probs)
    mega_probs = np.mean(all_probs, axis=0)
    mega_auc = roc_auc_score(y_test, mega_probs)
    mega_ks, _ = ks_2samp(mega_probs[y_test == 0], mega_probs[y_test == 1])
    results.append({
        "Model": "** MEGA ENSEMBLE **",
        "AUC": round(mega_auc, 4),
        "KS": round(mega_ks, 4),
        "Gini": round(2 * mega_auc - 1, 4),
        "F1": round(f1_score(y_test, (mega_probs >= 0.5).astype(int), zero_division=0), 4),
    })

    # Print comparison table
    df_results = pd.DataFrame(results).sort_values("AUC", ascending=False)
    print()
    print(f"  {'Model':<28} {'AUC':>8} {'KS':>8} {'Gini':>8} {'F1':>8}")
    print(f"  {'-'*60}")
    for _, row in df_results.iterrows():
        marker = " <<" if row["Model"].startswith("**") else ""
        print(
            f"  {row['Model']:<28} {row['AUC']:>8.4f} {row['KS']:>8.4f} "
            f"{row['Gini']:>8.4f} {row['F1']:>8.4f}{marker}"
        )
    print()

    return df_results


def save_models(ensemble, neural, segmented, cv_results):
    """Save all trained models."""

    print(f"\n{'='*70}")
    print(f"  STEP 7: SAVING MODELS")
    print(f"{'='*70}")

    os.makedirs("models", exist_ok=True)

    # Ensemble
    ensemble.save("models/ensemble_scorer.joblib")
    size = os.path.getsize("models/ensemble_scorer.joblib") / 1024 / 1024
    print(f"  Ensemble:  models/ensemble_scorer.joblib ({size:.1f} MB)")

    # Neural network
    if neural is not None:
        neural.save("models/neural_scorer.joblib")
        size = os.path.getsize("models/neural_scorer.joblib") / 1024 / 1024
        print(f"  Neural:    models/neural_scorer.joblib ({size:.1f} MB)")

    # Segmented scorer
    segmented.save("models/segmented_scorer.joblib")
    size = os.path.getsize("models/segmented_scorer.joblib") / 1024 / 1024
    print(f"  Segmented: models/segmented_scorer.joblib ({size:.1f} MB)")

    # Save training summary
    import json
    summary = {
        "models_trained": list(ensemble._fitted_models.keys()) + ["NeuralNet", "Segmented"],
        "ensemble_weights": {k: round(v, 4) for k, v in ensemble._model_weights.items()},
        "cv_mean_auc": cv_results["mean_auc"],
        "cv_std_auc": cv_results["std_auc"],
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("models/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:   models/training_summary.json")
    print()


def print_final_summary(
    hw_info, n_samples, results_df, cv_results, total_time,
    ensemble, neural, features,
):
    """Print final training summary."""
    print()
    print("=" * 70)
    print("  TRAINING COMPLETE - FINAL SUMMARY")
    print("=" * 70)
    print()
    print(f"  Hardware:        Apple Silicon ({hw_info['machine']}) "
          f"| {hw_info['cpu_count']} cores | {hw_info['mem_gb']:.0f} GB RAM")
    print(f"  GPU:             {'Metal (MPS)' if hw_info['mps_available'] else 'CPU only'}")
    print(f"  Dataset:         {n_samples:,} samples | {len(features)} features")
    print(f"  Models Trained:  {len(ensemble._fitted_models)} base + Neural + Segmented + Ensemble")
    print(f"  Total Time:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print()

    # Best model
    best = results_df.iloc[0]
    print(f"  Best Model:      {best['Model']}")
    print(f"  Best AUC:        {best['AUC']:.4f}")
    print(f"  Best KS:         {best['KS']:.4f}")
    print(f"  Best Gini:       {best['Gini']:.4f}")
    print()

    # CV stats
    print(f"  CV AUC:          {cv_results['mean_auc']:.4f} +/- {cv_results['std_auc']:.4f}")
    print(f"  CV KS:           {cv_results['mean_ks']:.4f} +/- {cv_results['std_ks']:.4f}")
    print()

    if neural is not None:
        print(f"  Neural Epochs:   {len(neural.training_history)}")
        print(f"  Neural Best AUC: {neural._best_auc:.4f}")
        print(f"  Neural Params:   {sum(p.numel() for p in neural._model.parameters()):,}")
    print()

    # Model files
    total_model_size = 0
    for f in os.listdir("models"):
        fpath = os.path.join("models", f)
        if os.path.isfile(fpath):
            total_model_size += os.path.getsize(fpath)
    print(f"  Model Files:     models/ ({total_model_size / 1024 / 1024:.1f} MB total)")
    print(f"  Data File:       data/synthetic_credit_data.csv")
    print()
    print("=" * 70)
    print("  IndiaLend models are trained and ready!")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description="IndiaLend Model Training Pipeline")
    parser.add_argument("--samples", type=int, default=200_000,
                        help="Number of synthetic samples (default: 200000)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Neural network epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Neural network batch size (default: 2048)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Cross-validation folds (default: 5)")
    parser.add_argument("--no-neural", action="store_true",
                        help="Skip neural network training")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run with 10K samples")
    args = parser.parse_args()

    if args.quick:
        args.samples = 10_000
        args.epochs = 20

    print_banner()
    total_start = time.time()

    # Step 0: Hardware detection
    hw_info = detect_hardware()

    # Step 1: Generate data
    X_train, X_val, X_test, y_train, y_val, y_test, features, df = generate_data(
        n_samples=args.samples, seed=args.seed,
    )

    # Step 2: Train ensemble (sklearn models, all CPU cores)
    ensemble = train_ensemble(X_train, X_val, y_train, y_val, features)

    # Step 3: Train neural network (Metal GPU)
    neural = None
    if not args.no_neural and hw_info.get("mps_available", False):
        neural = train_neural(
            X_train, X_val, y_train, y_val, features,
            epochs=args.epochs, batch_size=args.batch_size,
        )
    elif not args.no_neural:
        print("\n  [INFO] Metal GPU not available, training neural network on CPU...")
        neural = train_neural(
            X_train, X_val, y_train, y_val, features,
            epochs=args.epochs, batch_size=args.batch_size,
        )

    # Step 4: Train segmented scorer (CAT/NonCAT)
    segmented = train_original_scorer(X_train, X_val, y_train, y_val, features)

    # Step 5: Cross-validation
    cv_results = cross_validate_best(X_train, y_train, features, n_folds=args.n_folds)

    # Step 6: Final evaluation on test set
    results_df = evaluate_all(ensemble, neural, segmented, X_test, y_test, features)

    # Step 7: Save models
    save_models(ensemble, neural, segmented, cv_results)

    # Final summary
    total_time = time.time() - total_start
    print_final_summary(
        hw_info, args.samples, results_df, cv_results, total_time,
        ensemble, neural, features,
    )


if __name__ == "__main__":
    main()
