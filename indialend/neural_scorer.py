"""
IndiaLend Neural Credit Scorer
==============================
Deep neural network for credit scoring, optimized for Apple Silicon Metal GPU.

Uses PyTorch with MPS (Metal Performance Shaders) backend to leverage
the M-series GPU for training and inference. Architecture designed
specifically for tabular credit data with:

- Residual connections for gradient flow
- BatchNorm + SiLU activation (better than ReLU for tabular)
- Dropout regularization per layer
- Focal loss for class imbalance (common in credit: ~8% default)
- Cosine annealing LR schedule
- Early stopping with patience
- Mixed precision where supported

Indian market context:
    Most NBFCs rely solely on tree-based models (LightGBM/XGBoost).
    Neural networks capture non-linear feature interactions that trees
    miss, especially for complex bureau + alternative data combinations.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    TORCH_AVAILABLE = True
    _nn_Module = nn.Module
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore
    WeightedRandomSampler = None  # type: ignore

    # Stub base class so class definitions below don't fail when torch is missing.
    # Subclasses raise ImportError on instantiation via the guards in their __init__.
    class _nn_Module:  # type: ignore
        pass

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from .utils import risk_grade_from_score


def get_device():
    """Get the best available device: MPS (Metal) > CUDA > CPU."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required: pip install torch")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Focal Loss (handles class imbalance better than BCE)
# ---------------------------------------------------------------------------

class FocalLoss(_nn_Module):
    """Focal Loss for imbalanced classification.

    Focuses training on hard-to-classify examples (common in credit:
    default rate ~5-10% means most samples are easy negatives).

    Parameters
    ----------
    alpha : float
        Weighting factor for positive class.
    gamma : float
        Focusing parameter. Higher = more focus on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch")
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(_nn_Module):
    """Residual block with BatchNorm + SiLU + Dropout.

    Residual connections help gradient flow in deeper networks,
    critical for tabular data where depth matters less than width.
    """

    def __init__(self, dim: int, dropout: float = 0.2):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch")
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


# ---------------------------------------------------------------------------
# Credit Network Architecture
# ---------------------------------------------------------------------------

class CreditNet(_nn_Module):
    """Deep neural network for credit scoring.

    Architecture: Input -> [Linear -> BN -> SiLU -> Dropout] x N -> Residual -> Output (logits)

    Outputs raw logits (no sigmoid) - pair with BCEWithLogitsLoss for numerical
    stability with imbalanced data. Apply torch.sigmoid() manually for probabilities.

    Designed for tabular credit data (27-80 features).
    Uses SiLU (Swish) activation which outperforms ReLU on tabular data.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list of int
        Sizes of hidden layers.
    dropout_rates : list of float
        Dropout rate per hidden layer.
    use_residual : bool
        Add residual blocks after main layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rates: Optional[List[float]] = None,
        use_residual: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch")
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        if dropout_rates is None or len(dropout_rates) != len(hidden_dims):
            n = len(hidden_dims)
            dropout_rates = [max(round(0.3 - i * 0.05, 3), 0.05) for i in range(n)]

        layers = []
        prev_dim = input_dim

        # Input batch normalization (helps with feature scale differences)
        layers.append(nn.BatchNorm1d(input_dim))

        for dim, drop in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(drop))
            prev_dim = dim

        # Optional residual block for deeper feature learning
        if use_residual and len(hidden_dims) >= 2:
            layers.append(ResidualBlock(prev_dim, dropout=dropout_rates[-1]))

        # Output layer returns logits (no sigmoid - applied in loss/inference)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.network(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Neural Credit Scorer (Main API)
# ---------------------------------------------------------------------------

class NeuralCreditScorer:
    """Credit scorer using PyTorch neural network with Metal GPU acceleration.

    Trains a deep neural network on credit application data, leveraging
    Apple Silicon's Metal GPU (MPS backend) for fast training.

    Parameters
    ----------
    hidden_dims : list of int
        Hidden layer sizes. Default: [512, 256, 128, 64]
    epochs : int
        Maximum training epochs. Default: 100
    batch_size : int
        Training batch size. Default: 2048 (optimal for MPS)
    learning_rate : float
        Initial learning rate. Default: 1e-3
    weight_decay : float
        L2 regularization. Default: 0.01
    patience : int
        Early stopping patience (epochs). Default: 15
    focal_alpha : float
        Focal loss alpha (positive class weight). Default: 0.25
    focal_gamma : float
        Focal loss gamma (hard example focus). Default: 2.0
    device : str
        Device to use: "auto", "mps", "cuda", "cpu"
    verbose : bool
        Print training progress.

    Example
    -------
    >>> scorer = NeuralCreditScorer(epochs=50, verbose=True)
    >>> scorer.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    >>> scores = scorer.score(X_test)
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout_rates: Optional[List[float]] = None,
        use_residual: bool = True,
        epochs: int = 100,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        patience: int = 15,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.02,
        device: str = "auto",
        verbose: bool = True,
        random_state: int = 42,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch required for NeuralCreditScorer. "
                "Install: pip install torch"
            )

        self.hidden_dims = hidden_dims or [512, 256, 128, 64]
        # Auto-derive dropout rates if not provided or length mismatch
        if dropout_rates is None or len(dropout_rates) != len(self.hidden_dims):
            n = len(self.hidden_dims)
            # Decreasing dropout: higher dropout in early layers
            self.dropout_rates = [round(0.3 - (i * 0.05), 3) for i in range(n)]
            self.dropout_rates = [max(d, 0.05) for d in self.dropout_rates]
        else:
            self.dropout_rates = dropout_rates
        self.use_residual = use_residual
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.verbose = verbose
        self.random_state = random_state

        # Device setup
        if device == "auto":
            self._device = get_device()
        else:
            self._device = torch.device(device)

        # State
        self._model: Optional[CreditNet] = None
        self._scaler = StandardScaler()
        self._fitted = False
        self._training_history: List[Dict] = []
        self._best_auc = 0.0
        self._feature_names: List[str] = []
        self._input_dim: int = 0

        torch.manual_seed(random_state)
        if self._device.type == "mps":
            torch.mps.manual_seed(random_state)

    @property
    def device_name(self) -> str:
        """Human-readable device name."""
        if self._device.type == "mps":
            return "Apple Metal GPU (MPS)"
        elif self._device.type == "cuda":
            return f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        features: Optional[List[str]] = None,
    ) -> "NeuralCreditScorer":
        """Train the neural network.

        Parameters
        ----------
        X : DataFrame
            Training features.
        y : Series
            Binary target (1=default, 0=non-default).
        X_val : DataFrame, optional
            Validation features for early stopping.
        y_val : Series, optional
            Validation target.
        features : list of str, optional
            Feature columns to use. Uses all numeric if not specified.
        """
        # Select features
        if features is not None:
            available = [f for f in features if f in X.columns]
            X_use = X[available].copy()
            self._feature_names = available
        else:
            X_use = X.select_dtypes(include=[np.number]).copy()
            self._feature_names = list(X_use.columns)

        # Scale features
        X_scaled = self._scaler.fit_transform(X_use.values.astype(np.float32))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Raw binary labels (no smoothing - BCEWithLogitsLoss handles imbalance)
        y_orig = y.values.astype(np.float32)

        # Tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_orig, dtype=torch.float32)

        # Compute pos_weight for BCEWithLogitsLoss (standard imbalance handling)
        n_pos = int((y_orig == 1).sum())
        n_neg = len(y_orig) - n_pos
        pos_weight_value = n_neg / max(n_pos, 1)
        self._pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(self._device)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
            drop_last=True,
        )

        # Validation data (use raw labels, no smoothing)
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_use = X_val[self._feature_names].copy() if features else \
                X_val.select_dtypes(include=[np.number]).copy()
            X_val_scaled = self._scaler.transform(X_val_use.values.astype(np.float32))
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
            y_val_raw = y_val.values.astype(np.float32)
            y_val_t = torch.tensor(y_val_raw, dtype=torch.float32)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2)

        # Build model
        input_dim = X_scaled.shape[1]
        self._input_dim = input_dim
        self._model = CreditNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rates=self.dropout_rates,
            use_residual=self.use_residual,
        ).to(self._device)

        # BCEWithLogitsLoss with pos_weight - stable for imbalanced data
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)
        optimizer = optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        # Training loop
        best_val_auc = 0.0
        best_state = None
        patience_counter = 0

        if self.verbose:
            total_params = sum(p.numel() for p in self._model.parameters())
            print(f"\n{'='*70}")
            print(f"  Neural Credit Scorer - Training on {self.device_name}")
            print(f"{'='*70}")
            print(f"  Architecture: {self.hidden_dims} | Residual: {self.use_residual}")
            print(f"  Parameters: {total_params:,} | Features: {input_dim}")
            print(f"  Train: {len(X):,} | Val: {len(X_val) if X_val is not None else 0:,}")
            print(f"  Epochs: {self.epochs} | Batch: {self.batch_size} | LR: {self.learning_rate}")
            print(f"  Loss: BCEWithLogitsLoss | pos_weight: {pos_weight_value:.2f}")
            print(f"{'='*70}")
            print(f"  {'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Val AUC':>10} {'LR':>12} {'Time':>8}")
            print(f"  {'-'*65}")

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # --- Training ---
            self._model.train()
            train_losses = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                outputs = self._model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)

                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()
            avg_train_loss = np.mean(train_losses)

            # --- Validation ---
            val_loss = 0.0
            val_auc = 0.0

            if val_loader is not None:
                self._model.eval()
                val_losses = []
                all_probs = []
                all_targets = []

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self._device)
                        y_batch = y_batch.to(self._device)
                        logits = self._model(X_batch)
                        vloss = criterion(logits, y_batch)
                        val_losses.append(vloss.item())
                        # Apply sigmoid for probabilities / AUC
                        probs = torch.sigmoid(logits)
                        all_probs.extend(probs.cpu().numpy())
                        all_targets.extend(y_batch.cpu().numpy())

                val_loss = np.mean(val_losses)
                try:
                    raw_targets = (np.array(all_targets) > 0.5).astype(int)
                    val_auc = roc_auc_score(raw_targets, np.array(all_probs))
                except ValueError:
                    val_auc = 0.5

                # Early stopping
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

            epoch_time = time.time() - epoch_start

            # Log
            current_lr = optimizer.param_groups[0]["lr"]
            self._training_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "lr": current_lr,
                "time": epoch_time,
            })

            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1 or patience_counter == 0):
                marker = " *" if patience_counter == 0 and val_loader is not None else ""
                print(
                    f"  {epoch+1:>5} {avg_train_loss:>12.6f} {val_loss:>12.6f} "
                    f"{val_auc:>10.4f} {current_lr:>12.8f} {epoch_time:>6.1f}s{marker}"
                )

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"\n  Early stopping at epoch {epoch+1} (patience={self.patience})")
                break

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(self._device)

        total_time = time.time() - start_time
        self._best_auc = best_val_auc
        self._fitted = True

        if self.verbose:
            print(f"\n  Training complete in {total_time:.1f}s")
            print(f"  Best validation AUC: {best_val_auc:.4f}")
            print(f"{'='*70}\n")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict default probability (applies sigmoid to logits)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba()")

        X_use = X[self._feature_names].copy() if self._feature_names else \
            X.select_dtypes(include=[np.number]).copy()
        X_scaled = self._scaler.transform(X_use.values.astype(np.float32))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)

        self._model.eval()
        with torch.no_grad():
            batch_size = self.batch_size * 4
            all_probs = []
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                logits = self._model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs)

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score applicants, returning CIBIL-scale scores + risk grades."""
        probs = self.predict_proba(X)
        scores = 900 - (probs * 600)
        scores = np.clip(scores, 300, 900).astype(int)
        grades = [risk_grade_from_score(s) for s in scores]

        return pd.DataFrame({
            "default_probability": np.round(probs, 6),
            "cibil_score": scores,
            "risk_grade": grades,
            "model": "neural_network",
        }, index=X.index)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance."""
        from scipy.stats import ks_2samp

        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y, probs)
        ks_stat, _ = ks_2samp(probs[y == 0], probs[y == 1])
        gini = 2 * auc - 1

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y, preds)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)

        return {
            "model": "NeuralCreditScorer",
            "device": self.device_name,
            "auc_roc": round(auc, 4),
            "ks_statistic": round(ks_stat, 4),
            "gini_coefficient": round(gini, 4),
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "best_val_auc": round(self._best_auc, 4),
            "epochs_trained": len(self._training_history),
            "total_params": sum(p.numel() for p in self._model.parameters()),
        }

    @property
    def training_history(self) -> pd.DataFrame:
        """Return training history as DataFrame."""
        if not self._training_history:
            return pd.DataFrame()
        return pd.DataFrame(self._training_history)

    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib
        joblib.dump({
            "model_state": {k: v.cpu() for k, v in self._model.state_dict().items()},
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "hidden_dims": self.hidden_dims,
            "dropout_rates": self.dropout_rates,
            "use_residual": self.use_residual,
            "best_auc": self._best_auc,
            "training_history": self._training_history,
            "input_dim": self._input_dim,
        }, path)

    @classmethod
    def load(cls, path: str) -> "NeuralCreditScorer":
        """Load model from disk."""
        import joblib
        data = joblib.load(path)

        scorer = cls(
            hidden_dims=data["hidden_dims"],
            dropout_rates=data["dropout_rates"],
            use_residual=data["use_residual"],
        )
        scorer._scaler = data["scaler"]
        scorer._feature_names = data["feature_names"]
        scorer._best_auc = data["best_auc"]
        scorer._training_history = data["training_history"]

        # Rebuild model
        scorer._model = CreditNet(
            input_dim=data["input_dim"],
            hidden_dims=data["hidden_dims"],
            dropout_rates=data["dropout_rates"],
            use_residual=data["use_residual"],
        )
        scorer._model.load_state_dict(data["model_state"])
        scorer._model.to(scorer._device)
        scorer._fitted = True

        return scorer
