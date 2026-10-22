import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure OpenMP runtime stability on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class LSTMRULEstimator(nn.Module):
    """
    Compact LSTM Architecture for Turbofan RUL Estimation.
    Architecture:
        Sequence Input (batch, seq_len, num_features)
        -> LSTM Layer (hidden_dim units, 1 or 2 layers, batch_first=True)
        -> Dropout (p=dropout_rate)
        -> Fully Connected Layer (hidden_dim -> 32)
        -> ReLU Activation
        -> Dropout (p=dropout_rate)
        -> Dense Output Layer (32 -> 1)
        -> Predicted Remaining Useful Life (RUL)
    """
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=1, dropout=0.2):
        super(LSTMRULEstimator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the hidden representation from the final time step
        last_step = lstm_out[:, -1, :]
        out = self.dropout1(last_step)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        rul_pred = self.fc_out(out)
        return rul_pred.squeeze(-1)


def compute_nasa_score(y_true, y_pred):
    """
    NASA C-MAPSS Asymmetric Prognostic Scoring Function:
    Penalizes late predictions much more severely than early predictions.
    d = y_pred - y_true
    s = exp(-d / 13) - 1  for d < 0 (early prediction, safe)
    s = exp(d / 10) - 1   for d >= 0 (late prediction, dangerous failure risk)
    """
    d = np.array(y_pred) - np.array(y_true)
    early_mask = d < 0
    late_mask = ~early_mask
    
    score_early = np.exp(-d[early_mask] / 13.0) - 1.0
    score_late = np.exp(d[late_mask] / 10.0) - 1.0
    
    return float(np.sum(score_early) + np.sum(score_late))


def evaluate_metrics(y_true, y_pred):
    """Computes MAE, RMSE, and NASA prognostic score."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    nasa_score = compute_nasa_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "nasa_score": nasa_score}


def train_lstm_model(X_train, y_train, X_val=None, y_val=None,
                     hidden_dim=64, num_layers=1, dropout=0.2,
                     lr=0.001, epochs=20, batch_size=128,
                     device=None, verbose=True):
    """
    Trains the LSTM RUL estimator using PyTorch.
    Returns:
        trained_model: PyTorch model object
        history: training and validation loss curves
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    input_dim = X_train.shape[2]
    model = LSTMRULEstimator(input_dim=input_dim, hidden_dim=hidden_dim,
                             num_layers=num_layers, dropout=dropout).to(device)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(batch_y)
            
        epoch_train_loss = running_loss / len(train_dataset)
        history["train_loss"].append(epoch_train_loss)
        
        # Validation
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
                val_y = torch.tensor(y_val, dtype=torch.float32).to(device)
                val_preds = model(val_x)
                epoch_val_loss = criterion(val_preds, val_y).item()
                history["val_loss"].append(epoch_val_loss)
                scheduler.step(epoch_val_loss)
                
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_model_state = model.state_dict().copy()
        else:
            epoch_val_loss = epoch_train_loss
            history["val_loss"].append(epoch_val_loss)
            
        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train MSE: {epoch_train_loss:.2f} | Val MSE: {epoch_val_loss:.2f}")
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, history


def enable_dropout_only(model):
    """
    Enables dropout layers during inference for Monte Carlo Dropout uncertainty estimation
    while keeping Batch Normalization / other training mechanisms in evaluation mode.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def predict_monte_carlo_dropout(model, X_input, n_samples=50, device=None):
    """
    Computes RUL predictions and uncertainty intervals using Monte Carlo Dropout (MCD).
    Args:
        model: PyTorch LSTMRULEstimator
        X_input: numpy array of shape (batch_size, seq_len, num_features) or (seq_len, num_features)
        n_samples: number of stochastic forward passes (default: 50)
    Returns:
        dict containing:
            mean: mean predicted RUL (scalar or array)
            std: uncertainty / standard deviation (scalar or array)
            lower_bound: conservative 95% lower bound (mu - 1.96*std) clipped to >= 0
            upper_bound: 95% upper bound (mu + 1.96*std)
            samples: raw stochastic sample predictions (n_samples, batch_size)
    """
    if device is None:
        device = next(model.parameters()).device
        
    if isinstance(X_input, np.ndarray):
        if X_input.ndim == 2:
            X_input = np.expand_dims(X_input, axis=0)
        tensor_x = torch.tensor(X_input, dtype=torch.float32).to(device)
    else:
        tensor_x = X_input.to(device)
        if tensor_x.ndim == 2:
            tensor_x = tensor_x.unsqueeze(0)
            
    enable_dropout_only(model)
    
    stochastic_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds = model(tensor_x).cpu().numpy()
            stochastic_preds.append(preds)
            
    # shape: (n_samples, batch_size)
    samples_array = np.array(stochastic_preds)
    
    mean_rul = np.mean(samples_array, axis=0)
    std_rul = np.std(samples_array, axis=0)
    lower_bound = np.maximum(0.0, mean_rul - 1.96 * std_rul)
    upper_bound = mean_rul + 1.96 * std_rul
    
    # Return single float scalars if batch size == 1
    if len(mean_rul) == 1:
        return {
            "mean": float(mean_rul[0]),
            "std": float(std_rul[0]),
            "lower_bound": float(lower_bound[0]),
            "upper_bound": float(upper_bound[0]),
            "samples": samples_array[:, 0]
        }
    else:
        return {
            "mean": mean_rul,
            "std": std_rul,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "samples": samples_array
        }
