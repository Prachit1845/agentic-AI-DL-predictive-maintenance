import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd

from src.dataset import prepare_preprocessed_data, create_sequences, create_test_sequences
from src.prognostics import LSTMRULEstimator, train_lstm_model, evaluate_metrics, predict_monte_carlo_dropout

def main():
    print("=" * 75)
    print("STARTING DIRECT TRAINING ON NASA C-MAPSS FD001 TURBOFAN DATASET")
    print("=" * 75)

    # 1. Dataset Loading and Verification
    print("\n[Step 1/4] Loading and verifying dataset files...")
    train_scaled, test_scaled, rul_df, scaler = prepare_preprocessed_data()
    print(f"  Training records : {train_scaled.shape[0]} rows (100 run-to-failure engines)")
    print(f"  Testing records  : {test_scaled.shape[0]} rows (100 test engines)")
    print(f"  Ground truth     : {rul_df.shape[0]} true RUL targets")

    # 2. Sequence Generation
    print("\n[Step 2/4] Generating temporal sliding-window sequences (Window W = 30)...")
    X_train, y_train = create_sequences(train_scaled, seq_len=30)
    X_test, y_test, test_ids = create_test_sequences(test_scaled, rul_df, seq_len=30)
    print(f"  Input sequence shape (X_train): {X_train.shape}")
    print(f"  Target label shape   (y_train): {y_train.shape}")
    print(f"  Test sequence shape  (X_test) : {X_test.shape}")

    # Split into 85% train and 15% validation
    val_split = int(len(X_train) * 0.85)
    X_tr, y_tr = X_train[:val_split], y_train[:val_split]
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    print(f"  Training set size   : {len(X_tr)} sequences")
    print(f"  Validation set size : {len(X_val)} sequences")

    # 3. Model Training
    print("\n[Step 3/4] Training PyTorch LSTM RUL Estimator on CPU...")
    print("  Architecture: Input(14) -> LSTM(64) -> Dropout(0.2) -> Dense(32) -> ReLU -> Dropout(0.2) -> Linear(1)")
    print("  Loss Function: MSE | Optimizer: Adam (lr=0.001) | Batch Size: 128 | Epochs: 15\n")

    model, history = train_lstm_model(
        X_tr, y_tr, X_val, y_val,
        hidden_dim=64, num_layers=1, dropout=0.2, lr=0.001, epochs=15, batch_size=128, verbose=True
    )

    # Save trained model weights
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "best_lstm_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n  Saved trained weights to: {model_save_path} ({os.path.getsize(model_save_path)} bytes)")

    # 4. Evaluation on Test Dataset
    print("\n[Step 4/4] Evaluating trained model on 100 test engines...")
    mcd_results = predict_monte_carlo_dropout(model, X_test, n_samples=30)
    test_metrics = evaluate_metrics(y_test, mcd_results["mean"])

    print("\n" + "=" * 75)
    print("TRAINING AND EVALUATION RESULTS ON C-MAPSS DATASET:")
    print("=" * 75)
    print(f"  Root Mean Squared Error (RMSE) : {test_metrics['rmse']:.2f} cycles")
    print(f"  Mean Absolute Error (MAE)      : {test_metrics['mae']:.2f} cycles")
    print(f"  NASA Asymmetric Scoring Func   : {test_metrics['nasa_score']:.1f}")
    print(f"  Uncertainty Std Dev (Average)  : {float(np.mean(mcd_results['std'])):.2f} cycles")
    print("=" * 75)
    print("TRAINING COMPLETE! The model is fully trained on the C-MAPSS dataset.")

if __name__ == "__main__":
    main()
