import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pandas as pd
import json

from src.dataset import prepare_preprocessed_data, create_sequences, create_test_sequences, run_eda_and_save_visualizations
from src.prognostics import train_lstm_model, evaluate_metrics, predict_monte_carlo_dropout, LSTMRULEstimator
from src.agents import PrognosticAgent, ResourceAgent, CostAgent, CoordinatorAgent
from src.simulation import FleetMaintenanceSimulator, run_fleet_policy_comparison, run_sensitivity_analysis

def main():
    print("=" * 70)
    print("Agentic Predictive Maintenance Pipeline: End-to-End Training & Evaluation")
    print("=" * 70)
    
    # 1. Dataset Preprocessing & EDA
    print("\n[Phase 2 & 3] Loading C-MAPSS FD001 data and generating EDA visualizations...")
    train_scaled, test_scaled, rul_df, scaler = prepare_preprocessed_data()
    run_eda_and_save_visualizations(train_scaled)
    
    # 2. Experiment 1: Sequence Length Comparison (Window 30 vs Window 20)
    print("\n[Phase 4 / Exp 1] Comparing Sequence Length 30 vs 20...")
    
    # Model A: Window 30
    X_train_30, y_train_30 = create_sequences(train_scaled, seq_len=30)
    X_test_30, y_test_30, test_ids = create_test_sequences(test_scaled, rul_df, seq_len=30)
    
    # Split train into train/val
    split_idx_30 = int(len(X_train_30) * 0.85)
    X_tr_30, y_tr_30 = X_train_30[:split_idx_30], y_train_30[:split_idx_30]
    X_val_30, y_val_30 = X_train_30[split_idx_30:], y_train_30[split_idx_30:]
    
    print("\nTraining Model (Seq Len = 30, Hidden = 64, Dropout = 0.2)...")
    model_30, history_30 = train_lstm_model(
        X_tr_30, y_tr_30, X_val_30, y_val_30,
        hidden_dim=64, num_layers=1, dropout=0.2, lr=0.001, epochs=15, batch_size=128
    )
    
    # Save best model
    os.makedirs("models", exist_ok=True)
    torch.save(model_30.state_dict(), "models/best_lstm_model.pt")
    print("Saved best LSTM model weights to models/best_lstm_model.pt")
    
    # Model B: Window 20
    X_train_20, y_train_20 = create_sequences(train_scaled, seq_len=20)
    X_test_20, y_test_20, _ = create_test_sequences(test_scaled, rul_df, seq_len=20)
    split_idx_20 = int(len(X_train_20) * 0.85)
    X_tr_20, y_tr_20 = X_train_20[:split_idx_20], y_train_20[:split_idx_20]
    X_val_20, y_val_20 = X_train_20[split_idx_20:], y_train_20[split_idx_20:]
    
    print("\nTraining Model (Seq Len = 20, Hidden = 64, Dropout = 0.2)...")
    model_20, history_20 = train_lstm_model(
        X_tr_20, y_tr_20, X_val_20, y_val_20,
        hidden_dim=64, num_layers=1, dropout=0.2, lr=0.001, epochs=15, batch_size=128
    )
    
    # Evaluate both sequence models
    preds_30 = predict_monte_carlo_dropout(model_30, X_test_30, n_samples=30)
    preds_20 = predict_monte_carlo_dropout(model_20, X_test_20, n_samples=30)
    
    metrics_30 = evaluate_metrics(y_test_30, preds_30["mean"])
    metrics_20 = evaluate_metrics(y_test_20, preds_20["mean"])
    
    df_exp1 = pd.DataFrame([
        {"Sequence Length": 30, "MAE": round(metrics_30["mae"], 2), "RMSE": round(metrics_30["rmse"], 2), "NASA Score": round(metrics_30["nasa_score"], 1)},
        {"Sequence Length": 20, "MAE": round(metrics_20["mae"], 2), "RMSE": round(metrics_20["rmse"], 2), "NASA Score": round(metrics_20["nasa_score"], 1)}
    ])
    df_exp1.to_csv("results/exp1_sequence_length_comparison.csv", index=False)
    print("\nExperiment 1 Results (Sequence Length Comparison):")
    print(df_exp1)
    
    # 3. Experiment 2: Point RUL vs Uncertainty Interval
    print("\n[Phase 5 / Exp 2] Point RUL vs Uncertainty Interval...")
    point_preds = preds_30["mean"]
    lower_bounds = preds_30["lower_bound"]
    upper_bounds = preds_30["upper_bound"]
    
    in_interval = np.mean((y_test_30 >= lower_bounds) & (y_test_30 <= upper_bounds)) * 100.0
    avg_width = float(np.mean(upper_bounds - lower_bounds))
    
    exp2_summary = {
        "Empirical Coverage Rate (%)": round(in_interval, 2),
        "Average Uncertainty Interval Width (Cycles)": round(avg_width, 2),
        "Mean Model Uncertainty (Std Dev)": round(float(np.mean(preds_30["std"])), 2)
    }
    with open("results/exp2_uncertainty_metrics.json", "w") as f:
        json.dump(exp2_summary, f, indent=4)
    print("Experiment 2 Metrics:", exp2_summary)
    
    # 4. Multi-Agent Setup & Fleet Simulation
    print("\n[Phase 6 & 11] Initializing Agentic AI Layer and Simulating Fleet Policies...")
    prognostic_agent = PrognosticAgent(model=model_30, n_mc_samples=30)
    resource_agent = ResourceAgent(spare_lead_time=8, tech_capacity_per_cycle=2)
    cost_agent = CostAgent()
    coordinator_agent = CoordinatorAgent(prognostic_agent, resource_agent, cost_agent)
    
    # Ground truth fleet lifetimes from test engines
    # Current cycles: max observed cycle in test set for each engine
    test_current_cycles = test_scaled.groupby("engine_id")["cycle"].max().values
    true_lifetimes = test_current_cycles + y_test_30.astype(int)
    
    simulator = FleetMaintenanceSimulator(
        cost_pm=1000.0, cost_failure=5000.0, cost_downtime_per_cycle=100.0,
        spare_lead_time=8, tech_capacity_per_cycle=2, fixed_interval=125, naive_threshold=20
    )
    
    # Run 4-Policy Comparison
    policy_df = run_fleet_policy_comparison(
        simulator=simulator,
        coordinator=coordinator_agent,
        engine_ids=test_ids,
        current_cycles=list(test_current_cycles),
        sequences=X_test_30,
        true_lifetimes=list(true_lifetimes),
        predicted_ruls=list(preds_30["mean"])
    )
    print("\nFleet Policy Comparison Summary Table:")
    print(policy_df)
    
    # 5. Mandatory Experiment 6: Sensitivity Analysis
    print("\n[Exp 6] Running Sensitivity Analysis...")
    run_sensitivity_analysis(
        prognostic_agent=prognostic_agent,
        engine_ids=test_ids,
        current_cycles=list(test_current_cycles),
        sequences=X_test_30,
        true_lifetimes=list(true_lifetimes),
        predicted_ruls=list(preds_30["mean"])
    )
    
    # 6. Sample Section 13 Agent Decisions
    print("\n[Section 13] Generating Verified Sample Agent Decision Audits...")
    sample_engine_indices = [5, 16, 41]  # Representative test engines
    for idx in sample_engine_indices:
        eid = test_ids[idx]
        seq = X_test_30[idx]
        cur_cycle = int(test_current_cycles[idx])
        decision = coordinator_agent.coordinate_decision(
            engine_id=eid, current_cycle=cur_cycle, sensor_sequence=seq, book_slot=False
        )
        print("\n" + "-" * 50)
        print(decision["audit_trail_text"])
        print("-" * 50)
        
    print("\nPipeline execution complete! All artifacts and results saved.")

if __name__ == "__main__":
    main()
