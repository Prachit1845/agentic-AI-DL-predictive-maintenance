import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import torch
import numpy as np

from src.dataset import prepare_preprocessed_data, create_test_sequences
from src.prognostics import LSTMRULEstimator
from src.agents import PrognosticAgent, ResourceAgent, CostAgent, CoordinatorAgent, MessageBus

def main():
    parser = argparse.ArgumentParser(description="Agentic Predictive Maintenance Multi-Agent Decision Runner")
    parser.add_argument("--engine", type=int, default=17, help="Test Engine ID (1 to 100)")
    parser.add_argument("--lead_time", type=int, default=8, help="Spare-Part Lead Time in Cycles (default: 8)")
    parser.add_argument("--tech_capacity", type=int, default=2, help="Technicians Available per Cycle (default: 2)")
    parser.add_argument("--samples", type=int, default=40, help="Monte Carlo Dropout Samples (default: 40)")
    args = parser.parse_args()

    print("=" * 80)
    print("PS-05: MULTI-AGENT PREDICTIVE MAINTENANCE SCHEDULING SYSTEM")
    print("=" * 80)
    print(f"Loading environment and assets...")

    train_scaled, test_scaled, rul_df, scaler = prepare_preprocessed_data()
    X_test, y_test, test_ids = create_test_sequences(test_scaled, rul_df, seq_len=30)

    if args.engine not in test_ids:
        print(f"Error: Engine ID {args.engine} not found in test set. Available IDs: 1 to 100.")
        sys.exit(1)

    engine_idx = test_ids.index(args.engine)
    engine_seq = X_test[engine_idx]
    engine_data = test_scaled[test_scaled["engine_id"] == args.engine]
    current_cycle = int(engine_data["cycle"].max())
    true_remaining = int(y_test[engine_idx])

    # Load LSTM Model
    model = LSTMRULEstimator(input_dim=14, hidden_dim=64, num_layers=1, dropout=0.2)
    model_path = os.path.join("models", "best_lstm_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    else:
        print("Warning: Model weights not found. Please run train_and_evaluate.py first.")
        sys.exit(1)
    model.eval()

    # Instantiate Agents with central MessageBus
    bus = MessageBus()
    prognostic_agent = PrognosticAgent(model=model, n_mc_samples=args.samples, message_bus=bus)
    resource_agent = ResourceAgent(spare_lead_time=args.lead_time, tech_capacity_per_cycle=args.tech_capacity, message_bus=bus)
    cost_agent = CostAgent(cost_pm=1000.0, cost_failure=5000.0, cost_downtime_per_cycle=100.0, message_bus=bus)
    coordinator = CoordinatorAgent(prognostic_agent, resource_agent, cost_agent, message_bus=bus)

    print(f"\n[ENGINE TELEMETRY]")
    print(f"  Engine ID           : E-{str(args.engine).zfill(3)}")
    print(f"  Current Cycle       : {current_cycle}")
    print(f"  True Remaining Life : {true_remaining} cycles (Ground Truth)")
    print(f"  Spare Lead Time     : {args.lead_time} cycles")
    print(f"  Technician Slots    : {args.tech_capacity} / cycle")

    print("\n" + "=" * 80)
    print("STEP 1: MULTI-AGENT COMMUNICATION & NEGOTIATION TRANSCRIPT")
    print("=" * 80)

    decision = coordinator.coordinate_decision(
        engine_id=args.engine,
        current_cycle=current_cycle,
        sensor_sequence=engine_seq,
        book_slot=True
    )

    for msg in decision["message_transcript"]:
        sender_badge = f"[{msg.sender.upper()}]"
        receiver_badge = f"[{msg.receiver.upper()}]"
        print(f"--> {sender_badge:<20} -> {receiver_badge:<20} | {msg.performative:<25}")
        for k, v in msg.content.items():
            if k == "sensor_sequence":
                continue
            if isinstance(v, list) and len(v) > 5:
                print(f"      - {k}: {v[:5]} ... (total {len(v)} items)")
            else:
                print(f"      - {k}: {v}")
        print()

    print("=" * 80)
    print("STEP 2: SECTION 13 AUDIT LOG & FINAL RECOMMENDATION")
    print("=" * 80)
    print(decision["audit_trail_text"])
    print("=" * 80)

if __name__ == "__main__":
    main()
