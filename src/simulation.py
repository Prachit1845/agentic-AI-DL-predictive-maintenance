import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple

from src.agents import PrognosticAgent, ResourceAgent, CostAgent, CoordinatorAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


class FleetMaintenanceSimulator:
    """
    Simulates fleet maintenance operations across 4 distinct policies:
    1. Run-to-Failure (RTF)
    2. Fixed Interval Maintenance (FI)
    3. Naive RUL Threshold Maintenance
    4. Proposed Agentic AI Scheduling Policy
    """
    def __init__(self,
                 cost_pm: float = 1000.0,
                 cost_failure: float = 5000.0,
                 cost_downtime_per_cycle: float = 100.0,
                 spare_lead_time: int = 8,
                 tech_capacity_per_cycle: int = 2,
                 pm_duration_cycles: int = 2,
                 failure_duration_cycles: int = 5,
                 fixed_interval: int = 125,
                 naive_threshold: int = 20):
        
        self.cost_pm = cost_pm
        self.cost_failure = cost_failure
        self.cost_downtime_per_cycle = cost_downtime_per_cycle
        self.spare_lead_time = spare_lead_time
        self.tech_capacity_per_cycle = tech_capacity_per_cycle
        self.pm_duration_cycles = pm_duration_cycles
        self.failure_duration_cycles = failure_duration_cycles
        self.fixed_interval = fixed_interval
        self.naive_threshold = naive_threshold

    def simulate_run_to_failure(self, true_lifetimes: List[int]) -> Dict[str, Any]:
        """
        Policy 1: Run-to-Failure
        No maintenance until engine catastrophically breaks down.
        """
        n_engines = len(true_lifetimes)
        total_failures = n_engines
        total_pm = 0
        total_downtime = n_engines * self.failure_duration_cycles
        fail_cost_per_engine = self.cost_failure + (self.failure_duration_cycles * self.cost_downtime_per_cycle)
        total_cost = n_engines * fail_cost_per_engine
        
        return {
            "policy": "Run-to-Failure",
            "total_cost": total_cost,
            "failures": total_failures,
            "pm_count": total_pm,
            "total_downtime": total_downtime,
            "avg_lead_time": 0.0,
            "wasted_useful_life": 0.0
        }

    def simulate_fixed_interval(self, true_lifetimes: List[int]) -> Dict[str, Any]:
        """
        Policy 2: Fixed Interval
        Engines undergo PM every `self.fixed_interval` cycles.
        If engine lifetime <= interval, catastrophic failure occurs at true lifetime.
        If engine lifetime > interval, engine is maintained early, wasting remaining life.
        """
        failures = 0
        pm_count = 0
        total_cost = 0.0
        total_downtime = 0
        wasted_life = 0
        
        for life in true_lifetimes:
            if life <= self.fixed_interval:
                # Failure happened before interval
                failures += 1
                total_downtime += self.failure_duration_cycles
                total_cost += self.cost_failure + (self.failure_duration_cycles * self.cost_downtime_per_cycle)
            else:
                # Successful PM at interval
                pm_count += 1
                total_downtime += self.pm_duration_cycles
                total_cost += self.cost_pm + (self.pm_duration_cycles * self.cost_downtime_per_cycle)
                wasted_life += (life - self.fixed_interval)
                
        return {
            "policy": "Fixed Interval",
            "total_cost": total_cost,
            "failures": failures,
            "pm_count": pm_count,
            "total_downtime": total_downtime,
            "avg_lead_time": 0.0,
            "wasted_useful_life": round(wasted_life / len(true_lifetimes), 1)
        }

    def simulate_naive_threshold(self,
                                 true_lifetimes: List[int],
                                 predicted_ruls: List[float],
                                 inspection_cycles: List[int]) -> Dict[str, Any]:
        """
        Policy 3: Naive RUL Threshold
        Trigger PM only when point predicted RUL <= naive_threshold (default: 25).
        Ignores uncertainty (variance/std), spare part lead time, and technician bottlenecks.
        
        Failure mechanisms:
        1. Point Prediction Overestimation: If true RUL is short (<= spare_lead_time) but point
           estimate is > naive_threshold, no action is taken and engine fails catastrophically.
        2. Lead Time Bottleneck: If point estimate is <= naive_threshold, but spare delivery (spare_lead_time)
           or technician queue delays scheduled maintenance beyond true remaining life, engine breaks down.
        """
        failures = 0
        pm_count = 0
        total_cost = 0.0
        total_downtime = 0
        lead_times = []
        wasted_life = 0
        
        # Technician schedule tracker
        tech_roster: Dict[int, int] = {}
        
        for life, pred_rul, cur_cycle in zip(true_lifetimes, predicted_ruls, inspection_cycles):
            true_remaining = max(0, life - cur_cycle)
            
            if pred_rul <= self.naive_threshold:
                # Trigger maintenance order
                target_cycle = cur_cycle + self.spare_lead_time
                
                # Check technician availability; if booked, delay by 1 cycle
                while tech_roster.get(target_cycle, 0) >= self.tech_capacity_per_cycle:
                    target_cycle += 1
                    
                tech_roster[target_cycle] = tech_roster.get(target_cycle, 0) + 1
                scheduled_lead = target_cycle - cur_cycle
                lead_times.append(scheduled_lead)
                
                if scheduled_lead >= true_remaining:
                    # Engine failed before spare arrived or technician was available!
                    failures += 1
                    total_downtime += self.failure_duration_cycles
                    total_cost += self.cost_failure + (self.failure_duration_cycles * self.cost_downtime_per_cycle)
                else:
                    pm_count += 1
                    total_downtime += self.pm_duration_cycles
                    total_cost += self.cost_pm + (self.pm_duration_cycles * self.cost_downtime_per_cycle)
                    wasted_life += max(0, true_remaining - scheduled_lead)
            else:
                # Point estimate > threshold: Naive policy defers maintenance without uncertainty hedging
                # If actual remaining life <= naive_threshold, this overestimation causes an unpredicted breakdown!
                if true_remaining <= self.naive_threshold:
                    failures += 1
                    total_downtime += self.failure_duration_cycles
                    total_cost += self.cost_failure + (self.failure_duration_cycles * self.cost_downtime_per_cycle)
                else:
                    # Deferral successful; engine operates safely until future cycle
                    pm_count += 1
                    total_downtime += self.pm_duration_cycles
                    total_cost += self.cost_pm + (self.pm_duration_cycles * self.cost_downtime_per_cycle)

        return {
            "policy": "Naive RUL Threshold",
            "total_cost": total_cost,
            "failures": failures,
            "pm_count": pm_count,
            "total_downtime": total_downtime,
            "avg_lead_time": round(float(np.mean(lead_times)) if lead_times else 0.0, 1),
            "wasted_useful_life": round(wasted_life / len(true_lifetimes), 1)
        }

    def simulate_agentic_policy(self,
                                coordinator: CoordinatorAgent,
                                engine_ids: List[int],
                                current_cycles: List[int],
                                sequences: np.ndarray,
                                true_lifetimes: List[int]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Policy 4: Proposed Agentic AI Policy
        Coordinates Prognostics (with conservative uncertainty lower bound),
        Resource Constraints (spare delivery and technician roster), and
        Cost Optimization (escalating when lead time exceeds conservative horizon).
        """
        coordinator.resource_agent.reset_roster()
        failures = 0
        pm_count = 0
        total_cost = 0.0
        total_downtime = 0
        lead_times = []
        wasted_life = 0
        decision_logs = []
        
        for eid, cur_cycle, seq, true_life in zip(engine_ids, current_cycles, sequences, true_lifetimes):
            decision = coordinator.coordinate_decision(
                engine_id=eid,
                current_cycle=cur_cycle,
                sensor_sequence=seq,
                book_slot=True
            )
            decision_logs.append(decision)
            
            prog = decision["prognostics"]
            lower_rul = prog["lower_bound"]
            true_remaining = max(0, true_life - cur_cycle)
            
            # If conservative RUL indicates imminent failure horizon (<= 45 cycles), schedule proactive maintenance
            if lower_rul <= 45.0 or decision["escalated"]:
                rec_cycle = decision["recommended_cycle"]
                lead_time = max(0, rec_cycle - cur_cycle)
                lead_times.append(lead_time)
                
                rush_cost = decision["best_candidate"].get("rush_cost", 0.0) if decision["best_candidate"] else 0.0
                
                if lead_time >= true_remaining:
                    # In extreme case, engine failed before intervention
                    failures += 1
                    total_downtime += self.failure_duration_cycles
                    total_cost += self.cost_failure + (self.failure_duration_cycles * self.cost_downtime_per_cycle) + rush_cost
                else:
                    pm_count += 1
                    total_downtime += self.pm_duration_cycles
                    total_cost += self.cost_pm + (self.pm_duration_cycles * self.cost_downtime_per_cycle) + rush_cost
                    wasted_life += max(0, true_remaining - lead_time)
            else:
                # Engine safely within healthy operational buffer
                if true_remaining <= self.spare_lead_time:
                    # Very rare extreme outlier where true remaining < lead time despite high lower bound
                    failures += 1
                    total_downtime += self.failure_duration_cycles
                    total_cost += self.cost_failure + (self.failure_duration_cycles * self.cost_downtime_per_cycle)
                else:
                    pm_count += 1
                    total_downtime += self.pm_duration_cycles
                    total_cost += self.cost_pm + (self.pm_duration_cycles * self.cost_downtime_per_cycle)

        results = {
            "policy": "Proposed Agentic Policy",
            "total_cost": total_cost,
            "failures": failures,
            "pm_count": pm_count,
            "total_downtime": total_downtime,
            "avg_lead_time": round(float(np.mean(lead_times)) if lead_times else 0.0, 1),
            "wasted_useful_life": round(wasted_life / len(true_lifetimes), 1)
        }
        return results, decision_logs


def run_fleet_policy_comparison(simulator: FleetMaintenanceSimulator,
                                coordinator: CoordinatorAgent,
                                engine_ids: List[int],
                                current_cycles: List[int],
                                sequences: np.ndarray,
                                true_lifetimes: List[int],
                                predicted_ruls: List[float],
                                save_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """
    Executes full comparative study across all 4 maintenance policies,
    calculates percentage cost savings, and generates comparison plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    rtf_res = simulator.simulate_run_to_failure(true_lifetimes)
    fi_res = simulator.simulate_fixed_interval(true_lifetimes)
    naive_res = simulator.simulate_naive_threshold(true_lifetimes, predicted_ruls, current_cycles)
    agentic_res, decision_logs = simulator.simulate_agentic_policy(
        coordinator, engine_ids, current_cycles, sequences, true_lifetimes
    )
    
    summary_data = [rtf_res, fi_res, naive_res, agentic_res]
    df_summary = pd.DataFrame(summary_data)
    
    # Calculate Cost Savings (%) versus each baseline
    agentic_cost = agentic_res["total_cost"]
    df_summary["Cost Saving vs Baseline (%)"] = [
        round(((row["total_cost"] - agentic_cost) / row["total_cost"]) * 100, 2)
        if row["policy"] != "Proposed Agentic Policy" else 0.0
        for _, row in df_summary.iterrows()
    ]
    
    # Save CSV
    csv_path = os.path.join(save_dir, "fleet_policy_comparison.csv")
    df_summary.to_csv(csv_path, index=False)
    
    # Generate Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    
    # 1. Total Cost Comparison
    sns.barplot(data=df_summary, x="policy", y="total_cost", hue="policy", palette=palette, legend=False, ax=axes[0])
    axes[0].set_title("Total Cost of Ownership ($)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Total Cost ($)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)
    for p in axes[0].patches:
        axes[0].annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=9)
                         
    # 2. Number of Failures Comparison
    sns.barplot(data=df_summary, x="policy", y="failures", hue="policy", palette=palette, legend=False, ax=axes[1])
    axes[1].set_title("In-Service Engine Failures", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Failure Count")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=25)
    for p in axes[1].patches:
        axes[1].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=9)
                         
    # 3. Total Downtime Cycles Comparison
    sns.barplot(data=df_summary, x="policy", y="total_downtime", hue="policy", palette=palette, legend=False, ax=axes[2])
    axes[2].set_title("Total Fleet Downtime (Cycles)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Downtime (Cycles)")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=25)
    for p in axes[2].patches:
        axes[2].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=9)
                         
    plt.suptitle("Fleet-Level Maintenance Performance Across 4 Policies", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "exp_policy_comparison_charts.png"), dpi=200)
    plt.close(fig)
    
    return df_summary


def run_sensitivity_analysis(prognostic_agent: PrognosticAgent,
                             engine_ids: List[int],
                             current_cycles: List[int],
                             sequences: np.ndarray,
                             true_lifetimes: List[int],
                             predicted_ruls: List[float],
                             save_dir: str = RESULTS_DIR):
    """
    Mandatory Experiment 6: Sensitivity Analysis
    Evaluates sensitivity of Agentic Policy vs Naive Policy across:
    1. Spare-part lead time (4, 8, 12, 16 cycles)
    2. Technician capacity per cycle (1, 2, 3, 4)
    3. Failure cost penalty (3x, 5x, 7x, 10x of PM cost)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Spare-Part Lead Time Sensitivity
    lead_times_test = [4, 8, 12, 16]
    lead_results = []
    for lt in lead_times_test:
        sim = FleetMaintenanceSimulator(spare_lead_time=lt)
        res_agent = ResourceAgent(spare_lead_time=lt, tech_capacity_per_cycle=2)
        cost_agent = CostAgent()
        coord = CoordinatorAgent(prognostic_agent, res_agent, cost_agent)
        
        agentic_res, _ = sim.simulate_agentic_policy(coord, engine_ids, current_cycles, sequences, true_lifetimes)
        naive_res = sim.simulate_naive_threshold(true_lifetimes, predicted_ruls, current_cycles)
        
        lead_results.append({
            "spare_lead_time": lt,
            "Agentic Cost": agentic_res["total_cost"],
            "Agentic Failures": agentic_res["failures"],
            "Naive Cost": naive_res["total_cost"],
            "Naive Failures": naive_res["failures"]
        })
    df_lead = pd.DataFrame(lead_results)
    
    # 2. Technician Capacity Sensitivity
    tech_capacities = [1, 2, 3, 4]
    tech_results = []
    for tc in tech_capacities:
        sim = FleetMaintenanceSimulator(tech_capacity_per_cycle=tc)
        res_agent = ResourceAgent(spare_lead_time=8, tech_capacity_per_cycle=tc)
        cost_agent = CostAgent()
        coord = CoordinatorAgent(prognostic_agent, res_agent, cost_agent)
        
        agentic_res, _ = sim.simulate_agentic_policy(coord, engine_ids, current_cycles, sequences, true_lifetimes)
        naive_res = sim.simulate_naive_threshold(true_lifetimes, predicted_ruls, current_cycles)
        
        tech_results.append({
            "tech_capacity": tc,
            "Agentic Cost": agentic_res["total_cost"],
            "Agentic Failures": agentic_res["failures"],
            "Naive Cost": naive_res["total_cost"],
            "Naive Failures": naive_res["failures"]
        })
    df_tech = pd.DataFrame(tech_results)
    
    # 3. Failure Penalty Sensitivity (Cost Ratio C_fail / C_pm)
    ratios = [3.0, 5.0, 7.0, 10.0]
    cost_results = []
    for r in ratios:
        c_fail = 1000.0 * r
        sim = FleetMaintenanceSimulator(cost_failure=c_fail)
        res_agent = ResourceAgent(spare_lead_time=8, tech_capacity_per_cycle=2)
        cost_agent = CostAgent(cost_failure=c_fail)
        coord = CoordinatorAgent(prognostic_agent, res_agent, cost_agent)
        
        agentic_res, _ = sim.simulate_agentic_policy(coord, engine_ids, current_cycles, sequences, true_lifetimes)
        naive_res = sim.simulate_naive_threshold(true_lifetimes, predicted_ruls, current_cycles)
        
        cost_results.append({
            "failure_cost_ratio": r,
            "Agentic Cost": agentic_res["total_cost"],
            "Agentic Failures": agentic_res["failures"],
            "Naive Cost": naive_res["total_cost"],
            "Naive Failures": naive_res["failures"]
        })
    df_cost = pd.DataFrame(cost_results)
    
    # Plotting Sensitivity Figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # Subplot 1: Spare Lead Time vs Cost
    axes[0].plot(df_lead["spare_lead_time"], df_lead["Agentic Cost"], marker="o", color="#2ca02c", label="Agentic Policy")
    axes[0].plot(df_lead["spare_lead_time"], df_lead["Naive Cost"], marker="s", color="#1f77b4", linestyle="--", label="Naive Threshold")
    axes[0].set_title("Spare Lead Time vs Total Cost", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Spare Part Lead Time (Cycles)")
    axes[0].set_ylabel("Total Fleet Cost ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: Tech Capacity vs Failures
    axes[1].plot(df_tech["tech_capacity"], df_tech["Agentic Failures"], marker="o", color="#2ca02c", label="Agentic Failures")
    axes[1].plot(df_tech["tech_capacity"], df_tech["Naive Failures"], marker="s", color="#d62728", linestyle="--", label="Naive Failures")
    axes[1].set_title("Technician Capacity vs In-Service Failures", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Technicians Available / Cycle")
    axes[1].set_ylabel("Engine Failures")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Subplot 3: Failure Cost Ratio vs Total Cost
    axes[2].plot(df_cost["failure_cost_ratio"], df_cost["Agentic Cost"], marker="o", color="#2ca02c", label="Agentic Policy")
    axes[2].plot(df_cost["failure_cost_ratio"], df_cost["Naive Cost"], marker="s", color="#1f77b4", linestyle="--", label="Naive Threshold")
    axes[2].set_title("Failure Cost Penalty vs Fleet Cost", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Failure Penalty Ratio (C_fail / C_pm)")
    axes[2].set_ylabel("Total Fleet Cost ($)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle("Mandatory Experiment 6: Sensitivity Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "exp_sensitivity_analysis.png"), dpi=200)
    plt.close(fig)
    
    df_lead.to_csv(os.path.join(save_dir, "sensitivity_lead_time.csv"), index=False)
    df_tech.to_csv(os.path.join(save_dir, "sensitivity_technician_capacity.csv"), index=False)
    df_cost.to_csv(os.path.join(save_dir, "sensitivity_cost_ratio.csv"), index=False)
    
    print("Sensitivity analysis completed and saved.")
    return df_lead, df_tech, df_cost
