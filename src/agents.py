import math
import time
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.prognostics import predict_monte_carlo_dropout


@dataclass
class AgentMessage:
    """
    Standardized Inter-Agent Communication Message (FIPA-ACL Inspired).
    Fields:
        sender: Name of the initiating agent
        receiver: Target agent
        performative: Speech act (e.g. REQUEST, INFORM, EVALUATE, CONFIRM, ESCALATE)
        content: Data payload dictionary
        timestamp: Time of message dispatch
    """
    sender: str
    receiver: str
    performative: str
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3])

    def to_formatted_string(self) -> str:
        payload_preview = ", ".join([f"{k}: {v}" for k, v in list(self.content.items())[:4]])
        if len(self.content) > 4:
            payload_preview += ", ..."
        return f"[{self.timestamp}] [{self.sender} -> {self.receiver}] ({self.performative}): {payload_preview}"


class MessageBus:
    """
    Central event and message tracing bus for the Multi-Agent System.
    Logs and preserves the full chronological transcript of inter-agent negotiations.
    """
    def __init__(self):
        self.message_history: List[AgentMessage] = []

    def dispatch(self, message: AgentMessage) -> None:
        self.message_history.append(message)

    def get_transcript(self) -> List[AgentMessage]:
        return list(self.message_history)

    def clear(self) -> None:
        self.message_history.clear()

    def print_transcript(self) -> None:
        print("=" * 75)
        print("AGENT INTERACTION LOG & COMMUNICATION TRANSCRIPT")
        print("=" * 75)
        for msg in self.message_history:
            print(msg.to_formatted_string())
        print("=" * 75)


class PrognosticAgent:
    """
    Agent 1 — Prognostic Agent
    Responsibilities:
    - Loads the trained PyTorch LSTM network
    - Consumes recent multi-variate sensor trajectory
    - Executes Monte Carlo Dropout (MCD) to estimate RUL distribution
    - Responds to REQUEST_RUL messages with INFORM_RUL containing point prediction,
      epistemic standard deviation, and conservative 95% uncertainty interval.
    """
    def __init__(self, model, n_mc_samples: int = 50, message_bus: Optional[MessageBus] = None):
        self.name = "PrognosticAgent"
        self.model = model
        self.n_mc_samples = n_mc_samples
        self.message_bus = message_bus

    def process_request(self, message: AgentMessage) -> AgentMessage:
        """Processes REQUEST_RUL message and returns INFORM_RUL response."""
        sensor_sequence = message.content.get("sensor_sequence")
        engine_id = message.content.get("engine_id")
        
        rul_info = self.predict_rul(sensor_sequence)
        
        response = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            performative="INFORM_RUL",
            content={
                "engine_id": engine_id,
                "mean_rul": rul_info["mean_rul"],
                "std_rul": rul_info["std_rul"],
                "lower_bound": rul_info["lower_bound"],
                "upper_bound": rul_info["upper_bound"]
            }
        )
        if self.message_bus:
            self.message_bus.dispatch(response)
        return response

    def predict_rul(self, sensor_sequence: np.ndarray) -> Dict[str, float]:
        """
        Infers RUL distribution for an engine given its historical sensor window.
        Returns:
            mean_rul: Expected remaining useful life (cycles)
            std_rul: Epistemic & aleatoric uncertainty
            lower_bound: Conservative 95% bound (max(0, mean - 1.96*std))
            upper_bound: Optimistic 95% bound (mean + 1.96*std)
        """
        mcd_results = predict_monte_carlo_dropout(self.model, sensor_sequence, n_samples=self.n_mc_samples)
        return {
            "mean_rul": round(mcd_results["mean"], 1),
            "std_rul": round(mcd_results["std"], 2),
            "lower_bound": round(mcd_results["lower_bound"], 1),
            "upper_bound": round(mcd_results["upper_bound"], 1)
        }


class ResourceAgent:
    """
    Agent 2 — Resource Agent
    Responsibilities:
    - Tracks spare-part inventory and procurement lead times (L_spare)
    - Tracks technician availability roster across operating cycles (K_tech)
    - Responds to REQUEST_FEASIBLE_SLOTS with INFORM_FEASIBLE_SLOTS
    - Commits technician bookings upon CONFIRM_BOOKING messages.
    """
    def __init__(self, spare_lead_time: int = 8, tech_capacity_per_cycle: int = 2,
                 message_bus: Optional[MessageBus] = None):
        self.name = "ResourceAgent"
        self.spare_lead_time = spare_lead_time
        self.tech_capacity_per_cycle = tech_capacity_per_cycle
        self.roster: Dict[int, int] = {}
        self.message_bus = message_bus

    def process_request(self, message: AgentMessage) -> AgentMessage:
        if message.performative == "REQUEST_FEASIBLE_SLOTS":
            current_cycle = message.content["current_cycle"]
            horizon_cycles = message.content["horizon_cycles"]
            feasible = self.get_feasible_cycles(current_cycle, horizon_cycles)
            
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                performative="INFORM_FEASIBLE_SLOTS",
                content={
                    "spare_lead_time": self.spare_lead_time,
                    "earliest_spare_cycle": current_cycle + self.spare_lead_time,
                    "feasible_cycles": feasible,
                    "count": len(feasible)
                }
            )
            if self.message_bus:
                self.message_bus.dispatch(response)
            return response
            
        elif message.performative == "CONFIRM_BOOKING":
            cycle = message.content["target_cycle"]
            success = self.book_technician(cycle)
            
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                performative="ACK_BOOKING",
                content={
                    "target_cycle": cycle,
                    "success": success,
                    "booked_slots": self.roster.get(cycle, 0)
                }
            )
            if self.message_bus:
                self.message_bus.dispatch(response)
            return response
            
        raise ValueError(f"Unknown performative: {message.performative}")

    def is_technician_available(self, cycle: int) -> bool:
        booked = self.roster.get(cycle, 0)
        return booked < self.tech_capacity_per_cycle

    def book_technician(self, cycle: int) -> bool:
        if self.is_technician_available(cycle):
            self.roster[cycle] = self.roster.get(cycle, 0) + 1
            return True
        return False

    def get_feasible_cycles(self, current_cycle: int, horizon_cycles: int) -> List[int]:
        """
        Feasibility constraint logic:
        1. Cycle must satisfy spare part delivery: cycle >= current_cycle + spare_lead_time
        2. Cycle must have at least one free technician: booked < tech_capacity_per_cycle
        """
        earliest_cycle = current_cycle + self.spare_lead_time
        latest_cycle = current_cycle + horizon_cycles
        
        feasible_cycles = []
        for c in range(earliest_cycle, latest_cycle + 1):
            if self.is_technician_available(c):
                feasible_cycles.append(c)
        return feasible_cycles

    def reset_roster(self):
        self.roster.clear()


class CostAgent:
    """
    Agent 3 — Cost Agent
    Responsibilities:
    - Implements Total Cost of Ownership (TCO) evaluation:
        Total Cost = Preventive Cost + Corrective Failure Cost + Production Loss + Spare Logistics
    - Computes failure probability using prognostic Gaussian uncertainty distribution
    - Responds to REQUEST_COST_EVALUATION with INFORM_COST_EVALUATION ranking candidate cycles.
    """
    def __init__(self,
                 cost_pm: float = 1000.0,
                 cost_failure: float = 5000.0,
                 cost_downtime_per_cycle: float = 100.0,
                 pm_duration_cycles: int = 2,
                 failure_duration_cycles: int = 5,
                 expedited_spare_penalty: float = 800.0,
                 message_bus: Optional[MessageBus] = None):
        self.name = "CostAgent"
        self.cost_pm = cost_pm
        self.cost_failure = cost_failure
        self.cost_downtime_per_cycle = cost_downtime_per_cycle
        self.pm_duration_cycles = pm_duration_cycles
        self.failure_duration_cycles = failure_duration_cycles
        self.expedited_spare_penalty = expedited_spare_penalty
        self.message_bus = message_bus

    def process_request(self, message: AgentMessage) -> AgentMessage:
        current_cycle = message.content["current_cycle"]
        candidate_cycles = message.content["candidate_cycles"]
        mean_rul = message.content["mean_rul"]
        std_rul = message.content["std_rul"]
        is_expedited = message.content.get("is_expedited", False)
        
        evaluations = []
        for cycle in candidate_cycles:
            cost_info = self.calculate_candidate_cost(
                current_cycle=current_cycle,
                candidate_cycle=cycle,
                mean_rul=mean_rul,
                std_rul=std_rul,
                is_expedited=is_expedited
            )
            evaluations.append(cost_info)
            
        evaluations.sort(key=lambda x: x["expected_cost"])
        
        response = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            performative="INFORM_COST_EVALUATION",
            content={
                "candidate_evaluations": evaluations,
                "lowest_cost": evaluations[0]["expected_cost"] if evaluations else None,
                "best_cycle": evaluations[0]["candidate_cycle"] if evaluations else None
            }
        )
        if self.message_bus:
            self.message_bus.dispatch(response)
        return response

    def calculate_candidate_cost(self,
                                 current_cycle: int,
                                 candidate_cycle: int,
                                 mean_rul: float,
                                 std_rul: float,
                                 is_expedited: bool = False) -> Dict[str, Any]:
        """
        Computes expected cost for scheduling maintenance at `candidate_cycle`.
        Cycles until maintenance: delta_t = candidate_cycle - current_cycle.
        Failure probability P(failure before delta_t) using normal CDF.
        """
        delta_t = max(0, candidate_cycle - current_cycle)
        
        std = max(1e-2, std_rul)
        z = (delta_t - mean_rul) / std
        p_failure = float(stats.norm.cdf(z))
        p_success = 1.0 - p_failure
        
        pm_downtime_cost = self.pm_duration_cycles * self.cost_downtime_per_cycle
        total_pm_cost = self.cost_pm + pm_downtime_cost
        
        fail_downtime_cost = self.failure_duration_cycles * self.cost_downtime_per_cycle
        total_fail_cost = self.cost_failure + fail_downtime_cost
        
        rush_cost = self.expedited_spare_penalty if is_expedited else 0.0
        expected_cost = (p_success * total_pm_cost) + (p_failure * total_fail_cost) + rush_cost
        
        return {
            "candidate_cycle": candidate_cycle,
            "delta_t": delta_t,
            "p_failure": round(p_failure, 4),
            "expected_cost": round(expected_cost, 2),
            "total_pm_cost": total_pm_cost,
            "total_fail_cost": total_fail_cost,
            "rush_cost": rush_cost
        }


class CoordinatorAgent:
    """
    Agent 4 — Scheduling / Coordinator Agent
    Responsibilities:
    - Coordinates Prognostic, Resource, and Cost Agents via explicit message passing
    - Executes transparent candidate-window search (Section 14):
        1. Obtain RUL mean and conservative lower bound from Prognostic Agent
        2. Filter cycles where spare parts can arrive and technicians are available
        3. Evaluate expected cost for all candidate cycles within conservative bound
        4. Select lowest-cost feasible cycle
        5. If no cycle is feasible before failure, escalate to emergency intervention
    - Generates structured, human-readable audit trail explaining rationale (Section 13)
    - Records the complete agent dialogue transcript.
    """
    def __init__(self,
                 prognostic_agent: PrognosticAgent,
                 resource_agent: ResourceAgent,
                 cost_agent: CostAgent,
                 message_bus: Optional[MessageBus] = None):
        self.name = "CoordinatorAgent"
        self.prognostic_agent = prognostic_agent
        self.resource_agent = resource_agent
        self.cost_agent = cost_agent
        self.message_bus = message_bus if message_bus is not None else MessageBus()
        
        # Link shared message bus
        self.prognostic_agent.message_bus = self.message_bus
        self.resource_agent.message_bus = self.message_bus
        self.cost_agent.message_bus = self.message_bus

    def coordinate_decision(self,
                            engine_id: Any,
                            current_cycle: int,
                            sensor_sequence: np.ndarray,
                            book_slot: bool = True) -> Dict[str, Any]:
        """
        Orchestrates multi-agent decision flow for an engine.
        Returns full recommendation report with candidate breakdown, audit reasoning, and message transcript.
        """
        self.message_bus.clear()

        # Step 1: Dispatch REQUEST_RUL to Prognostic Agent
        msg_rul_req = AgentMessage(
            sender=self.name,
            receiver=self.prognostic_agent.name,
            performative="REQUEST_RUL",
            content={"engine_id": engine_id, "current_cycle": current_cycle, "sensor_sequence": sensor_sequence}
        )
        self.message_bus.dispatch(msg_rul_req)
        msg_rul_resp = self.prognostic_agent.process_request(msg_rul_req)
        
        prog_info = msg_rul_resp.content
        mean_rul = prog_info["mean_rul"]
        std_rul = prog_info["std_rul"]
        lower_rul = prog_info["lower_bound"]
        upper_rul = prog_info["upper_bound"]
        
        safe_horizon = max(self.resource_agent.spare_lead_time, int(lower_rul))
        
        # Step 2: Dispatch REQUEST_FEASIBLE_SLOTS to Resource Agent
        msg_res_req = AgentMessage(
            sender=self.name,
            receiver=self.resource_agent.name,
            performative="REQUEST_FEASIBLE_SLOTS",
            content={"current_cycle": current_cycle, "horizon_cycles": safe_horizon}
        )
        self.message_bus.dispatch(msg_res_req)
        msg_res_resp = self.resource_agent.process_request(msg_res_req)
        
        feasible_cycles = msg_res_resp.content["feasible_cycles"]
        candidates_evaluated = []
        best_candidate = None
        escalated = False
        decision_reason = ""
        
        if len(feasible_cycles) > 0:
            # Step 3 & 4: Dispatch REQUEST_COST_EVALUATION to Cost Agent
            msg_cost_req = AgentMessage(
                sender=self.name,
                receiver=self.cost_agent.name,
                performative="REQUEST_COST_EVALUATION",
                content={
                    "current_cycle": current_cycle,
                    "candidate_cycles": feasible_cycles,
                    "mean_rul": mean_rul,
                    "std_rul": std_rul,
                    "is_expedited": False
                }
            )
            self.message_bus.dispatch(msg_cost_req)
            msg_cost_resp = self.cost_agent.process_request(msg_cost_req)
            
            candidates_evaluated = msg_cost_resp.content["candidate_evaluations"]
            best_candidate = candidates_evaluated[0]
            recommended_cycle = best_candidate["candidate_cycle"]
            
            decision_reason = (
                f"Lowest feasible expected total cost ({best_candidate['expected_cost']} units) "
                f"while remaining safely within conservative RUL window [{lower_rul}, {upper_rul}] cycles."
            )
            
            # Step 5: Dispatch CONFIRM_BOOKING to Resource Agent
            if book_slot:
                msg_book = AgentMessage(
                    sender=self.name,
                    receiver=self.resource_agent.name,
                    performative="CONFIRM_BOOKING",
                    content={"engine_id": engine_id, "target_cycle": recommended_cycle}
                )
                self.message_bus.dispatch(msg_book)
                self.resource_agent.process_request(msg_book)
                
        else:
            # Step 6: Escalation logic — no standard slot feasible before conservative failure limit
            escalated = True
            earliest_possible = current_cycle + max(1, self.resource_agent.spare_lead_time // 2)
            
            msg_escalate = AgentMessage(
                sender=self.name,
                receiver="ALL_AGENTS",
                performative="TRIGGER_EMERGENCY_ESCALATION",
                content={
                    "engine_id": engine_id,
                    "reason": f"No feasible slot before conservative RUL ({lower_rul} cycles)",
                    "emergency_cycle": earliest_possible
                }
            )
            self.message_bus.dispatch(msg_escalate)
            
            # Cost evaluation with expedited spare penalty
            msg_cost_req = AgentMessage(
                sender=self.name,
                receiver=self.cost_agent.name,
                performative="REQUEST_COST_EVALUATION",
                content={
                    "current_cycle": current_cycle,
                    "candidate_cycles": [earliest_possible],
                    "mean_rul": mean_rul,
                    "std_rul": std_rul,
                    "is_expedited": True
                }
            )
            self.message_bus.dispatch(msg_cost_req)
            msg_cost_resp = self.cost_agent.process_request(msg_cost_req)
            
            candidates_evaluated = msg_cost_resp.content["candidate_evaluations"]
            best_candidate = candidates_evaluated[0]
            recommended_cycle = earliest_possible
            
            decision_reason = (
                f"ESCALATION ALERT: No feasible regular maintenance slot available before conservative limit "
                f"({lower_rul} cycles). Triggered emergency spare expedition to schedule earliest intervention "
                f"at cycle {recommended_cycle}."
            )
            
            if book_slot:
                msg_book = AgentMessage(
                    sender=self.name,
                    receiver=self.resource_agent.name,
                    performative="CONFIRM_BOOKING",
                    content={"engine_id": engine_id, "target_cycle": recommended_cycle, "is_emergency": True}
                )
                self.message_bus.dispatch(msg_book)
                self.resource_agent.process_request(msg_book)

        # Step 7: Build structured audit trail report matching Section 13
        formatted_audit_log = self._format_audit_log(
            engine_id=engine_id,
            current_cycle=current_cycle,
            prog_info=prog_info,
            spare_lead_time=self.resource_agent.spare_lead_time,
            feasible_cycles=feasible_cycles,
            candidates=candidates_evaluated,
            recommended_cycle=recommended_cycle,
            decision_reason=decision_reason,
            escalated=escalated
        )

        return {
            "engine_id": engine_id,
            "current_cycle": current_cycle,
            "prognostics": prog_info,
            "spare_lead_time": self.resource_agent.spare_lead_time,
            "feasible_cycles": feasible_cycles,
            "candidates_evaluated": candidates_evaluated,
            "recommended_cycle": recommended_cycle,
            "best_candidate": best_candidate,
            "escalated": escalated,
            "reason": decision_reason,
            "audit_trail_text": formatted_audit_log,
            "message_transcript": self.message_bus.get_transcript()
        }

    def _format_audit_log(self, engine_id, current_cycle, prog_info, spare_lead_time,
                          feasible_cycles, candidates, recommended_cycle, decision_reason, escalated):
        """Formats report exactly matching Section 13 of the project rubric."""
        lines = []
        lines.append(f"Engine: E-{str(engine_id).zfill(3)}")
        lines.append(f"Current Operating Cycle: {current_cycle}")
        lines.append(f"Predicted RUL: {prog_info['mean_rul']} cycles (std: {prog_info['std_rul']})")
        lines.append(f"Uncertainty interval: {prog_info['lower_bound']} - {prog_info['upper_bound']} cycles")
        lines.append(f"Spare-part lead time: {spare_lead_time} cycles")
        
        tech_str = ", ".join([f"Cycle {c}" for c in feasible_cycles[:6]]) if feasible_cycles else "None available within safe horizon"
        lines.append(f"Technician available: {tech_str}")
        lines.append("")
        
        for idx, cand in enumerate(candidates[:4]):
            cand_letter = chr(65 + idx)
            risk_desc = "minimal" if cand['p_failure'] < 0.05 else ("moderate" if cand['p_failure'] < 0.15 else "higher")
            lines.append(f"Candidate {cand_letter}: Maintain at Cycle {cand['candidate_cycle']}")
            lines.append(f"   Expected cost: {cand['expected_cost']} units (P(fail): {cand['p_failure']:.1%}, Risk penalty: {risk_desc})")
            
        lines.append("")
        if escalated:
            lines.append(f"Agent Recommendation: [ESCALATION] Expedite emergency maintenance at Cycle {recommended_cycle}.")
        else:
            lines.append(f"Agent Recommendation: Schedule preventive maintenance at Cycle {recommended_cycle}, provided spare ordering is initiated immediately.")
        lines.append(f"Reason: {decision_reason}")
        return "\n".join(lines)
