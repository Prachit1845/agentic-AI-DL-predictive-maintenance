# PS-05 Comprehensive Viva Preparation & Defense Guide

**Project Title:** Agentic Predictive Maintenance Scheduling under Remaining-Useful-Life (RUL) Uncertainty for Turbofan Engine Fleets  
**Target Level:** TY B.Tech Capstone | 150 Marks | 4 Students

---

## 1. System Architecture & High-Level Flow
```
┌───────────────────────────────────────────────────────────────┐
│              1. NASA C-MAPSS FD001 Telemetry                 │
│      100 Train Engines (to failure) + 100 Test Engines        │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│               2. Preprocessing & Feature Selection            │
│   - Drop 7 constant sensors (s1, s5, s10, s16, s18, s19, set3)│
│   - Min-Max Scaling (fitted strictly on training set)         │
│   - Piecewise Capped RUL Target (Cap = 125 cycles)            │
│   - Sliding Window Sequences (Window W = 30 cycles)           │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│               3. Deep Learning Prognostic Model               │
│   - Compact PyTorch LSTM (64 units) + Dropout (p=0.2) + Dense │
│   - Evaluated on RMSE, MAE, and NASA Asymmetric Scoring Func  │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│               4. Monte Carlo Dropout Uncertainty              │
│   - 50 Stochastic Inference Passes with active dropout        │
│   - Mean (μ), Std Dev (σ), Conservative Lower Bound (RUL_lower)│
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│               5. Multi-Agent Scheduling Layer                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐  │
│  │ Prognostic Agent │  │  Resource Agent  │  │ Cost Agent  │  │
│  └─────────┬────────┘  └─────────┬────────┘  └──────┬──────┘  │
│            │                     │                  │         │
│            ▼                     ▼                  ▼         │
│               Coordinator / Scheduling Agent                  │
│       Candidate-window optimization & Section 13 Audit Log    │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│               6. Fleet Maintenance Simulator                  │
│   - Compare 4 Policies: Run-to-Failure, Fixed-Interval,       │
│     Naive RUL Threshold, and Proposed Agentic Policy          │
│   - Total Cost of Ownership (TCO), Failures, Downtime, % Sav  │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│        7. Interactive Dashboard & Sensitivity Sandbox         │
│   - Streamlit Application (app.py) + Plotly Telemetry         │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. Four-Student Work Allocation & Individual Responsibilities

### Student 1: Data Preparation, EDA & Target Engineering
* **Core Responsibilities:**
  * Downloading, parsing, and cleaning NASA C-MAPSS FD001.
  * Conducting exploratory data analysis and producing the 6 mandatory EDA plots.
  * Identifying constant/zero-variance instrumentation and performing feature selection (14 informative sensors).
  * Enforcing proper normalization (MinMaxScaler fitted strictly on training data).
  * Formulating and justifying the piecewise linear capped RUL target ($RUL_{max} = 125$).
* **Primary Defense Topics:** Data cleaning rationale, leakage prevention, sensor correlations, physical meaning of turbofan degradation, justification of RUL capping.

### Student 2: Deep Learning LSTM Model & Uncertainty Quantification
* **Core Responsibilities:**
  * Implementing the PyTorch LSTM neural network architecture.
  * Setting up the training loop, Adam optimizer, loss function, and ReduceLROnPlateau scheduler.
  * Conducting hyperparameter tuning (Experiment 1: Sequence Length 30 vs. 20).
  * Evaluating model accuracy using MAE, RMSE, and the NASA Asymmetric Scoring Function.
  * Formulating and executing Monte Carlo Dropout (Gal & Ghahramani, 2016) across 50 stochastic passes to extract predictive distributions $[\text{RUL}_{lower}, \text{RUL}_{upper}]$.
* **Primary Defense Topics:** LSTM cell mechanics, mathematical formulation of dropout as a Bayesian approximation, NASA score asymmetric penalty math, epistemic vs. aleatoric uncertainty.

### Student 3: Fleet Maintenance Simulator & Baseline Policy Economics
* **Core Responsibilities:**
  * Defining the fleet simulation framework and economic cost structure ($C_{pm}, C_{fail}, C_{downtime}$).
  * Implementing Policy 1: Corrective Run-to-Failure (RTF).
  * Implementing Policy 2: Fixed-Interval Periodic Maintenance (FI).
  * Implementing Policy 3: Naive RUL Threshold Maintenance.
  * Tracking KPIs: Total Cost of Ownership, in-service failures, fleet downtime, wasted useful life, and percentage cost savings.
* **Primary Defense Topics:** Cost function formulation, why symmetric RMSE is misleading in operations, failure mechanisms of naive thresholding under spare lead times, asset utilization trade-offs.

### Student 4: Agentic AI Coordination Layer, Sensitivity & Dashboard
* **Core Responsibilities:**
  * Designing the 4-Agent architecture: Prognostic Agent, Resource Agent, Cost Agent, and Coordinator Agent.
  * Developing the transparent candidate-window search algorithm (Section 14).
  * Implementing the emergency escalation mechanism and generating Section 13 audit traces.
  * Conducting the Experiment 6 sensitivity analysis across spare lead times, technician capacity, and failure penalties.
  * Building the interactive Python Streamlit dashboard (`app.py`).
* **Primary Defense Topics:** Multi-agent interaction protocols, why rule-based candidate search is superior to black-box RL for two-month scope, auditability/explainability in aviation, sensitivity analysis conclusions.

---

## 3. High-Probability Viva Questions & Model Answers

### Question 1: Why did you choose C-MAPSS FD001 over FD002, FD003, or FD004?
**Answer:**  
C-MAPSS FD001 features a single sea-level operating condition and a single failure mode (High Pressure Compressor degradation). Subsets FD002 and FD004 introduce 6 complex operational flight conditions and multiple fault modes, which requires extensive operational condition clustering before prognostics can begin. For our project scope, the primary research contribution is positioned **above the prognostic model**—specifically in the multi-agent decision layer. FD001 provides a clean, validated benchmark that keeps prognostic evaluation manageable while allowing thorough focus on scheduling under uncertainty.

### Question 2: Why are CWRU and Paderborn bearing datasets excluded from this project?
**Answer:**  
The Case Western Reserve University (CWRU) and Paderborn bearing datasets are designed for **fault classification and condition monitoring** (identifying inner race, outer race, or ball defects at steady states). They do not contain continuous run-to-failure trajectories necessary to estimate Remaining Useful Life (RUL) as a countdown variable. Incorporating them would transform the problem into a discrete classifier rather than a continuous prognostic-to-scheduling pipeline.

### Question 3: Why cap the RUL target at 125 cycles? Why not use raw linear RUL?
**Answer:**  
In physical turbofans, components undergo negligible structural wear during early operating cycles; sensor readings in this regime reflect nominal baseline noise. If an uncapped linear target is enforced (e.g., $RUL = 300$ down to $0$), the neural network attempts to fit subtle random sensor fluctuations to arbitrary countdown targets, causing high variance and poor convergence. Grounded in foundational literature (**Heimes 2008**, **Zheng et al. 2017**), capping the target at $RUL_{max} = 125$ cycles lets the model learn that anything above 125 represents a healthy baseline, focusing gradient descent entirely on the progressive degradation phase.

### Question 4: How does an LSTM cell work, and why is it preferred over standard RNNs?
**Answer:**  
Standard Recurrent Neural Networks (RNNs) suffer from vanishing and exploding gradients when processing long sequences, preventing them from learning long-term temporal dependencies. The Long Short-Term Memory (LSTM) cell resolves this via an internal cell state ($C_t$) regulated by three gates:
1. **Forget Gate:** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ — decides what historical information to discard.
2. **Input Gate:** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ and $\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$ — determines what new sensor information to store in the cell state.
3. **Cell State Update:** $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$.
4. **Output Gate:** $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ and $h_t = o_t * \tanh(C_t)$ — yields the hidden representation passed to the next cycle.

### Question 5: What is the NASA Scoring Function, and why is it asymmetric?
**Answer:**  
Standard metrics like RMSE penalize errors symmetrically ($(\hat{y} - y)^2$). In aerospace operations, operational consequences are severely asymmetric:
* If the model predicts an RUL of 30 cycles when actual life is 40 cycles ($d = -10$, early prediction), the engine is inspected prematurely, causing a minor logistical inconvenience.
* If the model predicts an RUL of 40 cycles when actual life is 30 cycles ($d = +10$, late prediction), the engine remains in flight past failure, risking in-service disaster.
The NASA score penalizes late predictions exponentially:
$$s = \exp\left(-\frac{d}{13}\right) - 1 \quad (\text{for } d < 0), \qquad s = \exp\left(\frac{d}{10}\right) - 1 \quad (\text{for } d \ge 0)$$
A 10-cycle late error receives more than double the penalty of a 10-cycle early error.

### Question 6: What is the mathematical basis of Monte Carlo Dropout?
**Answer:**  
Gal & Ghahramani (2016) proved that a neural network with dropout applied before every weight layer is mathematically equivalent to an approximation of a Deep Gaussian Process. During standard inference, dropout is turned off, yielding a deterministic point prediction. In Monte Carlo Dropout, dropout is kept active during test time across $N$ stochastic forward passes. Each pass samples a random sub-network from the posterior distribution over weights, allowing us to compute the empirical mean $\mu$ (expected RUL) and standard deviation $\sigma$ (epistemic model uncertainty).

### Question 7: Why did the Naive RUL Threshold policy fail in your simulation?
**Answer:**  
The Naive Threshold policy triggers maintenance when the point prediction $\hat{\text{RUL}} \le 20$. It fails due to two fatal blindspots:
1. **Unhedged Prediction Error:** If an engine has a true remaining life of 8 cycles, but the model overestimates and predicts 22 cycles, the Naive policy takes zero action. By the time the point prediction drops below 20, the engine has already failed.
2. **Lead Time Blindness:** Spare parts require an 8-cycle lead time ($L_{spare}$). When an alert is triggered at cycle 15, the parts arrive at cycle 23. If the engine fails at cycle 20, the parts arrive too late.
The Agentic Policy avoids this by planning against the conservative lower bound $\text{RUL}_{lower} = \mu - 1.96\sigma$.

### Question 8: Explain the transparent candidate-window search algorithm.
**Answer:**  
Rather than employing an opaque, hard-to-verify reinforcement learning agent, we use a transparent candidate-window search algorithm (Section 14 of specification):
1. **Define Horizon:** Evaluates candidate cycles $t$ up to the conservative limit: $t \le t_{now} + \text{RUL}_{lower}$.
2. **Check Feasibility:** Filters out any cycle where $t < t_{now} + L_{spare}$ (spare cannot arrive in time) or where the technician roster has reached maximum capacity ($K_{tech}$).
3. **Cost Evaluation:** The Cost Agent calculates expected cost for all valid candidates using Gaussian failure probabilities.
4. **Optimization:** Selects the candidate cycle with the minimum expected cost.
5. **Escalation Protocol:** If no candidate cycle is feasible before the failure limit, the Coordinator Agent escalates the case: it orders expedited spare delivery (paying a rush fee) and schedules the earliest emergency maintenance.

### Question 9: Why did you not use Reinforcement Learning for scheduling?
**Answer:**  
As highlighted in Section 2 and Section 14 of the project brief, multi-agent reinforcement learning (MARL) requires millions of simulated fleet environment steps to converge, is prone to reward gaming, and functions as an uninterpretable black box. In aerospace and high-reliability industrial engineering, maintenance managers require **explainable, verifiable decisions** with auditable reasons. The transparent candidate-window search produces clear Section 13 audit logs that maintenance engineers can inspect and trust.

### Question 10: What are your key empirical findings from the 100-engine fleet simulation?
**Answer:**  
* **Versus Run-to-Failure:** Agentic policy reduced total cost by **75.05%** ($\$137,200$ vs. $\$550,000$) and eliminated 96 out of 100 catastrophic breakdowns.
* **Versus Fixed-Interval:** Fixed-Interval wasted an average of **80.4 operational cycles per engine** by stripping healthy components prematurely. Agentic policy extracts near-maximum life, wasting only 4.8 cycles of conservative buffer.
* **Versus Naive Threshold:** Agentic policy reduced total cost by **23.86%** ($\$137,200$ vs. $\$180,200$) and cut in-service failures from 14 to 4 by accounting for spare lead times and uncertainty.
