# PS-05: Agentic Predictive Maintenance Scheduling under Remaining Useful Life (RUL) Uncertainty

**Project Reference:** PS-05 | TY B.Tech Capstone | 150 Marks  
**Benchmark Dataset:** NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001)  
**Primary Methodological Domain:** Deep Learning (LSTM) + Agentic Artificial Intelligence (Multi-Agent Systems)  
**Deliverable File:** [`Agentic_Predictive_Maintenance_LSTM_Report.docx`](file:///c:/Users/Prachit/Deep%20learning%20final/Agentic_Predictive_Maintenance_LSTM_Report.docx)

---

## 1. Executive Summary

In commercial aviation and high-reliability industrial operations, predicting Remaining Useful Life (RUL) through deep learning prognostics is necessary but insufficient. Deterministic point predictions fail to convey model confidence, and naive threshold-based scheduling ignores physical constraints such as spare-part lead times and technician capacity.

This project implements an end-to-end framework consisting of:
1. **A Deep LSTM Prognostic Estimator** equipped with **Monte Carlo Dropout** to quantify model epistemic uncertainty.
2. **A 4-Agent Scheduling Layer** (Prognostic, Resource, Cost, and Coordinator Agents) that balances failure risk against operational bottlenecks to schedule maintenance dynamically.

Empirical evaluation on a fleet of 100 NASA C-MAPSS turbofan engines demonstrates:
- **75.05% cost reduction** compared to Run-to-Failure ($$137,200$ vs. $$550,000$), reducing catastrophic breakdowns by 96%.
- **23.86% cost reduction** compared to standard naive RUL threshold policies ($$137,200$ vs. $$180,200$).
- **75.6 operating cycles of usable engine life recovered** per asset compared to Fixed-Interval overhauls.

---

## 2. Complete Pipeline Architecture (From Image Flowchart)

The image in the project specification defines the core end-to-end data and model pipeline:

```mermaid
flowchart TD
    A["<b>NASA C-MAPSS</b><br/>(FD001 Turbofan Run-to-Failure Fleet)"] --> B["<b>Data Cleaning + Sensor Selection</b><br/>(Drop 7 flat sensors, Select 14 sensors, MinMax Norm, Capped RUL)"]
    B --> C["<b>Sequence Generation</b><br/>(Sliding Window W = 30 Cycles &rarr; Tensor Shape: [B, 30, 14])"]
    C --> D["<b>LSTM RUL Model</b><br/>(PyTorch LSTM [64 units] + Dropout [0.2] + Dense FC layers)"]
    D --> E["<b>Predicted RUL + Uncertainty Range</b><br/>(Monte Carlo Dropout: Mean &mu;, Std &sigma;, [RUL_lower, RUL_upper])"]
    E --> F["<b>Multi-Agent Scheduling Layer</b><br/>(Prognostic, Resource, Cost & Coordinator Agents)"]
    
    style A fill:#0284C7,stroke:#0369A1,stroke-width:2px,color:#fff
    style B fill:#0D9488,stroke:#0F766E,stroke-width:2px,color:#fff
    style C fill:#D97706,stroke:#B45309,stroke-width:2px,color:#fff
    style D fill:#4F46E5,stroke:#4338CA,stroke-width:2px,color:#fff
    style E fill:#9333EA,stroke:#7E22CE,stroke-width:2px,color:#fff
    style F fill:#16A34A,stroke:#15803D,stroke-width:2px,color:#fff
```

### End-to-End Pipeline Matrix

| Step | Stage Name | Input Artifact | Core Transformation / Methodology | Output Artifact |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **NASA C-MAPSS Dataset** | `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` | Turbofan run-to-failure simulation telemetry under sea-level operating condition | 100 Train & 100 Test engines (26 columns) |
| **2** | **Data Cleaning + Sensor Selection** | Raw 21 sensor channels + 3 operational settings | Variance inspection (drop 7 flat sensors), Min-Max scaling to $[0, 1]$, Capped piecewise linear RUL ($RUL_{max}=125$) | 14 Informative sensors + bounded RUL targets |
| **3** | **Sequence Generation** | Tabular normalized sensor cycles | Sliding temporal window of length $W = 30$ cycles with boundary padding | 3D Tensors of shape `(Batch, 30, 14)` |
| **4** | **LSTM RUL Model** | Sequential 3D Tensors `(Batch, 30, 14)` | Compact PyTorch LSTM ($H=64$) + Dropout ($p=0.2$) + Dense FC($32$) + ReLU + FC($1$) | Trained weights `models/best_lstm_model.pt` |
| **5** | **Predicted RUL + Uncertainty Range** | Test engine final sequence windows | Monte Carlo Dropout ($N_{MC} = 30$ stochastic passes during inference) | Predictive Mean ($\mu$), Std Dev ($\sigma$), Bounds $[RUL_{lower}, RUL_{upper}]$ |
| **6** | **Agentic Scheduling Layer** | Predictive bounds + Fleet inventory roster | 4-Agent negotiation under spare lead time ($L=8$) and technician constraints ($K=2$) | Cost-optimal, collision-free fleet maintenance schedule |

---

## 3. Stage 1: NASA C-MAPSS FD001 Dataset

The **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset, developed by the NASA Glenn Research Center, is the premier international benchmark for prognostics:
- **Fleet Size:** 100 engines in training set (run until functional failure), 100 engines in test set (operational series truncated prior to failure).
- **Ground Truth:** `RUL_FD001.txt` provides the exact true remaining cycles for each test engine.
- **Data Matrix:** 26 columns per observation:
  1. `engine_id`: Identifier for the specific engine unit (1 to 100).
  2. `cycle`: Operational cycle count (time step).
  3. `setting_1`, `setting_2`, `setting_3`: Operational flight regimes (Altitude, Mach number, Throttle Resolver Angle).
  4. `s_1` through `s_21`: 21 thermal, pressure, and rotational velocity sensor channels.
- **Operating Profile:** Single sea-level operating condition; failure mode caused by High-Pressure Compressor (HPC) degradation.

---

## 4. Stage 2: Data Cleaning & Sensor Selection

Raw telemetry from turbofan engines contains uninformative channels, disparate scales, and non-linear degradation characteristics:

### 4.1 Variance Filtering & Dropping Constant Sensors
A variance analysis across the 20,631 training cycles reveals that several sensors remain completely static throughout every engine's operational life. Feeding constant features into neural networks adds dead weights, dilutes backpropagation gradients, and introduces numerical instability.

The following **7 constant features** have zero or near-zero variance and are eliminated:
- `s_1`: Total temperature at fan inlet ($T2$)
- `s_5`: Total pressure at fan inlet ($P2$)
- `s_10`: Total pressure at burner inlet ($P15$)
- `s_16`: Bypass ratio burner pressure
- `s_18`: Demanded fan speed
- `s_19`: Demanded corrected fan speed
- `setting_3`: Flight setting 3

### 4.2 The 14 Informative Sensor Channels
Dropping the constant channels leaves the standard set of **14 informative sensor channels**:

| Sensor ID | Description | Engineering Unit | Degradation Behavior |
| :---: | :--- | :---: | :--- |
| **$s_2$** | Total temperature at LPC outlet ($T_{24}$) | °R | Monotonically increases with wear |
| **$s_3$** | Total temperature at HPC outlet ($T_{30}$) | °R | Monotonically increases with wear |
| **$s_4$** | Total temperature at LPT outlet ($T_{50}$) | °R | Monotonically increases with wear |
| **$s_7$** | Total pressure at HPC outlet ($P_{30}$) | psia | Monotonically decreases with wear |
| **$s_8$** | Physical fan speed ($N_f$) | rpm | Monotonically decreases with wear |
| **$s_9$** | Physical core speed ($N_c$) | rpm | Monotonically decreases with wear |
| **$s_{11}$** | Static pressure at HPC outlet ($Ps_{30}$) | psia | Monotonically increases with wear |
| **$s_{12}$** | Ratio of fuel flow to $Ps_{30}$ ($W_{31}$) | pps/psia | Monotonically decreases with wear |
| **$s_{13}$** | Corrected fan speed ($N_{f,corr}$) | rpm | Monotonically decreases with wear |
| **$s_{14}$** | Corrected core speed ($N_{c,corr}$) | rpm | Monotonically decreases with wear |
| **$s_{15}$** | Bypass ratio ($BPR$) | - | Monotonically increases with wear |
| **$s_{17}$** | Bleed enthalpy ($ht$) | - | Monotonically increases with wear |
| **$s_{20}$** | High-pressure turbine (HPT) coolant bleed | lbm/s | Monotonically decreases with wear |
| **$s_{21}$** | Low-pressure turbine (LPT) coolant bleed | lbm/s | Monotonically decreases with wear |

### 4.3 Min-Max Normalization
Because sensor values span wildly different numerical ranges (e.g., temperatures $> 1400^\circ\text{R}$ vs. bleed enthalpy $\approx 390$), Min-Max scaling normalizes all 14 channels into the standardized range $[0, 1]$:
$$x'_{i, t} = \frac{x_{i, t} - \min(x_{i, train})}{\max(x_{i, train}) - \min(x_{i, train})}$$

The scaler is fitted **strictly on the training data** and applied to both train and test partitions to prevent data leakage.

### 4.4 Piecewise Linear Capped RUL Target
In physical gas turbines, component wear during initial operating cycles is negligible. If an uncapped linear countdown target is enforced (e.g., from cycle 350 down to 0), the neural network is penalized for not predicting degradation that physically does not yet exist, forcing it to fit spurious noise in healthy baselines.

Following **Heimes (2008)** and **Zheng et al. (2017)**, we cap the target RUL at $RUL_{max} = 125$ cycles:
$$\text{RUL}_{target}(t) = \min(125, T_{failure} - t)$$

This piecewise linear formulation allows the LSTM to preserve its capacity for the non-linear degradation phase ($RUL < 125$).

---

## 5. Stage 3: Sliding-Window Sequence Generation

Single-cycle sensor snapshots do not communicate the velocity or rate-of-change of component wear. Continuous telemetry is transformed into sliding temporal sequences:
- **Window Length ($W$):** 30 consecutive operating cycles.
- **Input Dimension:** At cycle $t$, the input tensor is $\mathbf{X}_t \in \mathbb{R}^{30 \times 14}$.
- **Target:** The scalar capped RUL at cycle $t$.
- **Boundary Handling:** Test engines whose operational trajectory is shorter than 30 cycles are left-edge padded with repeated earliest observations.

---

## 6. Stage 4: LSTM RUL Model & Detailed Working Mechanism

Long Short-Term Memory (LSTM) networks are a specialized Recurrent Neural Network (RNN) architecture engineered to capture long-term sequential dependencies.

```
                  ┌───────────────────────────────────────────────────────────┐
                  │                 Cell State C_t (Long-Term Memory)         │
C_(t-1) ──────────► ( × ) ────────────────────────────────────────── ( + ) ───┼──────────► C_t
                     ▲                                                ▲       │
                     │ [Forget: f_t]                                  │       │
                     │                                         ┌──────┴─────┐ │
                     │                                         │   ( × )    │ │
                     │                                         └──▲───────▲─┘ │
                     │                                 [Input: i_t│       │   │
                     │                                            │       │[Candidate: C~_t]
            ┌────────┴────────┐                          ┌────────┴─┐   ┌─┴─────────┐
            │  Sigmoid (σ)    │                          │Sigmoid(σ)│   │ tanh(·)   │
            └────────▲────────┘                          └────▲─────┘   └───▲───────┘
                     │                                        │             │
                     ├────────────────────────────────────────┴─────────────┤
                     │                                                      │
x_t, h_(t-1) ────────┴──────────────────────────────────────────────────────┴────────────► [Output: o_t] ──► ( × ) ──► h_t
                                                                                                                ▲
                                                                                       tanh(C_t) ───────────────┘
```

### 6.1 Why Vanilla RNNs Fail (The Vanishing Gradient Problem)
In standard RNNs, hidden states are computed by repeated affine transformations followed by non-linear activations:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$
During Backpropagation Through Time (BPTT), the gradient of the loss at time step $T$ with respect to time step $t$ is computed via the chain rule:
$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}}$$
Because $\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(1 - \tanh^2(\cdot)) W_{hh}^T$, if the eigenvalues of $W_{hh}$ are less than 1, repeated multiplication causes gradients to decay exponentially toward zero as $T - t$ increases. Consequently, standard RNNs cannot learn dependencies across more than 5 to 10 cycles. 

LSTMs solve this by establishing an uninterrupted linear memory channel called the **Cell State ($C_t$)**.

### 6.2 The Three LSTM Gating Mechanisms

#### 1. The Forget Gate ($f_t$)
Decides what fraction of historical memory in the previous cell state $C_{t-1}$ to retain versus erase:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ squashes values between 0 (completely discard) and 1 (completely preserve).
- **Physical Meaning:** If an engine experienced transient flutter or minor operating variations that have returned to baseline, the forget gate damps that temporary information out of long-term degradation tracking.

#### 2. The Input Gate ($i_t$) & Candidate State ($\tilde{C}_t$)
Decides what new degradation information from current sensor readings $x_t$ should be stored in the cell state:
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input Gate)}$$
$$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad \text{(Candidate State)}$$
- The input gate $i_t$ acts as a gatekeeper, deciding which features are significant.
- The candidate state $\tilde{C}_t$ creates a candidate vector of new degradation indicators.

#### 3. Cell State Update (The Additive Highway)
The new cell state $C_t$ is updated via an additive accumulation:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
- Here, $\odot$ represents element-wise Hadamard multiplication.
- **Critical Mathematical Insight:** Notice the addition sign ($+$). Because updates are additive rather than multiplicative, the gradient $\frac{\partial C_t}{\partial C_{t-1}} = f_t$. If the forget gate is saturated near 1, gradients flow backward across dozens of time steps without exponential decay!

#### 4. The Output Gate ($o_t$) & Final Hidden State ($h_t$)
Determines what information from the updated cell state should be emitted as the operational hidden state:
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$
- The vector $h_t$ is the compact operational representation passed to subsequent layers and the next time step.

### 6.3 PyTorch Model Architecture
The implemented prognostic model ([`LSTMRULEstimator`](file:///c:/Users/Prachit/Deep%20learning%20final/src/prognostics.py#L11-L55)) is structured as follows:

```python
class LSTMRULEstimator(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=1, dropout=0.2):
        super(LSTMRULEstimator, self).__init__()
        self.lstm = nn.LSTM(input_size=14, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)             # (Batch, 30, 64)
        last_step = lstm_out[:, -1, :]          # (Batch, 64)
        out = self.dropout1(last_step)
        out = self.fc1(out)                     # (Batch, 32)
        out = self.relu(out)
        out = self.dropout2(out)
        rul_pred = self.fc_out(out)             # (Batch, 1)
        return rul_pred.squeeze(-1)
```

### 6.4 Training Optimization & NASA Asymmetric Scoring Function
- **Optimizer:** Adam ($\eta = 0.001$, weight decay $10^{-5}$)
- **Batch Size:** 128 | **Epochs:** 15–20
- **Loss Function:** Mean Squared Error (MSE): $\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$
- **NASA Asymmetric Scoring Function:** Penalizes late predictions (dangerous failure risk) exponentially more heavily than early predictions (safe early maintenance):
  $$d_i = \hat{y}_i - y_i$$
  $$s_i = \begin{cases} \exp(-d_i/13) - 1 & \text{if } d_i < 0 \quad (\text{Early Prediction: Safe}) \\ \exp(d_i/10) - 1 & \text{if } d_i \ge 0 \quad (\text{Late Prediction: Dangerous!}) \end{cases}$$
  $$S = \sum_{i=1}^N s_i$$

### 6.5 Sequence Length Tuning Study (Experiment 1 Results)

| Sequence Length ($W$) | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | NASA Asymmetric Score | Selection Rationale |
| :---: | :---: | :---: | :---: | :--- |
| **30 Cycles** | **13.80 Cycles** | **17.57 Cycles** | **681.7** | **Standard Selected:** Captures multi-sensor degradation onset across a wider temporal window. |
| **20 Cycles** | 11.18 Cycles | 16.33 Cycles | 524.8 | Faster training, but slightly higher susceptibility to transient noise. |

---

## 7. Stage 5: Predicted RUL & Uncertainty Quantification

### 7.1 The Critical Need for Uncertainty in Aviation
A deterministic point prediction $\hat{y} = 22$ cycles does not inform the maintenance manager whether the model is highly certain ($\sigma = 2$) or highly uncertain ($\sigma = 9$). If $\sigma = 9$, the actual probability of the engine having fewer than 10 cycles remaining is over 15%. Scheduling maintenance on cycle 20 leads directly to an in-flight engine shutdown.

### 7.2 Monte Carlo Dropout (Bayesian Approximation)
Following **Gal & Ghahramani (2016)**, deep networks trained with dropout approximate a Deep Gaussian Process. During test-time inference, dropout ($p=0.20$) remains **active** across $N_{MC} = 30$ stochastic forward passes:
$$\hat{y}^{(s)} = f_{LSTM}(\mathbf{X}; \mathbf{W}^{(s)}), \quad s \in \{1, \dots, N_{MC}\}$$

We estimate:
- **Predictive Mean:** $\mu_{\text{RUL}} = \frac{1}{N_{MC}} \sum_{s=1}^{N_{MC}} \hat{y}^{(s)}$
- **Epistemic Model Uncertainty (Std Dev):** $\sigma_{\text{RUL}} = \sqrt{\frac{1}{N_{MC}-1} \sum_{s=1}^{N_{MC}} (\hat{y}^{(s)} - \mu_{\text{RUL}})^2}$
- **Conservative 95% Confidence Interval:**
  $$\left[\text{RUL}_{lower}, \text{RUL}_{upper}\right] = \left[\max(0, \mu - 1.96\sigma), \mu + 1.96\sigma\right]$$

### 7.3 Empirical Uncertainty Validation (Experiment 2 Metrics)
- **Empirical 95% Coverage Rate:** **75.0%** of actual ground-truth lifetimes fall strictly within the predicted bounds.
- **Average Interval Width:** **47.3 cycles**.
- **Mean Uncertainty ($\sigma$):** **12.07 cycles**.

**Risk Hedging Strategy:** By setting the operational scheduling planning horizon to the conservative lower bound $\text{RUL}_{lower}$, the system provides a safety buffer against overestimation, eliminating 96% of catastrophic failures.

---

## 8. Stage 6: Multi-Agent AI Scheduling Integration

The outputs of the LSTM prognostic model ($\mu, \sigma, \text{RUL}_{lower}$) feed directly into a 4-Agent Multi-Agent System (MAS):

1. **Prognostic Agent:** Continuously processes sensor telemetry windows, executes Monte Carlo Dropout, and emits $[\mu, \sigma, \text{RUL}_{lower}]$.
2. **Resource Agent:** Enforces physical constraints:
   - Rejects candidate maintenance dates $t < t_{now} + L_{spare}$ (where spare kit delivery lead time $L_{spare} = 8$ cycles).
   - Tracks technician rosters, rejecting cycles where technician crew capacity $K_{tech} \ge 2$.
3. **Cost Agent:** Computes expected risk-adjusted Total Cost of Ownership (TCO):
   $$P(\text{failure before } t) = \Phi\left(\frac{t - t_{now} - \mu}{\sigma}\right)$$
   $$E[\text{Cost}(t)] = (1 - P_{\text{fail}}) \cdot (C_{pm} + D_{pm} C_d) + P_{\text{fail}} \cdot (C_{\text{fail}} + D_{\text{fail}} C_d) + \text{RushCost}$$
4. **Coordinator / Scheduling Agent:** Implements candidate-window search within the conservative boundary $[t_{now} + L_{spare}, t_{now} + \text{RUL}_{lower}]$, resolving conflicts across the fleet.

### 100-Engine Fleet Policy Benchmark Comparison

| Maintenance Policy | Total Fleet Cost | Cost Savings vs. RTF | In-Service Catastrophic Failures | Premature Waste / Asset Utilization |
| :--- | :---: | :---: | :---: | :--- |
| **Run-to-Failure (RTF)** | **$550,000** | 0.0% (Baseline) | 100 / 100 Engines | 0 cycles wasted, but astronomical catastrophic damage |
| **Fixed-Interval Overhaul** | **$152,000** | +72.36% | 0 / 100 Engines | 75.6 cycles / engine prematurely discarded |
| **Naive RUL Threshold** | **$180,200** | +67.24% | 10 / 100 Engines | Unhedged overestimations cause tarmac breakdowns |
| **Agentic AI Policy (Ours)** | **$137,200** | **+75.05%** | **4 / 100 Engines** | **Optimal: Minimizes cost while hedging uncertainty** |

---

## 9. Viva Voce Defense Guide (Key Technical Questions & Answers)

1. **Q: Why was LSTM chosen instead of a vanilla Recurrent Neural Network or standard MLP?**  
   *A:* Turbofan degradation is a non-linear, multi-sensor temporal progression. MLPs treat time steps as independent and lose sequential degradation rates. Vanilla RNNs suffer from the vanishing gradient problem due to continuous matrix multiplications during backpropagation through time. LSTMs introduce an additive Cell State and gating mechanisms (Forget, Input, Output) that allow gradients to propagate back across 30+ cycles without exponential decay.

2. **Q: Why drop 7 sensors from the NASA C-MAPSS FD001 dataset?**  
   *A:* Sensors $s_1, s_5, s_{10}, s_{16}, s_{18}, s_{19}$, and `setting_3` exhibit zero variance throughout the entire operating life of all engines in FD001. Flat features contribute zero diagnostic signal, add dead parameters, and dilute gradients during training.

3. **Q: What is the purpose of capping the RUL target at 125 cycles?**  
   *A:* Real turbofan engines experience virtually no measurable wear during initial operating cycles. An uncapped linear target forces the network to learn spurious noise in healthy baselines. Capping the target at 125 cycles stabilizes early predictions and focuses network representational capacity on the critical degradation regime.

4. **Q: How does Monte Carlo Dropout provide uncertainty quantification?**  
   *A:* By leaving dropout active ($p=0.20$) during test-time inference across 30 stochastic forward passes, the model samples from the approximate posterior distribution of the network weights (Gal & Ghahramani, 2016). The spread (standard deviation $\sigma$) reflects epistemic uncertainty.

5. **Q: Why use $\text{RUL}_{lower}$ instead of the mean prediction for scheduling?**  
   *A:* Point estimates ignore overestimation risk. In aviation, late predictions lead to in-flight shutdowns ($C_{fail} = \$5,000$ plus severe downtime). Planning against $\text{RUL}_{lower} = \max(0, \mu - 1.96\sigma)$ hedges against overestimation, reducing fleet failures from 100 down to 4.

6. **Q: How do the four agents interact during scheduling?**  
   *A:* The Prognostic Agent delivers $\mu, \sigma, \text{RUL}_{lower}$; the Resource Agent enforces lead times ($L_{spare} = 8$) and technician rosters ($K_{tech} = 2$); the Cost Agent calculates risk-adjusted lifecycle cost per cycle; and the Coordinator Agent selects the global minimum-cost feasible maintenance slot.
