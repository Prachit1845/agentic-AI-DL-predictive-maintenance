import base64
import os

def get_base64_img(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(img_path)[1].replace(".", "")
            if ext == "jpg":
                ext = "jpeg"
            return f"data:image/{ext};base64,{encoded}"
    return ""

def create_html_report():
    pipe_img = get_base64_img("results/pipeline_architecture_diagram.png")
    cell_img = get_base64_img("results/lstm_cell_diagram.png")
    chart_img = get_base64_img("results/exp_policy_comparison_charts.png")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PS-05 | Agentic Predictive Maintenance Scheduling & LSTM Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --primary: #1E3A8A;
      --primary-light: #3B82F6;
      --accent: #0D9488;
      --text-main: #1E293B;
      --text-muted: #64748B;
      --bg-page: #F8FAFC;
      --bg-card: #FFFFFF;
      --border: #E2E8F0;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-page);
      color: var(--text-main);
      line-height: 1.65;
      padding: 30px 20px;
    }}
    .container {{
      max-width: 900px;
      margin: 0 auto;
      background: var(--bg-card);
      border-radius: 16px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
      padding: 48px;
      border: 1px solid var(--border);
    }}
    .header {{
      text-align: center;
      padding-bottom: 24px;
      border-bottom: 2px solid var(--border);
      margin-bottom: 32px;
    }}
    .badge {{
      display: inline-block;
      background: #EFF6FF;
      color: #1D4ED8;
      font-weight: 700;
      font-size: 13px;
      padding: 6px 16px;
      border-radius: 9999px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      margin-bottom: 12px;
      border: 1px solid #BFDBFE;
    }}
    h1 {{
      font-size: 26px;
      font-weight: 800;
      color: var(--primary);
      margin-bottom: 8px;
      line-height: 1.25;
    }}
    .subtitle {{
      font-size: 15px;
      color: var(--text-muted);
      font-weight: 400;
    }}
    .meta-box {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      background: #F1F5F9;
      padding: 16px;
      border-radius: 10px;
      margin-top: 20px;
      text-align: left;
    }}
    .meta-item span {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      color: var(--text-muted);
      font-weight: 600;
    }}
    .meta-item strong {{
      font-size: 13px;
      color: #0F172A;
    }}
    h2 {{
      font-size: 20px;
      font-weight: 700;
      color: var(--primary);
      margin-top: 36px;
      margin-bottom: 16px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
    }}
    h3 {{
      font-size: 16px;
      font-weight: 600;
      color: var(--accent);
      margin-top: 20px;
      margin-bottom: 8px;
    }}
    p {{
      margin-bottom: 14px;
      font-size: 14.5px;
      color: #334155;
    }}
    ul, ol {{
      margin-left: 24px;
      margin-bottom: 16px;
      font-size: 14px;
      color: #334155;
    }}
    li {{
      margin-bottom: 6px;
    }}
    .callout {{
      background: #F0F9FF;
      border-left: 4px solid var(--primary-light);
      padding: 16px 20px;
      border-radius: 0 8px 8px 0;
      margin: 20px 0;
    }}
    .callout strong {{
      color: #0369A1;
      display: block;
      margin-bottom: 4px;
      font-size: 13px;
      text-transform: uppercase;
    }}
    .callout p {{
      margin: 0;
      font-size: 13.5px;
      color: #0C4A6E;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 14px;
      text-align: left;
      border: 1px solid var(--border);
    }}
    th {{
      background: #1E3A8A;
      color: #FFFFFF;
      font-weight: 600;
      font-size: 12.5px;
      text-transform: uppercase;
    }}
    tr:nth-child(even) {{
      background: #F8FAFC;
    }}
    .img-card {{
      background: #FFFFFF;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      margin: 24px 0;
      text-align: center;
    }}
    .img-card img {{
      max-width: 100%;
      height: auto;
      border-radius: 8px;
    }}
    .caption {{
      font-size: 12px;
      color: var(--text-muted);
      font-style: italic;
      margin-top: 8px;
    }}
    .formula-box {{
      background: #F8FAFC;
      border: 1px solid #CBD5E1;
      border-radius: 8px;
      padding: 12px 18px;
      font-family: 'Fira Code', monospace;
      font-size: 13px;
      color: #0F172A;
      margin: 12px 0 16px 0;
      overflow-x: auto;
    }}
    .action-bar {{
      position: sticky;
      top: 10px;
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-bottom: 20px;
      z-index: 100;
    }}
    .btn {{
      background: #2563EB;
      color: white;
      padding: 10px 18px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      border: none;
      cursor: pointer;
      box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
      transition: background 0.2s;
    }}
    .btn:hover {{
      background: #1D4ED8;
    }}
    .q-card {{
      background: #F8FAFC;
      border-left: 3px solid #10B981;
      padding: 14px 18px;
      margin: 14px 0;
      border-radius: 0 8px 8px 0;
    }}
    .q-title {{
      font-weight: 700;
      color: #065F46;
      font-size: 13.5px;
      margin-bottom: 4px;
    }}
    @media print {{
      body {{
        padding: 0;
        background: white;
      }}
      .container {{
        box-shadow: none;
        border: none;
        padding: 0;
        max-width: 100%;
      }}
      .action-bar {{
        display: none;
      }}
    }}
  </style>
</head>
<body>

  <div class="action-bar">
    <button class="btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
  </div>

  <div class="container">
    <div class="header">
      <div class="badge">PS-05 | TY B.Tech Capstone | 150 Marks</div>
      <h1>Agentic Predictive Maintenance Scheduling under Remaining-Useful-Life (RUL) Uncertainty</h1>
      <div class="subtitle">Deep Learning (LSTM) Prognostics + Monte Carlo Dropout + 4-Agent Optimization</div>
      
      <div class="meta-box">
        <div class="meta-item"><span>Benchmark Dataset</span><strong>NASA C-MAPSS (FD001)</strong></div>
        <div class="meta-item"><span>Model Type</span><strong>PyTorch Deep LSTM</strong></div>
        <div class="meta-item"><span>Uncertainty</span><strong>Monte Carlo Dropout (N=30)</strong></div>
        <div class="meta-item"><span>Fleet Cost Savings</span><strong>75.05% vs Run-to-Failure</strong></div>
      </div>
    </div>

    <!-- Section 1 -->
    <h2>1. Executive Summary & Problem Formulation</h2>
    <p>Predictive Maintenance (PdM) is essential for aircraft engine safety and airline operational economics. Standard deep learning regressors output a single point estimate for Remaining Useful Life (RUL). However, point estimates ignore model epistemic uncertainty, and standard threshold rules fail when real-world constraints such as spare-part lead times and technician bottlenecks exist.</p>
    
    <div class="callout">
      <strong>Core Project Achievement</strong>
      <p>By connecting a PyTorch LSTM model with Monte Carlo Dropout uncertainty bounds to a 4-Agent Scheduling Layer, this project cuts turbofan fleet costs by 75.05% compared to Run-to-Failure and prevents 96 out of 100 in-service catastrophic breakdowns.</p>
    </div>

    <!-- Section 2: Pipeline Architecture -->
    <h2>2. Complete Pipeline Architecture (From Image Flowchart)</h2>
    <p>The workflow consists of 5 core sequential stages, connecting raw NASA C-MAPSS telemetry to uncertainty-aware maintenance scheduling:</p>

    <div class="img-card">
      <img src="{pipe_img}" alt="PS-05 Complete Pipeline Flowchart">
      <div class="caption">Figure 1: PS-05 End-to-End System Pipeline Flowchart (as specified in Project Blueprint)</div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Stage #</th>
          <th>Stage Name</th>
          <th>Input</th>
          <th>Transformation / Method</th>
          <th>Output</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Stage 1</strong></td>
          <td>NASA C-MAPSS Dataset</td>
          <td>train_FD001.txt, test_FD001.txt</td>
          <td>Simulate run-to-failure degradation trajectories</td>
          <td>100 Train & 100 Test engines (26 columns)</td>
        </tr>
        <tr>
          <td><strong>Stage 2</strong></td>
          <td>Data Cleaning + Sensor Selection</td>
          <td>21 sensors + 3 flight settings</td>
          <td>Drop 7 flat sensors, MinMax scale [0,1], Cap RUL at 125</td>
          <td>14 informative sensor streams + bounded RUL</td>
        </tr>
        <tr>
          <td><strong>Stage 3</strong></td>
          <td>Sequence Generation</td>
          <td>Tabular scaled data</td>
          <td>Sliding temporal window (Length W = 30 cycles)</td>
          <td>3D Tensors: (Batch, 30, 14)</td>
        </tr>
        <tr>
          <td><strong>Stage 4</strong></td>
          <td>LSTM RUL Model</td>
          <td>3D Sequential Tensors</td>
          <td>PyTorch LSTM (64 units) + Dropout (0.2) + FC(32) + FC(1)</td>
          <td>Trained model weights (models/best_lstm_model.pt)</td>
        </tr>
        <tr>
          <td><strong>Stage 5</strong></td>
          <td>Predicted RUL + Uncertainty Range</td>
          <td>Test engine sequence windows</td>
          <td>Monte Carlo Dropout (N=30 stochastic forward passes)</td>
          <td>Mean μ, Std Dev σ, Safe Bound [RUL_lower, RUL_upper]</td>
        </tr>
        <tr>
          <td><strong>Stage 6</strong></td>
          <td>Multi-Agent Scheduling Layer</td>
          <td>Prognostic bounds + Fleet roster</td>
          <td>4-Agent negotiation under spare lead times & technician limits</td>
          <td>Optimal maintenance schedule & 75.05% cost saving</td>
        </tr>
      </tbody>
    </table>

    <!-- Section 3: Data Cleaning -->
    <h2>3. Data Cleaning & Sensor Selection</h2>
    <p>Raw engine telemetry cannot be fed directly into deep learning models due to uninformative features and disparate measurement scales:</p>
    <ul>
      <li><strong>Dropped Constant Sensors (Variance = 0):</strong> Sensors <code>s_1, s_5, s_10, s_16, s_18, s_19</code> and <code>setting_3</code> have zero variance in FD001. Flat features create dead weights and dilute gradients without diagnostic utility.</li>
      <li><strong>Selected 14 Informative Sensors:</strong> <code>s_2, s_3, s_4, s_7, s_8, s_9, s_11, s_12, s_13, s_14, s_15, s_17, s_20, s_21</code> show strong monotonic drift as High-Pressure Compressor (HPC) wear progresses.</li>
      <li><strong>Min-Max Normalization:</strong> Maps all 14 channels into [0, 1] using training set statistics.</li>
      <li><strong>Piecewise Capped RUL (125 Cycles):</strong> Wear during early flight hours is negligible. Capping at 125 cycles stabilizes early predictions and prevents the network from fitting noise in healthy engines.</li>
    </ul>

    <!-- Section 4: LSTM Working -->
    <h2>4. LSTM RUL Model & Comprehensive Working Mechanism</h2>
    <p>Long Short-Term Memory (LSTM) networks are designed specifically to overcome the <strong>Vanishing Gradient Problem</strong> of vanilla RNNs, enabling models to learn degradation trends spanning 30+ cycles.</p>

    <div class="img-card">
      <img src="{cell_img}" alt="LSTM Internal Cell Diagram">
      <div class="caption">Figure 2: Architectural Anatomy and Mathematical Information Flow inside the LSTM Cell</div>
    </div>

    <h3>Internal Gating Equations</h3>
    <p>An LSTM cell maintains a <strong>Cell State ($C_t$)</strong> for long-term memory and a <strong>Hidden State ($h_t$)</strong> for short-term working output:</p>
    
    <div class="formula-box">
1. Forget Gate:    f_t = σ(W_f · [h_(t-1), x_t] + b_f)        (Decides what to discard)
2. Input Gate:     i_t = σ(W_i · [h_(t-1), x_t] + b_i)        (Decides what to update)
3. Candidate:      C~_t = tanh(W_c · [h_(t-1), x_t] + b_c)    (Creates candidate updates)
4. Memory Highway: C_t = f_t ⊙ C_(t-1) + i_t ⊙ C~_t          (Additive cell state update)
5. Output Gate:    o_t = σ(W_o · [h_(t-1), x_t] + b_o)        (Decides what to emit)
6. Hidden State:   h_t = o_t ⊙ tanh(C_t)                      (Final operational output)
    </div>

    <h3>PyTorch Implementation Architecture</h3>
    <div class="formula-box">
Input (Batch, 30, 14) 
  ──► nn.LSTM(input_size=14, hidden_size=64, batch_first=True)
  ──► Last Time Step Slice [:, -1, :] (Batch, 64)
  ──► nn.Dropout(p=0.20)
  ──► nn.Linear(64, 32) + nn.ReLU()
  ──► nn.Dropout(p=0.20)
  ──► nn.Linear(32, 1) ──► Scalar RUL Prediction
    </div>

    <!-- Section 5: Uncertainty -->
    <h2>5. Predicted RUL & Uncertainty Quantification (Monte Carlo Dropout)</h2>
    <p>Rather than disabling dropout during test inference, <strong>Monte Carlo Dropout</strong> (Gal & Ghahramani, 2016) keeps dropout active (p=0.20) across N_MC = 30 stochastic forward passes:</p>
    
    <div class="formula-box">
• Predictive Mean:       μ_RUL = (1 / N_MC) * Σ y^(s)
• Epistemic Uncertainty:  σ_RUL = sqrt( (1 / (N_MC - 1)) * Σ (y^(s) - μ_RUL)^2 )
• Conservative 95% Bound: [RUL_lower, RUL_upper] = [max(0, μ - 1.96*σ), μ + 1.96*σ]
    </div>

    <p><strong>Operational Impact:</strong> Planning maintenance against <code>RUL_lower</code> provides a safety buffer against overestimation, which is the primary cause of tarmac failures in deterministic scheduling.</p>

    <!-- Section 6: Multi-Agent & Results -->
    <h2>6. Multi-Agent Scheduling & Fleet Economic Results</h2>
    <p>The 4-Agent Multi-Agent System (Prognostic Agent, Resource Agent, Cost Agent, Coordinator Agent) schedules maintenance under spare-part lead times ($L=8$) and technician crew capacities ($K=2$).</p>

    <div class="img-card">
      <img src="{chart_img}" alt="Fleet Maintenance Policy Benchmark">
      <div class="caption">Figure 3: 100-Engine Fleet Benchmark: Total Cost, Failure Counts, and Asset Utilization</div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Maintenance Policy</th>
          <th>Total Fleet Cost</th>
          <th>Cost Savings vs RTF</th>
          <th>In-Service Catastrophic Failures</th>
          <th>Premature Asset Waste</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Run-to-Failure (RTF)</strong></td>
          <td>$550,000</td>
          <td>0.0% (Baseline)</td>
          <td>100 / 100 Engines</td>
          <td>0 cycles (100% life extracted)</td>
        </tr>
        <tr>
          <td><strong>Fixed-Interval Overhaul</strong></td>
          <td>$152,000</td>
          <td>+72.36%</td>
          <td>0 / 100 Engines</td>
          <td>75.6 cycles / engine prematurely discarded!</td>
        </tr>
        <tr>
          <td><strong>Naive RUL Threshold</strong></td>
          <td>$180,200</td>
          <td>+67.24%</td>
          <td>10 / 100 Engines</td>
          <td>Overestimation causes tarmac breakdowns</td>
        </tr>
        <tr>
          <td><strong>Agentic AI Policy (Ours)</strong></td>
          <td><strong>$137,200</strong></td>
          <td><strong>+75.05%</strong></td>
          <td><strong>4 / 100 Engines</strong></td>
          <td><strong>Optimal: Minimizes cost & hedges uncertainty</strong></td>
        </tr>
      </tbody>
    </table>

    <!-- Section 7: Viva Defense -->
    <h2>7. Viva Voce Defense Guide (Key Technical Q&A)</h2>
    <div class="q-card">
      <div class="q-title">Q1: Why choose LSTM over vanilla RNNs?</div>
      <p>Vanilla RNNs suffer from the vanishing gradient problem due to repeated matrix multiplications during backpropagation through time. LSTMs introduce an additive Cell State highway that allows error gradients to propagate back across 30+ cycles without exponential decay.</p>
    </div>
    <div class="q-card">
      <div class="q-title">Q2: Why drop 7 sensors out of 21?</div>
      <p>Sensors s1, s5, s10, s16, s18, s19 and setting_3 have zero variance in subset FD001. Flat features contribute zero diagnostic signal and create dead weights in neural networks.</p>
    </div>
    <div class="q-card">
      <div class="q-title">Q3: Why cap the RUL target at 125 cycles?</div>
      <p>Wear in early cycles is negligible. An uncapped linear target forces the network to fit random sensor noise during healthy operation. Capping at 125 focuses representational capacity on the non-linear degradation regime.</p>
    </div>
    <div class="q-card">
      <div class="q-title">Q4: Why plan with RUL_lower instead of the mean?</div>
      <p>A point estimate of 20 cycles with standard deviation of 6 has a high probability of failing before 10 cycles. Planning with RUL_lower hedges against overestimation risk, eliminating 96% of catastrophic failures.</p>
    </div>

  </div>
</body>
</html>
"""
    with open("LSTM_Report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Successfully generated self-contained LSTM_Report.html")

if __name__ == "__main__":
    create_html_report()
