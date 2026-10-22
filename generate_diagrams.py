import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs("results", exist_ok=True)

def generate_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Title Banner
    header_box = patches.FancyBboxPatch((0.5, 9.8), 9.0, 0.9, boxstyle="round,pad=0.2",
                                        facecolor="#1E3A8A", edgecolor="#172554", linewidth=2)
    ax.add_patch(header_box)
    ax.text(5.0, 10.25, "PS-05 | Agentic Predictive Maintenance Scheduling | TY B.Tech | 150 Marks",
            ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    
    # Stages definition
    stages = [
        ("NASA C-MAPSS FD001 Dataset", 
         "100 Turbofan Engines (Run-to-Failure) | 21 Sensor Channels + 3 Settings\nGround Truth Remaining Useful Life (RUL_FD001.txt)", 
         "#0284C7", 8.5),
        ("Data Cleaning + Sensor Selection", 
         "Remove 7 Constant Sensors (s1, s5, s10, s16, s18, s19, setting_3)\nSelect 14 Informative Sensors + MinMax Normalization [0, 1] + Piecewise Capped RUL (Cap=125)", 
         "#0D9488", 6.8),
        ("Sequence Generation", 
         "Sliding Window (Length W = 30 Cycles) over 14 Sensors -> Tensor Shape (30, 14)\nCaptures Temporal Multi-Sensor Degradation Trajectory", 
         "#D97706", 5.1),
        ("LSTM RUL Model", 
         "PyTorch LSTM (64 Hidden Units) + Dropout (p=0.2) + Dense FC(32) + ReLU + Dense FC(1)\nLoss: MSE Loss | NASA Asymmetric Evaluation Scoring Metric", 
         "#4F46E5", 3.4),
        ("Predicted RUL + Uncertainty Range", 
         "Monte Carlo Dropout (N=30 Iterations at Test Time)\nOutputs: Mean RUL (μ), Std Dev (σ), and Conservative 95% Bound [RUL_lower, RUL_upper]", 
         "#9333EA", 1.7),
        ("Multi-Agent Maintenance Scheduling Layer", 
         "Prognostic Agent -> Coordinator Agent -> Resource Agent (Lead Time=8, Staff=2) + Cost Agent\nOptimal Feasible Maintenance Cycle Selection (75.05% Cost Reduction vs RTF)", 
         "#16A34A", 0.0)
    ]
    
    box_width = 8.4
    box_height = 1.15
    left_x = 0.8
    
    for i, (title, desc, color, y_pos) in enumerate(stages):
        # Draw box
        box = patches.FancyBboxPatch((left_x, y_pos), box_width, box_height, boxstyle="round,pad=0.2",
                                     facecolor="white", edgecolor=color, linewidth=2.5)
        ax.add_patch(box)
        
        # Color bar indicator on left
        bar = patches.FancyBboxPatch((left_x, y_pos), 0.3, box_height, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor=color)
        ax.add_patch(bar)
        
        # Step number badge
        badge = patches.Circle((left_x + 0.7, y_pos + box_height/2), 0.28, facecolor=color, edgecolor="none")
        ax.add_patch(badge)
        ax.text(left_x + 0.7, y_pos + box_height/2, str(i + 1), ha='center', va='center',
                color='white', fontsize=11, fontweight='bold')
        
        # Title
        ax.text(left_x + 1.2, y_pos + box_height - 0.35, title, ha='left', va='center',
                color='#0F172A', fontsize=12, fontweight='bold')
        
        # Description
        ax.text(left_x + 1.2, y_pos + 0.35, desc, ha='left', va='center',
                color='#475569', fontsize=8.5, multialignment='left')
        
        # Draw connecting arrow to next box
        if i < len(stages) - 1:
            next_y = stages[i+1][3] + box_height
            curr_y = y_pos
            ax.annotate('', xy=(5.0, next_y), xytext=(5.0, curr_y),
                        arrowprops=dict(arrowstyle="simple,tail_width=0.15,head_width=0.6,head_length=0.6",
                                        color="#64748B", shrinkA=4, shrinkB=4))
            
    plt.tight_layout()
    output_path = "results/pipeline_architecture_diagram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

def generate_lstm_cell_diagram():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6.0, 7.6, "Internal Working Mechanism of Long Short-Term Memory (LSTM) Cell",
            ha='center', va='center', color='#1E293B', fontsize=13, fontweight='bold')
    ax.text(6.0, 7.15, "Cell State (Long-Term Memory) + 3 Regulating Gates (Forget, Input, Output)",
            ha='center', va='center', color='#64748B', fontsize=10)
    
    # Outer Cell Box
    cell_box = patches.FancyBboxPatch((2.2, 0.8), 7.6, 5.8, boxstyle="round,pad=0.3",
                                      facecolor="#F8FAFC", edgecolor="#3B82F6", linewidth=2.5, linestyle="--")
    ax.add_patch(cell_box)
    ax.text(2.6, 6.2, "LSTM Cell (Time step t)", color="#2563EB", fontsize=11, fontweight='bold')
    
    # Cell State Line (Top Conveyor Belt)
    ax.annotate('', xy=(10.5, 5.4), xytext=(1.0, 5.4),
                arrowprops=dict(arrowstyle="->", lw=3.0, color="#D97706"))
    ax.text(0.8, 5.6, "C_(t-1)\n(Past Cell State)", ha='center', va='center', color="#B45309", fontsize=9, fontweight='bold')
    ax.text(11.2, 5.6, "C_t\n(Updated Cell State)", ha='center', va='center', color="#B45309", fontsize=9, fontweight='bold')
    
    # Gate 1: Forget Gate (Yellow/Orange)
    fg_box = patches.FancyBboxPatch((3.2, 2.8), 1.3, 1.0, boxstyle="round,pad=0.1",
                                    facecolor="#FEF3C7", edgecolor="#F59E0B", linewidth=2)
    ax.add_patch(fg_box)
    ax.text(3.85, 3.4, "σ", ha='center', va='center', color="#92400E", fontsize=14, fontweight='bold')
    ax.text(3.85, 3.0, "Forget Gate", ha='center', va='center', color="#B45309", fontsize=8, fontweight='bold')
    ax.text(3.85, 2.4, "f_t = σ(W_f·[h_prev, x_t])", ha='center', va='center', color="#475569", fontsize=7.5)
    
    # Gate 2: Input Gate & Candidate State (Green & Cyan)
    ig_box = patches.FancyBboxPatch((5.1, 2.8), 1.3, 1.0, boxstyle="round,pad=0.1",
                                    facecolor="#DCFCE7", edgecolor="#16A34A", linewidth=2)
    ax.add_patch(ig_box)
    ax.text(5.75, 3.4, "σ", ha='center', va='center', color="#15803D", fontsize=14, fontweight='bold')
    ax.text(5.75, 3.0, "Input Gate", ha='center', va='center', color="#166534", fontsize=8, fontweight='bold')
    ax.text(5.75, 2.4, "i_t = σ(W_i·[h_prev, x_t])", ha='center', va='center', color="#475569", fontsize=7.5)
    
    cand_box = patches.FancyBboxPatch((6.8, 2.8), 1.3, 1.0, boxstyle="round,pad=0.1",
                                      facecolor="#E0F2FE", edgecolor="#0284C7", linewidth=2)
    ax.add_patch(cand_box)
    ax.text(7.45, 3.4, "tanh", ha='center', va='center', color="#0369A1", fontsize=12, fontweight='bold')
    ax.text(7.45, 3.0, "Candidate", ha='center', va='center', color="#075985", fontsize=8, fontweight='bold')
    ax.text(7.45, 2.4, "C~_t = tanh(W_c·[h, x])", ha='center', va='center', color="#475569", fontsize=7.5)
    
    # Gate 3: Output Gate (Purple)
    og_box = patches.FancyBboxPatch((8.3, 2.8), 1.2, 1.0, boxstyle="round,pad=0.1",
                                    facecolor="#F3E8FF", edgecolor="#9333EA", linewidth=2)
    ax.add_patch(og_box)
    ax.text(8.9, 3.4, "σ", ha='center', va='center', color="#7E22CE", fontsize=14, fontweight='bold')
    ax.text(8.9, 3.0, "Output Gate", ha='center', va='center', color="#6B21A8", fontsize=8, fontweight='bold')
    ax.text(8.9, 2.4, "o_t = σ(W_o·[h, x])", ha='center', va='center', color="#475569", fontsize=7.5)
    
    # Operators on cell state line
    # Multiplier 1: f_t * C_(t-1)
    mult1 = patches.Circle((3.85, 5.4), 0.28, facecolor="white", edgecolor="#D97706", linewidth=2)
    ax.add_patch(mult1)
    ax.text(3.85, 5.4, "×", ha='center', va='center', color="#B45309", fontsize=14, fontweight='bold')
    # Connecting arrow from forget gate to multiplier 1
    ax.annotate('', xy=(3.85, 5.12), xytext=(3.85, 3.8), arrowprops=dict(arrowstyle="->", lw=1.5, color="#F59E0B"))
    
    # Multiplier 2: i_t * C~_t
    mult2 = patches.Circle((6.6, 4.4), 0.25, facecolor="white", edgecolor="#0284C7", linewidth=2)
    ax.add_patch(mult2)
    ax.text(6.6, 4.4, "×", ha='center', va='center', color="#0369A1", fontsize=12, fontweight='bold')
    ax.annotate('', xy=(6.4, 4.4), xytext=(5.75, 3.8), arrowprops=dict(arrowstyle="->", lw=1.5, color="#16A34A"))
    ax.annotate('', xy=(6.8, 4.4), xytext=(7.45, 3.8), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284C7"))
    
    # Adder on cell line: C_t = f_t * C_(t-1) + i_t * C~_t
    adder = patches.Circle((6.6, 5.4), 0.28, facecolor="white", edgecolor="#D97706", linewidth=2)
    ax.add_patch(adder)
    ax.text(6.6, 5.4, "+", ha='center', va='center', color="#B45309", fontsize=14, fontweight='bold')
    ax.annotate('', xy=(6.6, 5.12), xytext=(6.6, 4.65), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284C7"))
    
    # Output tanh and multiplier
    tanh_out = patches.FancyBboxPatch((8.5, 4.8), 0.8, 0.5, boxstyle="round,pad=0.05",
                                      facecolor="#F1F5F9", edgecolor="#64748B", linewidth=1.5)
    ax.add_patch(tanh_out)
    ax.text(8.9, 5.05, "tanh", ha='center', va='center', color="#334155", fontsize=9, fontweight='bold')
    ax.annotate('', xy=(8.9, 5.3), xytext=(7.5, 5.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))
    
    mult3 = patches.Circle((8.9, 4.1), 0.25, facecolor="white", edgecolor="#9333EA", linewidth=2)
    ax.add_patch(mult3)
    ax.text(8.9, 4.1, "×", ha='center', va='center', color="#7E22CE", fontsize=12, fontweight='bold')
    ax.annotate('', xy=(8.9, 4.35), xytext=(8.9, 4.8), arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748B"))
    ax.annotate('', xy=(8.9, 4.35), xytext=(8.9, 3.8), arrowprops=dict(arrowstyle="->", lw=1.5, color="#9333EA"))
    
    # Input bus at bottom (x_t and h_(t-1))
    ax.annotate('', xy=(1.0, 1.6), xytext=(0.5, 1.6), arrowprops=dict(arrowstyle="->", lw=2.0, color="#2563EB"))
    ax.text(0.7, 1.2, "h_(t-1)\n(Past Hidden)", ha='center', va='center', color="#1D4ED8", fontsize=8.5, fontweight='bold')
    
    ax.annotate('', xy=(2.0, 1.6), xytext=(2.0, 0.4), arrowprops=dict(arrowstyle="->", lw=2.0, color="#0F172A"))
    ax.text(2.0, 0.2, "x_t (Sensor Readings at t)", ha='center', va='center', color="#0F172A", fontsize=8.5, fontweight='bold')
    
    # Bus line
    ax.plot([1.0, 8.9], [1.6, 1.6], color="#64748B", lw=2)
    for x_bus in [3.85, 5.75, 7.45, 8.9]:
        ax.annotate('', xy=(x_bus, 2.8), xytext=(x_bus, 1.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748B"))
        
    # Output hidden state
    ax.annotate('', xy=(11.0, 4.1), xytext=(9.15, 4.1), arrowprops=dict(arrowstyle="->", lw=2.5, color="#2563EB"))
    ax.text(11.3, 4.1, "h_t\n(Hidden State / Output)", ha='center', va='center', color="#1D4ED8", fontsize=8.5, fontweight='bold')
    
    plt.tight_layout()
    output_path = "results/lstm_cell_diagram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_pipeline_diagram()
    generate_lstm_cell_diagram()
