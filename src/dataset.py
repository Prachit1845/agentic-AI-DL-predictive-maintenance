import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "CMAPSSData")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# NASA C-MAPSS column specifications
COLUMN_NAMES = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]

# Standard 14 informative sensors for C-MAPSS FD001
SELECTED_SENSORS = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9",
    "s_11", "s_12", "s_13", "s_14", "s_15", "s_17",
    "s_20", "s_21"
]

# Low variance / uninformative sensors in FD001
CONSTANT_SENSORS = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19", "setting_3"]

DATA_URLS = {
    "train_FD001.txt": "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/train_FD001.txt",
    "test_FD001.txt": "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/test_FD001.txt",
    "RUL_FD001.txt": "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/RUL_FD001.txt"
}


def download_dataset_if_needed(data_dir=DATA_DIR):
    """Ensures C-MAPSS FD001 dataset files exist locally, downloading if necessary."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    for filename, url in DATA_URLS.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"Successfully downloaded {filename}.")
        else:
            print(f"Found existing {filename} at {file_path}.")


def load_raw_data(data_dir=DATA_DIR):
    """Loads raw train, test, and RUL truth files into pandas DataFrames."""
    download_dataset_if_needed(data_dir)
    
    train_path = os.path.join(data_dir, "train_FD001.txt")
    test_path = os.path.join(data_dir, "test_FD001.txt")
    rul_path = os.path.join(data_dir, "RUL_FD001.txt")
    
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["true_rul"])
    rul_df["engine_id"] = rul_df.index + 1
    
    return train_df, test_df, rul_df


def add_rul_target(train_df, max_rul_cap=125):
    """
    Computes piecewise linear Remaining Useful Life (RUL) for training engines.
    Heimes (2008) and Zheng et al. (2017) demonstrated that early operating cycles
    do not exhibit measurable degradation, so capping the maximum target improves model stability.
    """
    df = train_df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.rename(columns={"cycle": "max_cycle"}, inplace=True)
    
    df = df.merge(max_cycles, on="engine_id", how="left")
    df["raw_rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["raw_rul"].clip(upper=max_rul_cap)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def prepare_preprocessed_data(data_dir=DATA_DIR, max_rul_cap=125):
    """
    Full preprocessing pipeline:
    1. Loads raw files
    2. Removes constant sensors
    3. Calculates capped RUL target on training set
    4. Fits MinMaxScaler on training selected sensors and transforms both train and test
    """
    train_df, test_df, rul_df = load_raw_data(data_dir)
    train_df = add_rul_target(train_df, max_rul_cap=max_rul_cap)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[SELECTED_SENSORS])
    
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    train_scaled[SELECTED_SENSORS] = scaler.transform(train_df[SELECTED_SENSORS])
    test_scaled[SELECTED_SENSORS] = scaler.transform(test_df[SELECTED_SENSORS])
    
    return train_scaled, test_scaled, rul_df, scaler


def create_sequences(df, seq_len=30, feature_cols=SELECTED_SENSORS, target_col="rul"):
    """
    Constructs sliding window sequences for training/validation.
    Returns:
        X: numpy array of shape (N, seq_len, num_features)
        y: numpy array of shape (N,)
    """
    sequences = []
    targets = []
    
    for engine_id, group in df.groupby("engine_id"):
        data = group[feature_cols].values
        labels = group[target_col].values
        num_records = len(data)
        
        if num_records >= seq_len:
            for start_idx in range(num_records - seq_len + 1):
                end_idx = start_idx + seq_len
                sequences.append(data[start_idx:end_idx])
                targets.append(labels[end_idx - 1])
                
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def create_test_sequences(test_df, rul_df, seq_len=30, feature_cols=SELECTED_SENSORS, max_rul_cap=125):
    """
    Constructs the final sequence window for each test engine to evaluate prediction at final observed cycle.
    Returns:
        X_test: numpy array of shape (num_test_engines, seq_len, num_features)
        y_test: numpy array of shape (num_test_engines,) true capped RUL
        engine_ids: list of engine IDs
    """
    sequences = []
    targets = []
    engine_ids = []
    
    for engine_id, group in test_df.groupby("engine_id"):
        data = group[feature_cols].values
        num_records = len(data)
        
        if num_records >= seq_len:
            seq = data[-seq_len:]
        else:
            # Pad with repeated earliest cycle if shorter than seq_len
            pad_len = seq_len - num_records
            pad = np.repeat(data[:1], pad_len, axis=0)
            seq = np.vstack([pad, data])
            
        true_rul_val = rul_df.loc[rul_df["engine_id"] == engine_id, "true_rul"].values[0]
        capped_rul_val = min(true_rul_val, max_rul_cap)
        
        sequences.append(seq)
        targets.append(capped_rul_val)
        engine_ids.append(engine_id)
        
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32), engine_ids


def run_eda_and_save_visualizations(train_df, save_dir=RESULTS_DIR):
    """
    Executes the 6 mandatory EDA analyses specified in the rubric:
    1. Engine life-cycle distribution
    2. Sensor trend plots for at least 5 engines
    3. Correlation heatmap of selected sensors
    4. Sensor variance analysis
    5. RUL distribution
    6. Comparison of early-life vs. late-life sensor behaviour
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Engine life-cycle distribution
    life_cycles = train_df.groupby("engine_id")["cycle"].max()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(life_cycles, bins=20, kde=True, color="#1f77b4", ax=ax)
    ax.axvline(life_cycles.mean(), color="red", linestyle="--", label=f"Mean Life: {life_cycles.mean():.1f} cycles")
    ax.set_title("1. Engine Life-Cycle Distribution (Time-to-Failure)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Max Operational Cycles")
    ax.set_ylabel("Engine Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "eda_1_engine_lifecycle_distribution.png"), dpi=200)
    plt.close(fig)
    
    # 2. Sensor trend plots for at least 5 engines (Sensors s_2, s_4, s_11, s_12)
    sample_engines = [1, 5, 10, 15, 20]
    sample_df = train_df[train_df["engine_id"].isin(sample_engines)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sensors_to_plot = [("s_2", "LPC Outlet Temp (s_2)"),
                       ("s_4", "LPT Outlet Temp (s_4)"),
                       ("s_11", "HPC Static Pressure (s_11)"),
                       ("s_12", "Fuel Flow Ratio (s_12)")]
    for ax, (s_col, s_title) in zip(axes.flat, sensors_to_plot):
        for eid in sample_engines:
            eng_data = sample_df[sample_df["engine_id"] == eid]
            ax.plot(eng_data["cycle"], eng_data[s_col], label=f"Eng {eid}", alpha=0.85)
        ax.set_title(s_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Operating Cycle")
        ax.set_ylabel("Measurement")
        ax.legend(fontsize=8, loc="best")
    plt.suptitle("2. Degradation Trajectories Across 5 Engines", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "eda_2_sensor_trends_5_engines.png"), dpi=200)
    plt.close(fig)
    
    # 3. Correlation heatmap of selected sensors + RUL
    corr_cols = SELECTED_SENSORS + (["rul"] if "rul" in train_df.columns else [])
    corr_matrix = train_df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, square=True)
    ax.set_title("3. Correlation Heatmap of Informative Sensors & Degradation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "eda_3_correlation_heatmap.png"), dpi=200)
    plt.close(fig)
    
    # 4. Sensor variance analysis (Distinguishing informative vs. constant sensors)
    all_sensors = [f"s_{i}" for i in range(1, 22)]
    variances = train_df[all_sensors].var().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ca02c" if s in SELECTED_SENSORS else "#d62728" for s in variances.index]
    ax.bar(variances.index, variances.values, color=colors)
    ax.set_yscale("log")
    ax.set_title("4. Sensor Variance Analysis (Green = Informative, Red = Constant / Low Info)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sensor Column")
    ax.set_ylabel("Variance (Log Scale)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "eda_4_sensor_variance_analysis.png"), dpi=200)
    plt.close(fig)
    
    # 5. RUL distribution (Raw vs. Capped Target)
    if "raw_rul" in train_df.columns and "rul" in train_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(train_df["raw_rul"], label="Raw Linear RUL", color="gray", linestyle="--", ax=ax)
        sns.kdeplot(train_df["rul"], label="Capped Piecewise RUL (Cap=125)", color="#ff7f0e", fill=True, alpha=0.3, ax=ax)
        ax.set_title("5. RUL Target Distribution: Linear vs. Piecewise Capped", fontsize=13, fontweight="bold")
        ax.set_xlabel("Remaining Useful Life (Cycles)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "eda_5_rul_distribution.png"), dpi=200)
        plt.close(fig)
        
    # 6. Early-life vs. Late-life sensor behaviour comparison
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    comparison_sensors = [("s_2", "LPC Outlet Temp (s_2)"),
                          ("s_4", "LPT Outlet Temp (s_4)"),
                          ("s_11", "HPC Static Pressure (s_11)")]
    
    # Define early-life as first 30 cycles, late-life as within 30 cycles of failure
    early_data = train_df[train_df["cycle"] <= 30]
    late_data = train_df[train_df["raw_rul"] <= 30] if "raw_rul" in train_df.columns else train_df[train_df["cycle"] > 150]
    
    for ax, (s_col, s_name) in zip(axes, comparison_sensors):
        sns.kdeplot(early_data[s_col], label="Early Life (Healthy)", color="green", fill=True, alpha=0.3, ax=ax)
        sns.kdeplot(late_data[s_col], label="Late Life (Degraded)", color="red", fill=True, alpha=0.3, ax=ax)
        ax.set_title(s_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Sensor Value")
        ax.legend(fontsize=8)
    plt.suptitle("6. Comparison of Early-Life vs. Late-Life Sensor Signatures", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "eda_6_early_vs_late_life_comparison.png"), dpi=200)
    plt.close(fig)
    
    print(f"All 6 EDA figures successfully generated and saved to {save_dir}")
