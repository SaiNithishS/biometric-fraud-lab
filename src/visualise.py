import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

os.makedirs("reports", exist_ok=True)
sns.set_theme(style="whitegrid")

def load_flagged():
    return pd.read_csv("data/processed/flagged_events.csv", parse_dates=["timestamp"])

def plot_match_score_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df.loc[~df["is_fraud"], "match_score"], bins=50, stat="density",
                 label="Legitimate", color="steelblue", alpha=0.6, ax=ax)
    sns.histplot(df.loc[df["is_fraud"], "match_score"], bins=50, stat="density",
                 label="Fraud", color="crimson", alpha=0.6, ax=ax)
    ax.axvline(0.55, color="orange", linestyle="--", label="Low-score rule (0.55)")
    ax.set_title("Match Score Distribution: Legitimate vs Fraud")
    ax.set_xlabel("Biometric match score")
    ax.legend()
    fig.tight_layout()
    fig.savefig("reports/match_score_dist.png", dpi=150)
    plt.close(fig)

def plot_fraud_by_pattern(df):
    fraud = df[df["is_fraud"]]
    counts = fraud.groupby(["fraud_pattern", "alert"]).size().unstack(fill_value=0)
    counts = counts.rename(columns={0: "Missed", 1: "Detected"})
    counts = counts.reindex(columns=["Detected", "Missed"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", stacked=True, color=["crimson", "lightgrey"], ax=ax)
    ax.legend(title=None)
    ax.set_title("Fraud Events by Pattern: Detected vs Missed")
    ax.set_xlabel("Fraud pattern")
    ax.set_ylabel("Events")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig("reports/fraud_by_pattern.png", dpi=150)
    plt.close(fig)

def plot_alerts_over_time(df):
    alerts = df[df["alert"] == 1].copy()
    alerts["date"] = alerts["timestamp"].dt.date
    alerts["type"] = alerts["is_fraud"].map({True: "True positive", False: "False positive"})
    daily = alerts.groupby(["date", "type"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    daily.plot(kind="bar", stacked=True, ax=ax,
               color={"True positive": "darkorange", "False positive": "grey"})
    ax.legend(title=None)
    ax.set_title("Daily Alert Volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Alerts")
    ax.set_xticklabels([str(d) for d in daily.index], rotation=60, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig("reports/alerts_over_time.png", dpi=150)
    plt.close(fig)

def plot_anomaly_heatmap(df):
    pivot = df.pivot_table(values="anomaly_raw", index="hour",
                           columns="modality", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap="RdYlGn", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Mean Isolation Forest Score by Hour and Modality (red = more anomalous)")
    fig.tight_layout()
    fig.savefig("reports/anomaly_heatmap.png", dpi=150)
    plt.close(fig)

def plot_confusion_matrix(df):
    cm = confusion_matrix(df["is_fraud"].astype(int), df["alert"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No alert", "Alert"], yticklabels=["Legitimate", "Fraud"])
    ax.set_title("Detection Outcome")
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    df = load_flagged()
    plot_match_score_distribution(df)
    plot_fraud_by_pattern(df)
    plot_alerts_over_time(df)
    plot_anomaly_heatmap(df)
    plot_confusion_matrix(df)
    print("Charts saved:", sorted(f for f in os.listdir("reports") if f.endswith(".png")))
