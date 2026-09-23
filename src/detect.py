import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

CITY_COORDS = {
    "London": (51.5074, -0.1278), "Manchester": (53.4808, -2.2426),
    "New York": (40.7128, -74.0060), "Lagos": (6.5244, 3.3792),
    "Singapore": (1.3521, 103.8198), "Berlin": (52.5200, 13.4050),
}
MAX_TRAVEL_SPEED_KMH = 900  # roughly commercial flight cruising speed

IF_FEATURES = [
    "match_score", "attempt_count", "session_duration_sec", "hour",
    "is_off_hours", "score_zscore", "impossible_travel",
    "log_mins_since_prev", "log_travel_speed",
]

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def load_and_prepare(path="data/raw/biometric_auth_logs.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_off_hours"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

    # Per-user behavioural baseline for match score
    g = df.groupby("user_id")["match_score"]
    df["user_score_mean"] = g.transform("mean")
    df["user_score_std"] = g.transform("std").fillna(0.01).replace(0, 0.01)
    df["score_zscore"] = (df["match_score"] - df["user_score_mean"]) / df["user_score_std"]

    # Impossible travel: speed needed to get from the previous login's location
    df["lat"] = df["location"].map({c: v[0] for c, v in CITY_COORDS.items()})
    df["lon"] = df["location"].map({c: v[1] for c, v in CITY_COORDS.items()})
    df["prev_location"] = df.groupby("user_id")["location"].shift()
    df["prev_lat"] = df.groupby("user_id")["lat"].shift()
    df["prev_lon"] = df.groupby("user_id")["lon"].shift()
    df["mins_since_prev"] = (
        df.groupby("user_id")["timestamp"].diff().dt.total_seconds() / 60
    )
    df["travel_km"] = haversine_km(df["prev_lat"], df["prev_lon"], df["lat"], df["lon"])
    hours = (df["mins_since_prev"] / 60).clip(lower=1 / 60)  # avoid divide-by-zero
    df["travel_speed_kmh"] = np.where(df["travel_km"] > 0, df["travel_km"] / hours, 0)
    df["impossible_travel"] = (df["travel_speed_kmh"] > MAX_TRAVEL_SPEED_KMH).astype(int)
    df["log_travel_speed"] = np.log1p(df["travel_speed_kmh"])
    df["log_mins_since_prev"] = np.log1p(df["mins_since_prev"].fillna(60 * 24 * 30))
    return df

def run_isolation_forest(df, contamination=0.06):
    X = df[IF_FEATURES].fillna(0)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    df["anomaly_raw"] = model.fit(X).decision_function(X)   # lower = more anomalous
    df["if_flagged"] = (model.predict(X) == -1).astype(int)
    return df, model

def apply_rule_based_flags(df):
    df["rule_low_score"] = (df["match_score"] < 0.55).astype(int)
    df["rule_rapid_retry"] = (df["attempt_count"] >= 8).astype(int)
    df["rule_off_hours"] = df["is_off_hours"]
    df["rule_score_zscore"] = (df["score_zscore"].abs() > 2.5).astype(int)
    df["other_rule_count"] = df[[
        "rule_low_score", "rule_rapid_retry", "rule_off_hours", "rule_score_zscore"
    ]].sum(axis=1)

    # Correlation: a jump straight after another impossible-travel login is the
    # "return leg" of the same incident, not a new one
    prev_it = df.groupby("user_id")["impossible_travel"].shift().fillna(0)
    df["return_leg"] = ((df["impossible_travel"] == 1) & (prev_it == 1)).astype(int)
    df["rule_impossible_travel"] = (
        (df["impossible_travel"] == 1) & (df["return_leg"] == 0)
    ).astype(int)

    df["rule_flag_count"] = df["other_rule_count"] + df["rule_impossible_travel"]
    # Impossible travel is high-severity on its own; others need corroboration
    df["rules_flagged"] = (
        (df["rule_flag_count"] >= 2) | (df["rule_impossible_travel"] == 1)
    ).astype(int)
    return df

def combined_flag(df):
    # Suppress return legs only when nothing else about the login is suspicious
    df["suppressed"] = ((df["return_leg"] == 1) & (df["other_rule_count"] == 0)).astype(int)
    df["alert"] = (
        ((df["if_flagged"] == 1) | (df["rules_flagged"] == 1)) & (df["suppressed"] == 0)
    ).astype(int)
    df["detected_by"] = np.select(
        [(df["alert"] == 1) & (df["if_flagged"] == 1) & (df["rules_flagged"] == 1),
         (df["alert"] == 1) & (df["if_flagged"] == 1),
         (df["alert"] == 1) & (df["rules_flagged"] == 1)],
        ["both", "isolation_forest", "rules"], default="none")
    return df

def evaluate(df):
    y, a = df["is_fraud"].astype(bool), df["alert"].astype(bool)
    tp, fp = int((a & y).sum()), int((a & ~y).sum())
    fn, tn = int((~a & y).sum()), int((~a & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    print(f"Alerts: {int(a.sum())} of {len(df)} events")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Precision={precision:.1%}  Recall={recall:.1%}")

    s = df["suppressed"].astype(bool)
    print(f"\nReturn legs linked to an existing incident (suppressed): {int(s.sum())}"
          f"  (of which fraud: {int((s & y).sum())})")

    print("\nRecall by fraud pattern:")
    fraud = df[y]
    print(fraud.groupby("fraud_pattern")["alert"].mean().map("{:.1%}".format))

    print("\nWhich detector caught the true positives:")
    print(df[a & y]["detected_by"].value_counts())

if __name__ == "__main__":
    df = load_and_prepare()
    df, _ = run_isolation_forest(df)
    df = apply_rule_based_flags(df)
    df = combined_flag(df)
    df.to_csv("data/processed/flagged_events.csv", index=False)
    evaluate(df)
