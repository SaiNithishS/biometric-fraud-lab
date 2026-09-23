import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

MODALITIES = ["fingerprint", "face", "voice", "keystroke_dynamics"]
DEVICES = [f"DEV-{i:04d}" for i in range(1, 51)]
HOME_CITIES = ["London", "Manchester", "New York"]
FRAUD_CITIES = ["Lagos", "Singapore", "Berlin"]

def generate_normal_event(user_id, timestamp, location):
    return {
        "event_id": fake.uuid4(),
        "user_id": user_id,
        "timestamp": timestamp,
        "modality": random.choice(MODALITIES),
        "match_score": round(float(np.clip(np.random.normal(0.82, 0.06), 0, 1)), 4),
        "device_id": random.choice(DEVICES),
        "location": location,
        "attempt_count": 1,
        "outcome": "success",
        "session_duration_sec": max(10, int(np.random.normal(180, 40))),
    }

def inject_fraud_patterns(events, fraud_rate=0.06):
    fraud_events = []
    for _ in range(int(len(events) * fraud_rate)):
        base = random.choice(events).copy()
        pattern = random.choice(["low_score_bypass", "geo_jump", "rapid_retry", "off_hours"])
        base["event_id"] = fake.uuid4()
        base["is_fraud"] = True
        base["fraud_pattern"] = pattern

        if pattern == "low_score_bypass":
            base["match_score"] = round(np.random.uniform(0.3, 0.55), 4)
        elif pattern == "geo_jump":
            base["location"] = random.choice(FRAUD_CITIES)
            base["timestamp"] = base["timestamp"] + timedelta(minutes=random.randint(2, 15))
        elif pattern == "rapid_retry":
            base["attempt_count"] = random.randint(8, 20)
            base["match_score"] = round(np.random.uniform(0.4, 0.65), 4)
        elif pattern == "off_hours":
            base["timestamp"] = base["timestamp"].replace(
                hour=random.randint(2, 4), minute=random.randint(0, 59))
        fraud_events.append(base)
    return fraud_events

def build_dataset(n_users=200, days=30):
    start = datetime(2024, 1, 1)
    events = []
    for uid in range(1, n_users + 1):
        user_id = f"USR-{uid:04d}"
        # Each user has a home city and takes 0-3 whole-day trips elsewhere
        home = random.choice(HOME_CITIES)
        away = random.choice([c for c in HOME_CITIES if c != home])
        travel_days = set(random.sample(range(days), k=random.randint(0, 3)))
        for _ in range(random.randint(10, 60)):
            day = random.randint(0, days - 1)
            ts = start + timedelta(days=day,
                                   hours=random.randint(7, 21),
                                   minutes=random.randint(0, 59))
            location = away if day in travel_days else home
            e = generate_normal_event(user_id, ts, location)
            e["is_fraud"] = False
            e["fraud_pattern"] = None
            events.append(e)
    events.extend(inject_fraud_patterns(events))
    return pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)

if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("data/raw/biometric_auth_logs.csv", index=False)
    print(f"Generated {len(df)} events, {df['is_fraud'].sum()} fraud cases.")
