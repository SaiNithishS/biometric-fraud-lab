import os
from datetime import datetime
import pandas as pd
from fpdf import FPDF
from fpdf.fonts import FontFace

DATA = "data/processed/flagged_events.csv"
OUT = "reports/biometric_fraud_report.pdf"


class FraudReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 8, "Biometric Fraud Pattern Analysis Report", align="C",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, f"Generated {datetime.now():%Y-%m-%d %H:%M} | CONFIDENTIAL | "
                        "Synthetic lab data", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(30, 30, 60)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def para(self, body):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, body, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def figure(self, path, caption, width=170):
        if not os.path.exists(path):
            self.para(f"[Missing chart: {path}]")
            return
        self.image(path, x=(self.w - width) / 2, w=width)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def simple_table(self, headers, rows, col_widths=None):
        self.set_font("Helvetica", "", 9)
        self.set_fill_color(255, 255, 255)
        self.set_text_color(0, 0, 0)
        with self.table(col_widths=col_widths, text_align="CENTER",
                        headings_style=FontFace(emphasis="BOLD", fill_color=(220, 222, 235)),
                        cell_fill_color=(245, 246, 250), cell_fill_mode="ROWS") as table:
            r = table.row()
            for h in headers:
                r.cell(h)
            for row in rows:
                r = table.row()
                for v in row:
                    r.cell(str(v))
        self.ln(3)


def compute(df):
    y, a = df["is_fraud"].astype(bool), df["alert"].astype(bool)
    m = {
        "total": len(df), "alerts": int(a.sum()), "fraud": int(y.sum()),
        "tp": int((a & y).sum()), "fp": int((a & ~y).sum()), "fn": int((~a & y).sum()),
        "suppressed": int(df["suppressed"].sum()) if "suppressed" in df else 0,
        "start": df["timestamp"].min().date(), "end": df["timestamp"].max().date(),
    }
    m["precision"] = m["tp"] / (m["tp"] + m["fp"]) if m["tp"] + m["fp"] else 0
    m["recall"] = m["tp"] / (m["tp"] + m["fn"]) if m["tp"] + m["fn"] else 0
    m["days"] = (m["end"] - m["start"]).days + 1
    m["alerts_per_day"] = m["alerts"] / m["days"]

    # Per-pattern detection
    pat = df[y].groupby("fraud_pattern").agg(events=("alert", "size"),
                                             detected=("alert", "sum"))
    pat["recall"] = pat["detected"] / pat["events"]
    m["patterns"] = pat.sort_values("events", ascending=False)
    m["weakest"] = pat["recall"].idxmin()

    # Match score evidence for the threshold recommendation
    legit, fraud = df.loc[~y, "match_score"], df.loc[y, "match_score"]
    m["legit_median"], m["fraud_median"] = legit.median(), fraud.median()
    m["fraud_below_055"] = (fraud < 0.55).mean()
    m["legit_below_065"] = (legit < 0.65).mean()
    m["fraud_below_065"] = (fraud < 0.65).mean()

    # Temporal
    daily = df[a].groupby(df["timestamp"].dt.date).size()
    m["peak_day"], m["peak_count"] = daily.idxmax(), int(daily.max())

    # Accounts to investigate (uses only what an analyst would see - no labels)
    m["top_users"] = (
        df[a].groupby("user_id")
        .agg(alerts=("alert", "sum"), mean_score=("match_score", "mean"),
             locations=("location", lambda s: ", ".join(sorted(s.unique()))))
        .sort_values("alerts", ascending=False).head(10)
    )
    return m


def generate_report():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    m = compute(df)
    pdf = FraudReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.section("1. Executive Summary")
    pdf.para(
        f"Analysis period: {m['start']} to {m['end']} ({m['days']} days).\n"
        f"Authentication events analysed: {m['total']:,}. Fraud events in dataset: {m['fraud']:,}.\n"
        f"Alerts raised: {m['alerts']:,} (about {m['alerts_per_day']:.0f} per day).\n"
        f"Detected {m['tp']} of {m['fraud']} fraud events (recall {m['recall']:.1%}) "
        f"with {m['fp']} false positives (precision {m['precision']:.1%}).\n"
        f"{m['suppressed']} duplicate impossible-travel alerts were correlated into existing "
        f"incidents instead of raising new alerts.\n\n"
        f"The weakest-detected pattern was '{m['weakest']}' "
        f"({m['patterns'].loc[m['weakest'], 'recall']:.1%} recall). "
        f"The {m['fn']} missed events should be reviewed to improve coverage."
    )

    pdf.section("2. Detection Performance")
    pdf.para(
        "Detection combines rule-based checks (low match score, rapid retries, off-hours "
        "access, per-user score deviation, speed-based impossible travel) with an "
        "Isolation Forest anomaly model. Rules provide explainable reasons for each alert; "
        "the model catches unusual combinations the rules do not encode."
    )
    pdf.figure("reports/confusion_matrix.png", "Figure 1: Detection outcome", width=90)

    pdf.section("3. Fraud Pattern Breakdown")
    rows = [(p, int(r.events), int(r.detected), f"{r.recall:.1%}")
            for p, r in m["patterns"].iterrows()]
    pdf.simple_table(["Pattern", "Events", "Detected", "Recall"], rows)
    pdf.figure("reports/fraud_by_pattern.png", "Figure 2: Fraud events by pattern")

    pdf.section("4. Match Score Analysis")
    pdf.para(
        f"Median match score: legitimate {m['legit_median']:.2f}, fraud {m['fraud_median']:.2f}. "
        f"{m['fraud_below_055']:.1%} of fraud events scored below the 0.55 rule threshold. "
        f"Fraud that passes with a normal-looking score (e.g. off-hours or geo-jump sessions) "
        f"is caught by behavioural and travel signals, not by score alone."
    )
    pdf.figure("reports/match_score_dist.png", "Figure 3: Match score distribution")

    pdf.section("5. Temporal Analysis")
    pdf.para(
        f"Alert volume averaged {m['alerts_per_day']:.1f} per day, peaking at "
        f"{m['peak_count']} alerts on {m['peak_day']}."
    )
    pdf.figure("reports/alerts_over_time.png", "Figure 4: Daily alert volume")
    pdf.figure("reports/anomaly_heatmap.png",
               "Figure 5: Mean anomaly score by hour and modality (red = more anomalous)")

    pdf.section("6. Accounts Recommended for Investigation")
    rows = [(u, int(r.alerts), f"{r.mean_score:.2f}", r.locations)
            for u, r in m["top_users"].iterrows()]
    pdf.simple_table(["User", "Alerts", "Mean score", "Locations seen"], rows,
                     col_widths=(25, 15, 20, 60))

    pdf.section("7. Recommendations")
    pdf.para(
        f"1. Require step-up verification for match scores below 0.65. This would affect "
        f"{m['legit_below_065']:.1%} of legitimate logins and {m['fraud_below_065']:.1%} "
        f"of fraudulent ones.\n"
        "2. Enforce step-up authentication or temporary lockout when attempt count exceeds 5.\n"
        "3. Keep speed-based impossible-travel detection (>900 km/h) with return-leg "
        "correlation; add a known-VPN/proxy allowlist before production use.\n"
        "4. Treat off-hours access (23:00-06:00) as a corroborating signal and require MFA "
        "rather than blocking outright.\n"
        "5. Retrain the Isolation Forest monthly and tune its sensitivity against analyst "
        "feedback on closed alerts."
    )

    pdf.section("8. Methodology and Limitations")
    pdf.para(
        "All data is synthetic, generated with known fraud patterns so detection quality can be "
        "measured. Real traffic is noisier: legitimate users travel, use VPNs and log in at "
        "unusual hours. The anomaly model's sensitivity (6%) is close to the dataset's true "
        "fraud rate, which a production system would not know in advance. Tuning history is "
        "recorded in reports/tuning_log.md. In this dataset, hours 02-04 and 22 contain only "
        "fraud events, so the red rows in Figure 5 reflect how the data was generated, "
        "not a real-world finding."
    )

    pdf.output(OUT)
    print(f"Report saved to {OUT}")


if __name__ == "__main__":
    generate_report()
