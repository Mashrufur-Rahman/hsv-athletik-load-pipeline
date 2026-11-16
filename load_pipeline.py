"""
Squad workload monitoring pipeline (GPS + sRPE).

Steps:  1) EXTRACT  - generate simulated data OR read a real CSV export
        2) TRANSFORM - clean, impute, engineer load metrics (ACWR, 7d sums, z-scores)
        3) LOAD/REPORT - save processed CSV and a 2-page PDF briefing

Usage:
    python load_pipeline.py                      # simulated demo data
    python load_pipeline.py --input my_data.csv  # real data (see README for columns)
"""
import argparse
import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

REQUIRED_COLUMNS = [
    "date", "player_id", "position", "session_type", "duration_min",
    "total_distance_m", "hsr_distance_m", "sprint_count", "acc_dec_count", "session_rpe",
]
SIM_BANNER = "SIMULATED DATA - DEMONSTRATION ONLY"


# ---------------------------------------------------------
# 1. EXTRACT
# ---------------------------------------------------------
def generate_synthetic_data(seed=42, end_date="2026-11-15", days=42):
    """Simulate 6 weeks of GPS + sRPE logs for a 22-player squad.

    Load spikes are deliberately planted for 3 players in the final 7 days so
    the flagging logic can be demonstrated. These are NOT real findings.
    """
    rng = np.random.RandomState(seed)  # same stream as np.random.seed(seed)
    dates = pd.date_range(end=end_date, periods=days, freq="D")
    players = [f"Player_{i:02d}" for i in range(1, 23)]
    positions = ["GK"] * 2 + ["CB"] * 4 + ["FB"] * 4 + ["CM"] * 6 + ["WG"] * 4 + ["ST"] * 2
    pos_map = dict(zip(players, positions))
    base = {"GK": 4200, "CB": 8800, "FB": 10400, "CM": 11200, "WG": 10600, "ST": 9600}
    spike_players = {"Player_07", "Player_14", "Player_19"}

    records = []
    for p in players:
        for d in dates:
            dow = d.dayofweek  # Monday=0
            if dow == 1:       # Tuesday: day off
                continue
            if dow == 5:
                intensity, session = rng.uniform(1.15, 1.35), "Match"
            elif dow == 6:
                intensity, session = rng.uniform(0.35, 0.50), "Recovery"
            else:
                intensity, session = rng.uniform(0.70, 1.05), "Training"
            if p in spike_players and d >= dates[-7]:
                intensity *= 1.32  # planted spike

            dist = base[pos_map[p]] * intensity + rng.normal(0, 350)
            hsr = dist * rng.uniform(0.06, 0.11)
            sprints = int(max(0, hsr / 45 + rng.normal(0, 2)))
            acc_dec = int(max(10, dist / 110 + rng.normal(0, 6)))
            rpe = float(np.clip(round(intensity * 6.5 + rng.normal(0, 0.7), 1), 1, 10))
            duration = 90 if session == "Match" else int(rng.randint(55, 85))
            if rng.rand() < 0.03:  # simulated sensor dropout
                dist = np.nan

            records.append({
                "date": d, "player_id": p, "position": pos_map[p],
                "session_type": session, "duration_min": duration,
                "total_distance_m": round(dist, 1) if not np.isnan(dist) else np.nan,
                "hsr_distance_m": round(hsr, 1), "sprint_count": sprints,
                "acc_dec_count": acc_dec, "session_rpe": rpe,
            })
    return pd.DataFrame(records)


def load_real_csv(path):
    """Read a real session-level CSV export and validate its columns."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df[REQUIRED_COLUMNS]


# ---------------------------------------------------------
# 2. TRANSFORM
# ---------------------------------------------------------
def classify_load(acwr):
    if pd.isna(acwr):
        return "Insufficient Data"
    if acwr > 1.35:
        return "High Load Flag"
    if acwr < 0.80:
        return "Low Load"
    return "Within Range"


def transform_workload_metrics(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Impute missing GPS distance with the player's median for that session type
    df["total_distance_m"] = df["total_distance_m"].fillna(
        df.groupby(["player_id", "session_type"])["total_distance_m"].transform("median")
    )

    # Internal load: sRPE (AU) = session RPE x duration (min)
    df["internal_load_au"] = df["session_rpe"] * df["duration_min"]

    # Full daily grid so rest days count as zero load in rolling windows
    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    players = df[["player_id", "position"]].drop_duplicates()
    grid = players.merge(pd.DataFrame({"date": all_dates}), how="cross")
    daily = grid.merge(df, on=["player_id", "position", "date"], how="left")

    fill_cols = ["duration_min", "total_distance_m", "hsr_distance_m",
                 "sprint_count", "acc_dec_count", "internal_load_au"]
    daily[fill_cols] = daily[fill_cols].fillna(0)
    daily = daily.sort_values(["player_id", "date"]).reset_index(drop=True)

    grp = daily.groupby("player_id")
    daily["acute_load_7d"] = grp["internal_load_au"].transform(lambda x: x.rolling(7, min_periods=7).mean())
    daily["chronic_load_28d"] = grp["internal_load_au"].transform(lambda x: x.rolling(28, min_periods=28).mean())
    daily["acwr"] = (daily["acute_load_7d"] / daily["chronic_load_28d"]).round(2)

    for col, name in [("hsr_distance_m", "hsr_7d"), ("total_distance_m", "dist_7d"),
                      ("acc_dec_count", "acc_7d"), ("sprint_count", "sprints_7d")]:
        daily[name] = grp[col].transform(lambda x: x.rolling(7, min_periods=7).sum())

    # Individual z-score of 7-day HSR vs the player's own 28-day baseline
    grp = daily.groupby("player_id")
    mean28 = grp["hsr_7d"].transform(lambda x: x.rolling(28, min_periods=14).mean())
    std28 = grp["hsr_7d"].transform(lambda x: x.rolling(28, min_periods=14).std().replace(0, 1))
    daily["hsr_zscore"] = ((daily["hsr_7d"] - mean28) / std28).round(2)

    daily["load_status"] = daily["acwr"].apply(classify_load)
    return daily


# ---------------------------------------------------------
# 3. REPORT
# ---------------------------------------------------------
def _wrap(text, width):
    return "\n".join(textwrap.fill(line, width, subsequent_indent="  ") for line in text.split("\n"))


def _banner(fig, simulated):
    if simulated:
        fig.text(0.96, 0.945, SIM_BANNER, fontsize=9, fontweight="bold", color="#C0392B", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec="#C0392B"))


def generate_pdf_report(daily_df, output_pdf="Squad_Load_Report.pdf", simulated=True, target_player=None):
    latest = daily_df["date"].max()
    snap = daily_df[daily_df["date"] == latest].copy()
    flagged = snap[snap["load_status"] == "High Load Flag"]
    n_flag = len(flagged)

    if target_player is None:  # default: player with the highest ACWR today
        target_player = snap.sort_values("acwr").iloc[-1]["player_id"]
    t_row = snap[snap["player_id"] == target_player].iloc[0]
    t_pos = t_row["position"]

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"

    with PdfPages(output_pdf) as pdf:
        # ---------------- PAGE 1 ----------------
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")
        fig.text(0.04, 0.94, "SQUAD WORKLOAD MONITORING BRIEFING", fontsize=16, fontweight="bold", color="#0A2540")
        fig.text(0.04, 0.91,
                 f"GPS & sRPE load pipeline | Report date: {latest:%Y-%m-%d} | Squad size: {len(snap)} players",
                 fontsize=9.5, color="#555555")
        _banner(fig, simulated)

        kpis = [
            ("Avg 7-day distance", f"{snap['dist_7d'].mean() / 1000:.1f} km", "#0A2540"),
            ("Avg 7-day high-speed running", f"{snap['hsr_7d'].mean():.0f} m", "#0A2540"),
            ("Squad mean ACWR", f"{snap['acwr'].mean():.2f}", "#1B7F5C"),
            ("Load flags (ACWR > 1.35)", f"{n_flag} player{'s' if n_flag != 1 else ''}", "#C0392B"),
        ]
        for i, (label, val, col) in enumerate(kpis):
            ax = fig.add_axes([0.04 + i * 0.235, 0.81, 0.215, 0.075])
            ax.set_facecolor("#F4F6F9")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#D0D7DE")
            ax.text(0.08, 0.62, label, fontsize=8.5, color="#555555")
            ax.text(0.08, 0.18, val, fontsize=14, fontweight="bold", color=col)

        # Chart 1: ACWR matrix
        ax1 = fig.add_axes([0.06, 0.14, 0.43, 0.60])
        top = max(1.75, snap["acwr"].max() + 0.1)
        ax1.axhspan(0.80, 1.35, color="#E8F8F5", alpha=0.8, label="Reference range (0.80-1.35)")
        ax1.axhspan(1.35, top, color="#FDEDEC", alpha=0.8, label="High-load zone (>1.35)")
        ax1.axhline(1.35, color="#C0392B", linestyle="--", linewidth=1)
        ax1.axhline(0.80, color="#2980B9", linestyle=":", linewidth=1)
        cols = snap["load_status"].map({"High Load Flag": "#C0392B", "Within Range": "#1B7F5C", "Low Load": "#2980B9"})
        ax1.scatter(snap["acute_load_7d"], snap["acwr"], c=cols, s=75, edgecolors="black", linewidth=0.6, zorder=3)
        for _, r in snap.iterrows():
            if r["acwr"] > 1.30 or r["acwr"] < 0.82:
                ax1.annotate(r["player_id"].replace("Player_", "P"),
                             (r["acute_load_7d"] + 6, r["acwr"] + 0.015), fontsize=8, fontweight="bold")
        ax1.set_title("1. Acute load vs. ACWR", fontsize=11, fontweight="bold", loc="left", pad=10)
        ax1.set_xlabel("7-day mean internal load (AU)", fontsize=9)
        ax1.set_ylabel("Acute:chronic workload ratio (ACWR)", fontsize=9)
        ax1.set_ylim(0.65, top)
        ax1.legend(loc="lower right", fontsize=8, frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.4)

        # Chart 2: HSR z-scores
        ax2 = fig.add_axes([0.58, 0.42, 0.38, 0.32])
        top_hsr = snap.sort_values("hsr_zscore").tail(10)
        ax2.barh(top_hsr["player_id"], top_hsr["hsr_zscore"],
                 color=["#C0392B" if z > 1.5 else "#2E86C1" for z in top_hsr["hsr_zscore"]], height=0.65)
        ax2.axvline(1.5, color="#C0392B", linestyle="--", linewidth=1, label="+1.5 SD threshold")
        ax2.set_title("2. High-speed running (>21 km/h) z-score", fontsize=11, fontweight="bold", loc="left", pad=10)
        ax2.set_xlabel("Z-score vs. player's own 28-day baseline", fontsize=9)
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, axis="x", linestyle="--", alpha=0.4)

        # Notes box
        ax_n = fig.add_axes([0.58, 0.10, 0.38, 0.24])
        ax_n.set_facecolor("#FFF9E6")
        ax_n.set_xticks([]); ax_n.set_yticks([])
        for s in ax_n.spines.values():
            s.set_edgecolor("#F1C40F")
        names = ", ".join(flagged["player_id"]) if n_flag else "none"
        data_note = ("Data: simulated. Load spikes were planted for P07, P14, P19 to demonstrate the flagging logic."
                     if simulated else "Data: imported from CSV export; missing distances imputed.")
        notes = (
            "FLAGS FOR STAFF REVIEW\n"
            f"- Players above ACWR 1.35: {names}.\n"
            "- ACWR is a load-monitoring indicator, not a validated injury predictor; interpret with "
            "context (match schedule, wellness, medical input).\n"
            f"- {data_note}"
        )
        ax_n.text(0.03, 0.93, _wrap(notes, 58), fontsize=8.2, va="top", color="#2C3E50", linespacing=1.4)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- PAGE 2 ----------------
        fig2 = plt.figure(figsize=(11.69, 8.27))
        fig2.patch.set_facecolor("white")
        fig2.text(0.04, 0.94, f"PLAYER PROFILE: {target_player} (Position: {t_pos})", fontsize=15,
                  fontweight="bold", color="#0A2540")
        fig2.text(0.04, 0.91, "Rolling workload trajectory and comparison with position group", fontsize=9.5,
                  color="#555555")
        _banner(fig2, simulated)

        hist = daily_df[daily_df["player_id"] == target_player].dropna(subset=["acwr"])
        ax3 = fig2.add_axes([0.07, 0.50, 0.86, 0.34])
        ax3b = ax3.twinx()
        ax3.bar(hist["date"], hist["internal_load_au"], color="#D6EAF8", width=0.65, label="Daily load (AU)")
        ax3.plot(hist["date"], hist["acute_load_7d"], color="#1F618D", linewidth=2, label="7-day acute load")
        ax3.plot(hist["date"], hist["chronic_load_28d"], color="#117A65", linewidth=2, linestyle="--",
                 label="28-day chronic load")
        ax3b.plot(hist["date"], hist["acwr"], color="#C0392B", linewidth=2.2, marker="o", markersize=4)
        ax3b.axhline(1.35, color="#C0392B", linestyle=":", alpha=0.7)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax3.set_title(f"3. {target_player}: rolling workload and ACWR (red line, right axis)", fontsize=11,
                      fontweight="bold", loc="left", pad=8)
        ax3.set_ylabel("Workload (AU)", fontsize=9)
        ax3b.set_ylabel("ACWR", fontsize=9, color="#C0392B")
        ax3.set_ylim(0, hist["internal_load_au"].max() * 1.35)
        ax3.legend(loc="upper left", fontsize=8, ncol=3)
        ax3.grid(True, linestyle="--", alpha=0.4)

        group = snap[snap["position"] == t_pos]
        metrics = ["dist_7d", "hsr_7d", "sprints_7d", "acc_7d"]
        labels = ["7-day distance", "7-day HSR", "7-day sprints", "7-day acc/dec"]
        pct = [(t_row[m] / group[m].mean()) * 100 if group[m].mean() > 0 else 100 for m in metrics]
        ax4 = fig2.add_axes([0.17, 0.12, 0.31, 0.28])
        y = np.arange(len(labels))
        ax4.barh(y, pct, color=["#C0392B" if v > 125 else "#2E86C1" for v in pct], height=0.55)
        ax4.axvline(100, color="#333333", linestyle="--", linewidth=1)
        ax4.set_yticks(y)
        ax4.set_yticklabels(labels, fontsize=9)
        ax4.set_xlim(0, max(200, max(pct) + 20))
        ax4.set_xlabel(f"% of {t_pos} group mean (n={len(group)}); dashed line = 100%", fontsize=9)
        ax4.set_title(f"4. {target_player} vs. {t_pos} group", fontsize=11, fontweight="bold", loc="left", pad=8)
        ax4.grid(True, axis="x", linestyle="--", alpha=0.4)

        ax_t = fig2.add_axes([0.55, 0.12, 0.39, 0.28])
        ax_t.set_facecolor("#F4F6F9")
        ax_t.set_xticks([]); ax_t.set_yticks([])
        for s in ax_t.spines.values():
            s.set_edgecolor("#D0D7DE")
        src = ("simulated session-level GPS and sRPE logs" if simulated else "session-level GPS and sRPE CSV exports")
        tech = (
            "PIPELINE AND METHOD\n\n"
            f"1. Extract: Python reads {src}.\n"
            "2. Clean: missing distances are imputed with the player's median for that session type; "
            "rest days are filled with zero load.\n"
            "3. Features: sRPE load = RPE x duration; rolling 7-day acute and 28-day chronic load "
            "(coupled ACWR); 7-day sums; HSR z-score vs. the player's own baseline.\n"
            "4. Output: processed CSV and this PDF briefing."
        )
        ax_t.text(0.04, 0.93, _wrap(tech, 60), fontsize=8.3, va="top", color="#1C2833", linespacing=1.4)
        pdf.savefig(fig2)
        plt.close(fig2)


def main():
    ap = argparse.ArgumentParser(description="Squad workload monitoring pipeline")
    ap.add_argument("--input", help="Path to a real CSV export (default: simulated data)")
    ap.add_argument("--out-csv", default="squad_load_processed.csv")
    ap.add_argument("--out-pdf", default="Squad_Load_Report.pdf")
    ap.add_argument("--player", help="Player ID for the page-2 profile (default: highest ACWR)")
    args = ap.parse_args()

    simulated = args.input is None
    raw = generate_synthetic_data() if simulated else load_real_csv(args.input)
    processed = transform_workload_metrics(raw)
    processed.to_csv(args.out_csv, index=False)
    generate_pdf_report(processed, args.out_pdf, simulated=simulated, target_player=args.player)
    print(f"Done: wrote '{args.out_csv}' and '{args.out_pdf}' "
          f"({'simulated data' if simulated else 'data from ' + args.input}).")


if __name__ == "__main__":
    main()