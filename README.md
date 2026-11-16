# Squad Workload Monitoring Pipeline (GPS + sRPE)

A small Python ETL pipeline that turns session-level GPS and sRPE data into
load-monitoring metrics and a 2-page PDF briefing for staff.

> **Data note:** the demo runs on **simulated data** because no club data is
> available. Load spikes are deliberately planted for three players
> (Player_07, 14, 19) to demonstrate the flagging logic. Nothing in the demo
> report is a real finding. The same code runs on a real CSV export
> (see "Using real data").

## What it does
1. **Extract**: generates simulated 6-week squad data (22 players, weekly
   microcycle with match, recovery and rest days, 3% sensor dropouts) or reads a CSV.
2. **Transform**
   - imputes missing distances with the player's median for that session type
   - sRPE internal load = session RPE x duration (AU)
   - full daily grid so rest days count as zero load
   - rolling 7-day acute load, 28-day chronic load and ACWR (coupled)
   - 7-day sums: distance, high-speed running (>21 km/h), sprints, accelerations/decelerations
   - HSR z-score versus the player's own 28-day baseline
3. **Report**: writes `squad_load_processed.csv` and `Squad_Load_Report.pdf`
   (squad overview + individual player profile compared with the position group).

## Run it
```bash
pip install -r requirements.txt
python load_pipeline.py                      # simulated demo
python load_pipeline.py --player Player_07   # choose the profile player
```

## Using real data
```bash
python load_pipeline.py --input my_export.csv
```
Required columns (one row per player per session):

| column | meaning |
|---|---|
| date | session date (YYYY-MM-DD) |
| player_id, position | identifiers |
| session_type | Match / Training / Recovery |
| duration_min | session duration |
| total_distance_m, hsr_distance_m | GPS distance and high-speed running distance |
| sprint_count, acc_dec_count | counts |
| session_rpe | 1-10 rating of perceived exertion |

## Limitations
- ACWR is a load-monitoring indicator, not a validated injury predictor; the
  1.35 / 0.80 thresholds are common conventions, not guarantees.
- Coupled ACWR (acute window included in chronic window) is used for simplicity.
- Needs at least 28 days of data before ACWR is available.
- Simulated data cannot show real-world relationships between load and outcomes.

## Possible next steps
- Add real open tracking data (e.g. Metrica Sports sample data) to compute speed zones from raw positions
- Interactive Streamlit dashboard
- Unit tests for the transform step