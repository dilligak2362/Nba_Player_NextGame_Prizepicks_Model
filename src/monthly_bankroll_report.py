import pandas as pd
from pathlib import Path

BET_LOG = Path("data/bankroll/bet_log.csv")

df = pd.read_csv(BET_LOG)
df["date"] = pd.to_datetime(df["date"])

monthly = df.groupby(df["date"].dt.to_period("M")).agg(
    bets=("stake", "count"),
    total_staked=("stake", "sum"),
    profit=("pnl", "sum")
).reset_index()

monthly["roi"] = monthly["profit"] / monthly["total_staked"]

print("\n📊 Monthly Performance\n")
print(monthly)

monthly.to_csv("data/bankroll/monthly_report.csv", index=False)