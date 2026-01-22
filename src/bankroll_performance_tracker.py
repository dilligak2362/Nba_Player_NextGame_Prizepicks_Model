import pandas as pd
from pathlib import Path

BET_LOG = Path("data/bankroll/bet_log.csv")

df = pd.read_csv(BET_LOG)

# Only settle pending bets
pending = df[df["result"] == "PENDING"].copy()

print(f"Pending bets: {len(pending)}")

# Manual input for now
for i, row in pending.iterrows():
    print("\n------------------------")
    print("Players:", row["players"])
    print("Props:", row["props"])
    print("Directions:", row["directions"])
    res = input("Result (W/L): ").strip().upper()

    if res not in ["W", "L"]:
        continue

    stake = row["stake"]
    payout = stake * row["payout_mult"]

    if res == "W":
        pnl = payout - stake
    else:
        pnl = -stake

    df.loc[i, "result"] = res
    df.loc[i, "pnl"] = pnl

# Rebuild bankroll
bankroll = None
for i, row in df.iterrows():
    if bankroll is None:
        bankroll = row["bankroll_before"]
    bankroll += row["pnl"]
    df.loc[i, "bankroll_after"] = bankroll

df.to_csv(BET_LOG, index=False)

print("\nUpdated bankroll:", bankroll)
print("Saved log.")