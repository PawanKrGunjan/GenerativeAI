# save_nse_universe.py

import nsepython
import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("DATA")
DATA_DIR.mkdir(exist_ok=True)

print("Fetching and saving NSE stock universe...\n")

# 1. NIFTY 50
def get_nifty50():
    try:
        payload = nsepython.nsefetch("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050")
        data = payload["data"]
        return [
            {
                "symbol": item["symbol"],
                "companyName": item.get("meta", {}).get("companyName", item["symbol"])
            }
            for item in data
        ]
    except Exception as e:
        print(f"NIFTY 50 fetch failed: {e}")
        return []

nifty50 = get_nifty50()
print(f"NIFTY 50: {len(nifty50)} stocks")

# 2. Top 250 by Market Cap (F&O securities)
def get_top_250():
    try:
        payload = nsepython.nsefetch("https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O")
        df = pd.DataFrame(payload["data"])
        df["marketCap"] = pd.to_numeric(df.get("ffmc", 0), errors="coerce").fillna(0)
        df = df.sort_values("marketCap", ascending=False).head(250)
        return [
            {
                "symbol": row["symbol"],
                "companyName": row.get("meta", {}).get("companyName", row["symbol"]),
                "marketCap": float(row.get("ffmc", 0))
            }
            for _, row in df.iterrows()
        ]
    except Exception as e:
        print(f"Top 250 fetch failed: {e}")
        return []

top250 = get_top_250()
print(f"Top 250 F&O: {len(top250)} stocks")

# Save combined universe
with open(DATA_DIR / "nse_universe.json", "w") as f:
    json.dump({
        "nifty50": nifty50,
        "top250": top250,
        "updated": "2025-12-29"
    }, f, indent=2)

print(f"Saved nse_universe.json")

# 3. All NSE Equities from CSV → clean flat list
csv_path = DATA_DIR / "EQUITY_L.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"{csv_path} not found! Download from NSE archives.")

# Read CSV properly
df = pd.read_csv(csv_path, header=None, skiprows=1)  # Skip header if malformed
df.columns = ['SYMBOL', 'COMPANY_NAME', 'SERIES', 'LISTING_DATE', 'PAID_UP_VALUE', 'MARKET_LOT', 'ISIN', 'FACE_VALUE']

# Clean and filter
df = df[df['SERIES'] == 'EQ'].copy()
df['SYMBOL'] = df['SYMBOL'].str.strip()
df['COMPANY_NAME'] = df['COMPANY_NAME'].str.strip()

# Save as FLAT LIST of dicts (critical for vector DB)
all_stocks = [
    {
        "symbol": row['SYMBOL'],
        "companyName": row['COMPANY_NAME']
    }
    for _, row in df.iterrows()
]

with open(DATA_DIR / "all_nse_stocks.json", "w") as f:
    json.dump(all_stocks, f, indent=2)  # ← Direct list, no wrapper!

print(f"Saved all_nse_stocks.json → {len(all_stocks)} stocks (flat list)")

# Sample output
print("\nSample entries:")
print(json.dumps(all_stocks[:5], indent=2))