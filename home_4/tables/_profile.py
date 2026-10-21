import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

files = sorted([f for f in os.listdir('.') if f.endswith('.csv') and not f.startswith('_')])
for f in files:
    print("="*100)
    print(f)
    try:
        df = pd.read_csv(f, nrows=5)
        with open(f, encoding='utf-8', errors='ignore') as fh:
            n_full = sum(1 for _ in fh) - 1
        print(f"rows~={n_full}, cols={len(df.columns)}")
        print(list(df.columns))
        print(df.head(3).to_string())
    except Exception as e:
        print("ERROR:", e)
