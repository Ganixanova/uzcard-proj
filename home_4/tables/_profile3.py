import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

def sec(t):
    print("\n" + "="*100)
    print(t)

apps = pd.read_csv('scoring_applications.csv', parse_dates=['decision_date'])
apps_full = pd.read_csv('scoring_applications_full.csv', parse_dates=['decision_date'])
customers = pd.read_csv('scoring_customers.csv')
clients_old = pd.read_csv('scoring_clients_old.csv')
cards = pd.read_csv('scoring_cards.csv')
snaps = pd.read_csv('scoring_bureau_snapshots.csv', parse_dates=['snapshot_date'])
txn = pd.read_csv('scoring_card_transactions.csv')

sec("A) exact overlap counts between id spaces")
apps_ids = set(apps.client_id)
cust_ids = set(customers.customer_id)
old_ids = set(clients_old.client_id)
txn_ids = set(txn.client_id)
snap_ids = set(snaps.client_id)
print("apps.client_id total unique:", len(apps_ids), "max:", apps.client_id.max())
print("overlap apps & customers:", len(apps_ids & cust_ids))
print("overlap apps & clients_old:", len(apps_ids & old_ids))
print("overlap apps & card_transactions:", len(apps_ids & txn_ids))
print("overlap apps & bureau_snapshots:", len(apps_ids & snap_ids))
print("customers & bureau_snapshots overlap:", len(cust_ids & snap_ids))
print("customers & card_transactions overlap:", len(cust_ids & txn_ids))

sec("B) is apps.client_id range actually disjoint-ish from customers id range or just numeric coincidence?")
print("apps client_id describe:\n", apps.client_id.describe())
print("how many apps.client_id <= 8000 (customers max):", (apps.client_id <= 8000).sum())
sub = apps[apps.client_id <= 8000]
print("of those, how many ARE in customers ids:", sub.client_id.isin(cust_ids).sum(), "/", len(sub))

sec("C) applications_full.bureau_score_latest vs as-of(decision_date) snapshot score - is it leaking future?")
merged = apps_full[['application_id','client_id','decision_date','bureau_score_latest']].merge(
    snaps, on='client_id', how='left')
asof = merged[merged.snapshot_date <= merged.decision_date].sort_values('snapshot_date').groupby('application_id').tail(1)
asof = asof[['application_id','bureau_score','snapshot_date']].rename(columns={'bureau_score':'asof_score'})
cmp = apps_full[['application_id','bureau_score_latest']].merge(asof, on='application_id', how='left')
print("rows:", len(cmp), "have as-of snapshot:", cmp.asof_score.notna().sum())
print("match rate (latest == as-of decision):", (cmp.bureau_score_latest == cmp.asof_score).mean())
future_snap = merged[merged.snapshot_date > merged.decision_date]
latest_overall = merged.sort_values('snapshot_date').groupby('application_id').tail(1)[['application_id','bureau_score','snapshot_date']]
cmp2 = apps_full[['application_id','bureau_score_latest','decision_date']].merge(latest_overall, on='application_id', how='left')
print("match rate (latest == truly latest snapshot overall incl future):", (cmp2.bureau_score_latest == cmp2.bureau_score).mean())

sec("D) cards.customer_id vs customers full coverage")
print("cards.customer_id nunique:", cards.customer_id.nunique(), "customers n:", len(customers))
print("every customer has >=1 card?", set(customers.customer_id).issubset(set(cards.customer_id)))

sec("E) employment_months anomalies")
print("negative values:", (apps.employment_months < 0).sum())
print("value counts of extreme (>=240):", (apps.employment_months >= 240).sum())
print(apps.employment_months.value_counts().sort_index().tail(10))
