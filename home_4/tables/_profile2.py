import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

def sec(t):
    print("\n" + "="*100)
    print(t)

apps = pd.read_csv('scoring_applications.csv', parse_dates=['decision_date'])
apps_full = pd.read_csv('scoring_applications_full.csv', parse_dates=['decision_date'])
contracts = pd.read_csv('scoring_contracts.csv', parse_dates=['issue_date'])
customers = pd.read_csv('scoring_customers.csv', parse_dates=['birth_date','onboarding_date'])
clients_old = pd.read_csv('scoring_clients_old.csv', parse_dates=['birthdate'])
cards = pd.read_csv('scoring_cards.csv')
payments = pd.read_csv('scoring_payments.csv', parse_dates=['due_date','paid_date'])
payments_bak = pd.read_csv('scoring_payments_backup.csv', parse_dates=['due_date','paid_date'])
snaps = pd.read_csv('scoring_bureau_snapshots.csv', parse_dates=['snapshot_date'])
bureau_loans = pd.read_csv('scoring_bureau_loans.csv', parse_dates=['open_date','planned_close_date','close_date'])
inquiries = pd.read_csv('scoring_bureau_inquiries.csv', parse_dates=['inquiry_date'])
model_v1 = pd.read_csv('scoring_model_v1_scores.csv')
collections = pd.read_csv('scoring_collection_actions.csv', parse_dates=['action_date'])
txn = pd.read_csv('scoring_card_transactions.csv', parse_dates=['txn_ts'])
txn_sample = pd.read_csv('scoring_txn_sample.csv', parse_dates=['txn_ts'])
disputes = pd.read_csv('scoring_card_disputes.csv', parse_dates=['opened_date'])

sec("1) applications: final_decision counts")
print(apps['final_decision'].value_counts(dropna=False))
print("decision_date range:", apps.decision_date.min(), apps.decision_date.max())

sec("2) applications_full vs contracts: application_id sets equal?")
print("apps_full ids == contracts.application_id ids:",
      set(apps_full.application_id) == set(contracts.application_id))
print("apps_full n:", apps_full.application_id.nunique(), "contracts n:", contracts.application_id.nunique())
print("apps_full ids subset of approved apps:",
      set(apps_full.application_id).issubset(set(apps[apps.final_decision=='approved'].application_id)))

sec("3) applications_full extra columns look post-decision?")
print(apps_full.columns.tolist())
print(apps_full[['bureau_score_latest','dpd_max','collection_calls_cnt','contract_status_now']].describe(include='all'))

sec("4) customers vs clients_old: same person universe?")
print("customers n:", len(customers), "clients_old n:", len(clients_old))
merged = customers.merge(clients_old, left_on='customer_id', right_on='client_id', suffixes=('_new','_old'))
print("matched by id:", len(merged))
print("name match rate:", (merged.full_name_new == merged.full_name_old).mean())
print("id ranges -> customers:", customers.customer_id.min(), customers.customer_id.max(),
      " clients_old:", clients_old.client_id.min(), clients_old.client_id.max())
print("customers cols:", customers.columns.tolist())
print("clients_old cols:", clients_old.columns.tolist())

sec("5) client_id universe in applications vs customers/clients_old")
print("apps client_id range:", apps.client_id.min(), apps.client_id.max(), "nunique:", apps.client_id.nunique())
print("client_id subset of customers.customer_id:", set(apps.client_id).issubset(set(customers.customer_id)))
print("client_id subset of clients_old.client_id:", set(apps.client_id).issubset(set(clients_old.client_id)))

sec("6) cards.customer_id vs customers/clients_old universe")
print("cards.customer_id subset of customers:", set(cards.customer_id).issubset(set(customers.customer_id)))
print("cards.customer_id subset of clients_old:", set(cards.customer_id).issubset(set(clients_old.client_id)))

sec("7) payments vs payments_backup: identical rows?")
print("payments shape:", payments.shape, "backup shape:", payments_bak.shape)
common_ids = set(payments.payment_id) & set(payments_bak.payment_id)
print("common payment_id count:", len(common_ids))
cmp = payments.set_index('payment_id').loc[sorted(common_ids)].sort_index()
cmp2 = payments_bak.set_index('payment_id').loc[sorted(common_ids)].sort_index()
print("identical on common ids:", cmp.equals(cmp2))
print("max payment_id payments:", payments.payment_id.max(), "backup:", payments_bak.payment_id.max())
print("payments due_date range:", payments.due_date.min(), payments.due_date.max())
print("backup due_date range:", payments_bak.due_date.min(), payments_bak.due_date.max())

sec("8) open contracts: observation window not closed as of 2025-05-31")
extract_date = pd.Timestamp('2025-05-31')
last_due = payments.groupby('contract_id').due_date.max().rename('last_due_date')
c = contracts.merge(last_due, on='contract_id', how='left')
c['months_since_issue'] = (extract_date - c.issue_date).dt.days/30.44
print("contracts with last_due_date > extract_date (future installments still owed):",
      (c.last_due_date > extract_date).sum(), "/", len(c))
print("contracts issued within last 12m (window likely open):", (c.months_since_issue < 12).sum(), "/", len(c))

sec("9) payments: unpaid / missing paid_date as of extract date, only for due_date already passed")
p = payments.copy()
p['is_due_passed'] = p.due_date <= extract_date
p['is_unpaid'] = p.paid_date.isna()
print(pd.crosstab(p.is_due_passed, p.is_unpaid))

sec("10) bureau_snapshots: per-client snapshot count & date range vs applications decision_date")
snap_stats = snaps.groupby('client_id').snapshot_date.agg(['min','max','count'])
print(snap_stats.describe())
merged_snap = apps.merge(snap_stats, left_on='client_id', right_index=True, how='left')
print("apps with NO bureau snapshot at all:", merged_snap['count'].isna().sum())
has_prior_snap = apps.merge(snaps, on='client_id').query('snapshot_date <= decision_date')
print("apps with >=1 snapshot before/at decision_date:", has_prior_snap.application_id.nunique(), "/", apps.application_id.nunique())

sec("11) bureau_loans: open_date relative to decision_date (future loans opened after our decision)")
bl = bureau_loans.merge(apps[['client_id','decision_date']], on='client_id')
print("bureau loans opened AFTER our decision_date (must exclude for as-of features):",
      (bl.open_date > bl.decision_date).sum(), "/", len(bl))

sec("12) model_v1_scores vs approved_by_old_policy correlation (selection bias signal)")
mv = apps.merge(model_v1, on='application_id')
print(mv.groupby('approved_by_old_policy').score_v1.describe())
print(mv.groupby('final_decision').score_v1.describe())

sec("13) card_transactions client_id universe vs applications client_id universe")
print("txn client_id nunique:", txn.client_id.nunique())
print("overlap with apps.client_id:", len(set(txn.client_id) & set(apps.client_id)), "/", apps.client_id.nunique())
print("txn_type values:", txn.txn_type.value_counts())
print("txn_ts range:", txn.txn_ts.min(), txn.txn_ts.max())

sec("14) txn_sample vs card_transactions: overlapping keys?")
print("txn_sample cols:", txn_sample.columns.tolist())
print("card_id overlap:", len(set(txn_sample.card_id) & set(txn.card_id)))
print("txn_sample has client_id?", 'client_id' in txn_sample.columns)
print("txn_sample txn_ts range:", txn_sample.txn_ts.min(), txn_sample.txn_ts.max())
print("txn.txn_id range:", txn.txn_id.min(), txn.txn_id.max())
print("txn_sample.transaction_id range:", txn_sample.transaction_id.min(), txn_sample.transaction_id.max())

sec("15) collection_actions contract_id linkage")
print("contract_id subset of contracts:", set(collections.contract_id).issubset(set(contracts.contract_id)))
print("collections action_date vs contract issue_date -> always after issue?")
cc = collections.merge(contracts[['contract_id','issue_date']], on='contract_id')
print((cc.action_date >= cc.issue_date).mean())

sec("16) nulls in declared_income / employment_months / other anketa fields")
print(apps[['requested_amount_uzs','declared_income_uzs','employment_months']].isna().mean())
print(apps.employment_months.describe())
print("employment_months==0 share:", (apps.employment_months==0).mean())

sec("17) declined applications - do they have contracts/payments (should NOT)?")
declined_ids = set(apps[apps.final_decision!='approved'].application_id)
print("declined with contract anyway:", len(declined_ids & set(contracts.application_id)))
