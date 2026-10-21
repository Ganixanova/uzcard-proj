import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

payments = pd.read_csv('scoring_payments.csv', parse_dates=['due_date', 'paid_date'])
contracts = pd.read_csv('scoring_contracts.csv', parse_dates=['issue_date'])

print("rows:", len(payments))
print("paid_date null share:", payments.paid_date.isna().mean())

paid = payments[payments.paid_date.notna()].copy()
print("\nAmong paid rows:")
print("paid_amount < due_amount (partial payment) share:", (paid.paid_amount_uzs < paid.due_amount_uzs).mean())
print("paid_amount > due_amount (overpaid) share:", (paid.paid_amount_uzs > paid.due_amount_uzs).mean())
diff_ratio = (paid.paid_amount_uzs - paid.due_amount_uzs).abs() / paid.due_amount_uzs
print("of partial/over, max abs relative diff:", diff_ratio.max(), "share with diff>1%:", (diff_ratio>0.01).mean())

paid['dpd'] = (paid.paid_date - paid.due_date).dt.days
print("\ndpd stats (paid rows):")
print(paid.dpd.describe())
print("share paid EARLY (dpd<0):", (paid.dpd<0).mean())
print("min dpd:", paid.dpd.min(), "max dpd:", paid.dpd.max())

print("\nany duplicate (contract_id, installment_no)?", payments.duplicated(subset=['contract_id','installment_no']).sum())

# contracts with zero payment rows
print("\ncontracts with zero payment rows:", (~contracts.contract_id.isin(payments.contract_id)).sum())

# check installment_no continuity / any contract missing early installments
grp = payments.groupby('contract_id').installment_no
gaps = 0
for cid, s in grp:
    s = sorted(s)
    if s != list(range(1, len(s)+1)):
        gaps += 1
print("contracts with non-contiguous installment_no starting at 1:", gaps)

extract_date = pd.Timestamp('2025-05-31')
no_pay = contracts[~contracts.contract_id.isin(payments.contract_id)]
print("\nzero-payment contracts issue_date range:", no_pay.issue_date.min(), no_pay.issue_date.max())
print("months since issue for these:", ((extract_date - no_pay.issue_date).dt.days/30.44).describe())
