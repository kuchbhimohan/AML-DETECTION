# EDA summary — IBM AML HI-Small

## Size and imbalance
- Transactions: 5,078,345
- Laundering transactions: 5,177 (0.1019%)
- Ratio normal:laundering ≈ 980:1
- Date range: 2022-09-01 00:00:00 → 2022-09-18 16:18:00
- Unique accounts: 515,080
- Unique banks: 30,470
- Currencies: 15 | Payment formats: 7

## Daily volume
- Days: 18; median transactions/day: 207,406
- Sparse tail days (<5% of median volume): 8 days from 2022-09-11, 1,108 txns, laundering rate 59.12%
  → These days are unrepresentative; the time split should be based on the dense period.

## Laundering rate by attribute (overall rate = 0.102%)
- **payment_format** top: ACH 0.746% (lift 7.3), Bitcoin 0.038% (lift 0.4), Cash 0.022% (lift 0.2)
- **payment_currency** top: Saudi Riyal 0.420% (lift 4.1), Euro 0.117% (lift 1.2), US Dollar 0.101% (lift 1.0)
- **hour of day**: highest rate at hour 12 (0.174%), lowest at hour 0 (0.028%)
- **cross_currency**: 0: 0.103% of 5,006,175, 1: 0.000% of 72,170
- **self_transfer**: 0: 0.115% of 4,487,133, 1: 0.002% of 591,212
- **cross_bank**: 0: 0.015% of 691,332, 1: 0.116% of 4,387,013
- **amount (US Dollar)** median: normal 901.48 vs laundering 5,970.08

## Accounts
- Accounts that sent ≥1 laundering txn: 3,376; received ≥1: 3,984
- Median unique recipients — laundering senders: 3 vs others: 1
- Median unique senders — laundering receivers: 4 vs others: 2

## Patterns file check
- Pattern transactions listed: 3,209
  - GATHER-SCATTER: 716
  - SCATTER-GATHER: 626
  - STACK: 466
  - FAN-OUT: 342
  - FAN-IN: 318
  - CYCLE: 287
  - BIPARTITE: 263
  - RANDOM: 191
- Laundering transactions matched to a pattern: 62.0%
  → OK: per-pattern recall analysis is possible in Step 7.