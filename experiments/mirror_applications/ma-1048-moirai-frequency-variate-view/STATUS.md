# MA-1048 status

- Status: SCREENING; protocol and development selection frozen.
- Branch: research/ma-1048-moirai-frequency-variate-view-20261115
- Official Moirai-1.1-R-small CPU inference works on the pinned 55.3 MB checkpoint.
- New source: PEMS-BAY, 325 sensors, 52,116 five-minute rows; first 14 finite sensor columns were predeclared.
- Fit and validation forecast samples exist for 16 seen frequency×variate pairs. Validation selected seasonal naive; the final-20% audit on 12 held-out pairs remains unopened.
- No Moirai weights were updated. Fresh CRPS/WQL/MASE, serialization, and runtime are pending.
