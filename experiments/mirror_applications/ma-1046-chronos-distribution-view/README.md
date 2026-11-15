# MA-1046 — Chronos frequency-conditioned distribution View

Status: SCREENING. The frozen protocol separates ETTh1/ETTm1 development from Chronos-paper zero-shot Weather and Australian Electricity audit data. Development metrics have been inspected and the code is frozen in runs/DEV_FREEZE.json; fresh audit remains unscored.

The experiment shares one official Chronos-T5-Tiny checkpoint across series and tests a two-coordinate frequency-conditioned Givens rotation over forecast-sample shape. It compares frozen zero-shot output, native location/temperature calibration, a normalized Hermite residual control, and seasonal naive. See PROTOCOL.json before data access.
