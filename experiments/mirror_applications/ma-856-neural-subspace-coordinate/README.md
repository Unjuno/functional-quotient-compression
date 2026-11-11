# MA-856 — PA231 learned neural line coordinate

PA231 Algorithm 1 trains independently initialized endpoints by sampling alpha uniformly for each minibatch, evaluating the interpolated network, and updating both endpoints under cross-entropy plus beta times squared cosine similarity. Default beta=1; beta=0 is required as a diversity ablation.

Use sklearn Digits scaled by 1/16 with a stratified fixed 60/20/20 train/development/test split. Model: MLP 64→32→10, ReLU, no BatchNorm/dropout (~2,410 parameters). Tune only learning rate and beta on development. Fresh training replicas use seeds 61,79,97 on the heldout test partition; they are not unseen-task generalization.

Evaluate alpha=0,1/16,…,1 and native alpha vs Mirror m=2alpha−1 on the same trained checkpoint. Record worst/mean accuracy and NLL, endpoint disagreement, probability distance, ensemble benefit, endpoint+code bytes, 17 materialized-view bytes, independent-model banks, reconstruction MACs and query compute. The Mirror coordinate is expected to be an exact rename of PA231 interpolation; this experiment maps its quality/storage utility without presuming incremental value.
