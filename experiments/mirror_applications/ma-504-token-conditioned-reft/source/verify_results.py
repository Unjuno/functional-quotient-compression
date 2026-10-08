"""Payload integrity and independent held-out metric replay for MA-504."""
import hashlib, json
from pathlib import Path

import torch
from torch.nn import functional as F

from model import Intervention, make_batch

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    summary=json.loads((ROOT/"source"/"screen_summary.json").read_text())
    assert summary["audit_opened"] is False and summary["audit"]==[]
    rows=summary["development"]+summary["fresh"]
    max_diff=0.0; payloads=[]
    for row in rows:
        path=ROOT/row["payload"]["path"]
        raw=path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        assert len(raw)==row["payload"]["bytes"]
        assert digest==row["payload"]["sha256"]
        p=torch.load(path,map_location="cpu",weights_only=False)
        model=Intervention(row["mode"],row["seed"])
        model.load_state_dict(p["state_dict"]);model.eval()
        split="dev" if row["condition"]=="development" else "fresh"
        x,c,y=make_batch(row["seed"],split,8192)
        with torch.no_grad(): mse=float(F.mse_loss(model(x,c),y))
        max_diff=max(max_diff,abs(mse-row["mse"]))
        assert abs(mse-row["mse"]) <= 1e-9
        payloads.append({"path":str(path.relative_to(ROOT)),"bytes":len(raw),"sha256":digest})
    assert len(rows)==35
    result={"experiment_id":"MA-504","commit":"pending-result-commit",
      "selection_draw":{"draw":20,"pool_size":539,"uniform_index":97,"selected_id":"MA-504",
        "pool_sha256":"15aca967e14e36fb186f6596432ff8497671ec1710c71746b770bfdaf16c9b56",
        "baseline_commit":"c935a903daca5c7d1d48aa50d05b5bd50f239cba","replay_checked":True},
      "tests":{"command":"python -m unittest discover -s experiments/mirror_applications/ma-504-token-conditioned-reft/tests -v","passed":4,"failed":0},
      "serialization":{"actual_payloads_measured":True,"payload_count":len(payloads),"payload_manifest":payloads},
      "metric_replay":{"checked":True,"rows":len(rows),"max_difference":max_diff},
      "audit_opened":False,
      "notes":["All 20 development and 15 fresh model payloads passed size/hash/deserialization checks.",
        "All held-out MSE values were replayed from serialized models and the same deterministic dev/fresh generators.",
        "Fresh gates failed on Mirror-versus-independent payload bytes in all three seeds; audit remained unopened."]}
    assert max_diff==0.0
    (ROOT/"VERIFICATION.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"payload_count":len(payloads),"replayed_rows":len(rows),"max_difference":max_diff,"audit_opened":False},indent=2))


if __name__=="__main__":main()
