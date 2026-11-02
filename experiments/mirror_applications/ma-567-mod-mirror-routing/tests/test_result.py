from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from source.verify import verify
def test_fresh_compute_and_payload(): assert verify(ROOT)['verified']
