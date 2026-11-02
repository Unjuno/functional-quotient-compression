from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from source.verify import verify
def test_fresh_payload_and_direct_alias(): assert verify(ROOT)['verified']
