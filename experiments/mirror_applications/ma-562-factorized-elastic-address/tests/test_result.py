from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from source.verify import verify

def test_fresh_bytes_and_control_alias():
    x=verify(ROOT); assert x['fresh_worlds']==3 and x['rows']==48 and x['direct_alias_exact']
