"""Sanity checks for MA-1059's predeclared experiment and frozen development split."""
import json
from pathlib import Path
PROTOCOL=Path(__file__).resolve().parents[1]/'PROTOCOL.json'
def validate(path=PROTOCOL):
    p=json.loads(Path(path).read_text())
    required={'experiment_id','prior_art_refs','hypothesis','controls','success_gate','failure_gate','fresh'}
    missing=required-p.keys()
    if missing: raise ValueError(f'missing protocol fields: {sorted(missing)}')
    if p['experiment_id']!='MA-1059' or p['prior_art_refs']!=['PA328','PA330','PA335']:
        raise ValueError('MA ID/prior-art mismatch')
    if p['fresh']['locked_before_access'] is not True: raise ValueError('audit split not frozen')
    groups=p['data']['series_ids']; fit=set(groups['fit']); dev=set(groups['development']); audit=set(groups['audit'])
    if fit&dev or fit&audit or dev&audit: raise ValueError('overlapping series identity split')
    if (fit|dev|audit)&set(p['data']['exposed_excluded']): raise ValueError('previewed series reused')
    return p
if __name__=='__main__':
    validate(); print('MA-1059 protocol and disjoint ID split valid; fresh audit remains unopened.')
