#!/usr/bin/env python3
"""Select split groups from NOMAD metadata only; do not fetch raw files here."""
import argparse, hashlib, json, re, requests

URL='https://nomad-lab.eu/prod/v1/api/v1/entries'
DATASET='4r6KEYExRb-lQRUtpCfL7g'

def get_entries():
    params=[('owner','public'),('page_size','500'),('json_query',json.dumps({'datasets.dataset_id':DATASET},separators=(',',':'))),('include','entry_id'),('include','upload_id'),('include','mainfile')]
    cursor=None; rows=[]
    while True:
        q=params.copy()
        if cursor: q.append(('page_after_value',cursor))
        r=requests.get(URL,params=q,timeout=120); r.raise_for_status(); d=r.json()
        rows.extend(d['data']); cursor=d['pagination'].get('next_page_after_value')
        if not cursor: break
    return rows

def parts(row):
    path=row['mainfile'].split('/')
    stage=path[0]
    facet=next((re.search(r'(cu[0-9]+(?:_v[0-9]+)?)',p).group(1) for p in path if re.search(r'(cu[0-9]+(?:_v[0-9]+)?)',p)), 'unknown')
    try:
        idx=path.index('1_results'); run=path[idx+1]
    except (ValueError,IndexError): run='no_run_group'
    return stage,facet,run

def stable_key(stage,facet,run):
    return hashlib.sha256(f'MA-1003-v1|{stage}|{facet}|{run}'.encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--per-facet',type=int,default=8); a=ap.parse_args()
    rows=get_entries(); groups={}
    for row in rows:
        stage,facet,run=parts(row); groups.setdefault((stage,facet,run),[]).append(row)
    stages={
      'basis_fit':['1_initial_data_0_AIMD','2_initial_data_0_MD'],
      'hyperparameter_development':['3_new_datapoints_1','4_new_datapoints_2'],
      'few_shot_support':['5_new_datapoints_3'],
      'fresh_audit':['6_new_datapoints_4']}
    result={'schema':'ma1003-nomad-run-manifest-v1','dataset_id':DATASET,'selection_rule':'within each stage role and facet, order complete mainfile run groups by SHA256(MA-1003-v1|stage|facet|run_id), select first N groups; metadata only; no raw files read','per_facet_groups':a.per_facet,'api_total_entries':len(rows),'roles':{}}
    for role,stage_names in stages.items():
        result['roles'][role]={}
        for stage in stage_names:
            by_facet={}
            for (s,f,r),items in groups.items():
                if s==stage: by_facet.setdefault(f,[]).append((stable_key(s,f,r),r,items))
            result['roles'][role][stage]={}
            for facet,gs in sorted(by_facet.items()):
                for key,run,items in sorted(gs)[:a.per_facet]:
                    result['roles'][role][stage].setdefault(facet,[]).append({'run_id':run,'selection_sha256':key,'entries':[{'entry_id':x['entry_id'],'upload_id':x['upload_id'],'mainfile':x['mainfile']} for x in sorted(items,key=lambda x:x['mainfile'])]})
    with open(a.output,'w') as f: json.dump(result,f,indent=2,sort_keys=True); f.write('\n')
    print('entries',len(rows),'groups',len(groups),'manifest_sha256',hashlib.sha256(open(a.output,'rb').read()).hexdigest())
if __name__=='__main__': main()
