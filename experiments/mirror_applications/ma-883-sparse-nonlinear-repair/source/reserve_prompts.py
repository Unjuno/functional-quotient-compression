"""Reserve article-heldout prompt row IDs without computing model outcomes."""
import hashlib,json,re
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
MODELS=('EleutherAI/pythia-70m-deduped','EleutherAI/pythia-160m-deduped','HuggingFaceTB/SmolLM2-135M')
TITLE_RE=re.compile(r'^= ([^=].*?) =$')
MIN_TOKENS,MAX_TOKENS=16,128

def tokenizer(mid):
    p=Path.home()/'.cache/huggingface/hub'/('models--'+mid.replace('/','--'))/'snapshots'
    return AutoTokenizer.from_pretrained(sorted(p.iterdir())[0],local_files_only=True,use_fast=True)

def title_of(text):
    m=TITLE_RE.fullmatch(text.strip()); return m.group(1).strip() if m else None

def main():
    toks={x:tokenizer(x) for x in MODELS}
    excluded_titles=set(); excluded_texts=set()
    for split in ('train','validation','test'):
        for row in load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split=split):
            raw=row['text'].strip(); title=title_of(raw)
            if title is not None: excluded_titles.add(title)
            if len(raw)>=32: excluded_texts.add(raw)
    buckets=[[] for _ in range(4)]; seen=set(); current=None
    ds=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',split='train')
    for row_id,row in enumerate(ds):
        raw=row['text'].strip(); title=title_of(raw)
        if title is not None: current=title; continue
        if current is None or current in excluded_titles or len(raw)<32 or raw in excluded_texts or raw in seen: continue
        enc={mid:toks[mid](raw,add_special_tokens=False,return_offsets_mapping=True) for mid in MODELS}
        ref=enc[MODELS[2]]['offset_mapping']
        if any(enc[mid]['offset_mapping']!=ref for mid in MODELS[:2]): continue
        n=len(enc[MODELS[2]]['input_ids'])
        if not MIN_TOKENS<=n<=MAX_TOKENS: continue
        seen.add(raw); title_hash=hashlib.sha256(current.encode()).hexdigest()
        bucket=int(title_hash[:8],16)%4
        rec={'row':row_id,'title_sha256':title_hash,'text_sha256':hashlib.sha256(raw.encode()).hexdigest(),'tokens':n}
        buckets[bucket].append(rec)
        if all(len(x)>=64 for x in buckets): break
    if min(map(len,buckets))<48: raise RuntimeError(f'insufficient eligible prompts per article bucket: {[len(x) for x in buckets]}')
    reserved={}
    for i,items in enumerate(buckets):
        items.sort(key=lambda x:hashlib.sha256(f"{x['title_sha256']}/{x['row']}".encode()).hexdigest())
        reserved[str(i)]=items[:48]
    # Confirm selected article-title groups and text rows are disjoint by bucket.
    title_sets=[{x['title_sha256'] for x in reserved[str(i)]} for i in range(4)]
    text_sets=[{x['text_sha256'] for x in reserved[str(i)]} for i in range(4)]
    for i in range(4):
        for j in range(i):
            assert not title_sets[i]&title_sets[j] and not text_sets[i]&text_sets[j]
    out={'dataset':'Salesforce/wikitext-103-raw-v1','split':'train','old_dataset_exclusion':'all Wikitext-2 raw-v1 split titles and exact rows','group_rule':'int(SHA256(title)[:8],16)%4','minimum_preflight_candidates_per_bucket':[len(x) for x in buckets], 'reserved_prompts_per_bucket':48,'buckets':reserved}
    (ROOT/'prompt_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'eligible_candidates_per_bucket':out['minimum_preflight_candidates_per_bucket'],'reserved_per_bucket':48,'titles_excluded':len(excluded_titles),'exact_rows_excluded':len(excluded_texts)},indent=2))

if __name__=='__main__': main()
