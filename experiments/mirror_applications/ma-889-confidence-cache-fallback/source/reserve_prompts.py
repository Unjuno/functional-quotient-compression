"""Reserve article-heldout prompts unseen in MA-883/886/887; no model-quality metrics."""
import hashlib,json,re
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
MODELS=('EleutherAI/pythia-70m-deduped','EleutherAI/pythia-160m-deduped','HuggingFaceTB/SmolLM2-135M')
TITLE_RE=re.compile(r'^= ([^=].*?) =$'); MIN_TOKENS=16; MAX_TOKENS=128

def tokenizer(mid):
 p=Path.home()/'.cache/huggingface/hub'/('models--'+mid.replace('/','--'))/'snapshots'
 return AutoTokenizer.from_pretrained(sorted(p.iterdir())[0],local_files_only=True,use_fast=True)
def main():
 toks={mid:tokenizer(mid) for mid in MODELS}; base=Path(__file__).resolve().parents[2]
 old_titles=set(); old_texts=set()
 for exp in ('ma-883-sparse-nonlinear-repair','ma-886-cross-model-kv-fanout','ma-887-exact-kv-view-boundary'):
  old=json.loads((base/exp/'prompt_manifest.json').read_text())
  for group in old['buckets'].values():
   for x in group: old_titles.add(x['title_sha256']); old_texts.add(x['text_sha256'])
 excluded_titles=set(); excluded_texts=set()
 for split in ('train','validation','test'):
  for row in load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split=split):
   raw=row['text'].strip(); mt=TITLE_RE.fullmatch(raw)
   if mt: excluded_titles.add(mt.group(1).strip())
   if len(raw)>=32: excluded_texts.add(raw)
 buckets=[[] for _ in range(4)]; seen=set(); current=None
 ds=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',split='train')
 for row_id,row in enumerate(ds):
  raw=row['text'].strip(); mt=TITLE_RE.fullmatch(raw)
  if mt: current=mt.group(1).strip(); continue
  if current is None or current in excluded_titles or len(raw)<32 or raw in excluded_texts or raw in seen: continue
  th=hashlib.sha256(current.encode()).hexdigest(); xh=hashlib.sha256(raw.encode()).hexdigest()
  if th in old_titles or xh in old_texts: continue
  enc={mid:toks[mid](raw,add_special_tokens=False,return_offsets_mapping=True) for mid in MODELS}
  offsets=enc[MODELS[0]]['offset_mapping']
  if any(enc[mid]['offset_mapping']!=offsets for mid in MODELS[1:]): continue
  n=len(enc[MODELS[0]]['input_ids'])
  if not MIN_TOKENS<=n<=MAX_TOKENS: continue
  seen.add(raw); bi=int(th[:8],16)%4
  buckets[bi].append({'row':row_id,'title_sha256':th,'text_sha256':xh,'tokens':n})
  if all(len(v)>=64 for v in buckets): break
 if min(map(len,buckets))<48: raise RuntimeError(f'insufficient candidates: {[len(v) for v in buckets]}')
 reserved={}
 for i,items in enumerate(buckets):
  items.sort(key=lambda x:hashlib.sha256(f"MA889/{x['title_sha256']}/{x['row']}".encode()).hexdigest()); reserved[str(i)]=items[:48]
 for i in range(4):
  for j in range(i): assert {x['title_sha256'] for x in reserved[str(i)]}.isdisjoint(x['title_sha256'] for x in reserved[str(j)])
 out={'dataset':'Salesforce/wikitext-103-raw-v1','split':'train','previous_experiment_exclusion':'all MA-883, MA-886, and MA-887 selected title and text hashes','old_dataset_exclusion':'all Wikitext-2 raw-v1 split titles and exact rows','group_rule':'int(SHA256(title)[:8],16)%4','eligible_candidates_per_bucket':[len(v) for v in buckets],'reserved_prompts_per_bucket':48,'buckets':reserved}
 (ROOT/'prompt_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'eligible_candidates_per_bucket':out['eligible_candidates_per_bucket'],'reserved_per_bucket':48,'previous_experiment_titles_excluded':len(old_titles),'titles_excluded':len(excluded_titles),'exact_rows_excluded':len(excluded_texts)},indent=2))
if __name__=='__main__': main()
