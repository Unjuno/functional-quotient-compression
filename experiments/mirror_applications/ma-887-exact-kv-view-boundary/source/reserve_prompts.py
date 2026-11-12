"""Reserve a third title-disjoint Wikitext-103 prompt set without reading model outcomes."""
import hashlib,json,re
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; MODELS=('EleutherAI/pythia-70m-deduped','HuggingFaceTB/SmolLM2-135M'); RX=re.compile(r'^= ([^=].*?) =$')
def tok(mid):
 p=Path.home()/'.cache/huggingface/hub'/('models--'+mid.replace('/','--'))/'snapshots';return AutoTokenizer.from_pretrained(sorted(p.iterdir())[0],local_files_only=True,use_fast=True)
def main():
 ts={m:tok(m) for m in MODELS};root=ROOT.parents[0];exclude_t=set();exclude_x=set()
 for exp in ('ma-883-sparse-nonlinear-repair','ma-886-cross-model-kv-fanout'):
  old=json.loads((root/exp/'prompt_manifest.json').read_text())
  for group in old['buckets'].values():
   for z in group:exclude_t.add(z['title_sha256']);exclude_x.add(z['text_sha256'])
 ex_titles=set();ex_text=set()
 for split in ('train','validation','test'):
  for r in load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split=split):
   x=r['text'].strip();m=RX.fullmatch(x)
   if m:ex_titles.add(m.group(1).strip())
   if len(x)>=32:ex_text.add(x)
 buckets=[[] for _ in range(4)];seen=set();current=None;ds=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',split='train')
 for i,r in enumerate(ds):
  x=r['text'].strip();m=RX.fullmatch(x)
  if m:current=m.group(1).strip();continue
  if not current or current in ex_titles or len(x)<32 or x in ex_text or x in seen:continue
  th=hashlib.sha256(current.encode()).hexdigest();xh=hashlib.sha256(x.encode()).hexdigest()
  if th in exclude_t or xh in exclude_x:continue
  a={z:ts[z](x,add_special_tokens=False,return_offsets_mapping=True) for z in MODELS}
  if a[MODELS[0]]['offset_mapping']!=a[MODELS[1]]['offset_mapping']:continue
  n=len(a[MODELS[0]]['input_ids'])
  if not 16<=n<=128 or n!=len(a[MODELS[1]]['input_ids']):continue
  seen.add(x);buckets[int(th[:8],16)%4].append({'row':i,'title_sha256':th,'text_sha256':xh,'tokens':n})
  if min(map(len,buckets))>=64:break
 if min(map(len,buckets))<48:raise RuntimeError([len(x) for x in buckets])
 res={}
 for j,arr in enumerate(buckets):arr.sort(key=lambda z:hashlib.sha256(f"887/{z['title_sha256']}/{z['row']}".encode()).hexdigest());res[str(j)]=arr[:48]
 out={'dataset':'Salesforce/wikitext-103-raw-v1','split':'train','excluded_experiments':['MA-883','MA-886'],'excluded_old_dataset':'all Wikitext-2 raw-v1 splits','article_rule':'int(SHA256(title)[:8],16)%4','eligible_candidates_per_bucket':[len(z) for z in buckets],'prompts_per_bucket':48,'buckets':res}
 (ROOT/'prompt_manifest.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'eligible':out['eligible_candidates_per_bucket'],'reserved':48,'previous_hashes_excluded':len(exclude_t)},indent=2))
if __name__=='__main__':main()
