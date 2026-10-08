"""Build the checked result table and concise development summary."""
import csv,hashlib,json,statistics
from pathlib import Path
from model import FRAMES,HEIGHT,WIDTH,TOKENS,TOKEN_DIM
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'source'

def decoder_mac_proxy():
 h=TOKEN_DIM;d=TOKEN_DIM//3;n=TOKENS;b=256
 return int(20*h+h*h+2*n*h*h/b+2*n*h+h*h+h*48+48*3)

raw=json.loads((SOURCE/'development_raw.json').read_text());rows=raw['rows']
for x in rows:
 factors=x['method'] in ('additive','mirror','mirror_private')
 x['inference_macs_proxy']=FRAMES*HEIGHT*WIDTH*decoder_mac_proxy()+TOKENS*x['generator_macs_per_token']
 x['training_macs_proxy']=(800*256*decoder_mac_proxy()*3+(600*4*TOKENS*x['rank']*TOKEN_DIM*3 if factors else 0)+
     600*256*(decoder_mac_proxy()+TOKENS*x['generator_macs_per_token'])*3)
(SOURCE/'development_raw.json').write_text(json.dumps(raw,indent=2)+'\n')
fields=['base_seed','clip','method','rank','library_k','library_payload_bytes','library_payload_sha256','payload_artifact','single_video_payload_bytes','marginal_video_code_bytes','heldout_mse','heldout_psnr_db','heldout_ssim','eval_coordinates','query_pixels_per_second','decoder_pretrain_examples','decoder_pretrain_updates','training_token_examples_generator','training_updates_generator','training_examples_code_fit','optimizer_updates','total_training_examples_including_shared','total_optimizer_updates_including_shared','training_macs_proxy','inference_macs_proxy','decoder_macs_per_query_proxy','generator_macs_per_token','generator_elementwise_multiplies_per_token','active_attention_logits_bytes','peak_process_rss_kib','decoder_pretrain_wall_s','shared_generator_fit_wall_s','train_code_wall_s','token_generation_wall_s','decoder_eval_wall_s','heldout_split_indices_sha256','status_note']
with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for x in rows:
  z={k:x.get(k,'') for k in fields};z['status_note']='development-only, frozen decoder/token-code screen; fresh clips unopened'
  for k in ('heldout_mse','heldout_psnr_db','heldout_ssim','query_pixels_per_second','decoder_pretrain_wall_s','shared_generator_fit_wall_s','train_code_wall_s','token_generation_wall_s','decoder_eval_wall_s'):
   if z[k]!='':z[k]=f"{z[k]:.10g}"
  w.writerow(z)
(ROOT/'RESULTS_CORE.csv').write_bytes((ROOT/'RESULTS_CORE.csv').read_bytes().replace(b'\r\n',b'\n'))
summary=json.loads((SOURCE/'development_summary.json').read_text())
summary['decision']='FAIL-development-gate; fresh clips remain unopened'
summary['selected_rank']=None;summary['fresh_accessed']=False
summary['row_count']=len(rows)
summary['metric_scope']='held-out 20% pixel coordinates from the two development clips; full query decoder run over all coordinates'
summary['fact']='All ranks passed the full-library-byte, marginal-video-code-byte, and query-throughput clauses. No product-Mirror rank passed the complete gate on both clips and both seeds: ranks 2/4/8 miss the native CoANeRV full-token PSNR tolerance on container. Rank 8 passes the matched-simple-control clause on all four seed/clip conditions and is within 1 dB on both carphone conditions, but misses PSNR by 1.15-1.35 dB on container.'
summary['interpretation']='The product factor code lowers marginal per-video token state by about 70% and the K=2 serialized library by 10-14%, with similar CPU query throughput. Quality loss is content-dependent; container retains a clear native full-token advantage. This is a FAIL under the predeclared quality conjunction.'
summary['hypothesis']='Video/time/region product Views may encode regular clips efficiently, but heterogeneous spatial-temporal content needs a private residual or more direct token state. Rank-2 private residual improves mean PSNR at added state, but it was a diagnostic control and was not authorized to replace the product-Mirror development gate or trigger fresh.'
summary['counter_hypothesis']='The apparent product-Mirror advantage over additive factors could reflect an underfit reduced decoder and two development clips; the full native token bank still wins on the more complex container sequence. The real feed-forward CoANeRV tokenizer was not tested.'
summary['uncertainties']=['fresh video replication','full CoANeRV feed-forward encoder/token former','standard coded-video bitstream RD','high-resolution GPU memory and decoding FPS','near-convergence fixed-byte capacity','larger and more diverse clip families']
summary['raw_sha256']=hashlib.sha256((SOURCE/'development_raw.json').read_bytes()).hexdigest()
summary['results_core_sha256']=hashlib.sha256((ROOT/'RESULTS_CORE.csv').read_bytes()).hexdigest()
summary['payloads']={}
for x in rows:
 p=ROOT/x['payload_artifact'];blob=p.read_bytes()
 summary['payloads'][x['payload_artifact']]={'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}
summary['total_code_fit_wall_s']=sum(x['train_code_wall_s'] for x in rows)
summary['total_decoder_pretrain_wall_s_by_seed']=sum(next(x['decoder_pretrain_wall_s'] for x in rows if x['base_seed']==s) for s in sorted(set(x['base_seed'] for x in rows)))
(SOURCE/'development_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'decision':summary['decision'],'raw_sha256':summary['raw_sha256'],'results_core_sha256':summary['results_core_sha256'],'payload_files':len(summary['payloads'])},indent=2))
