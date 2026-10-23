"""MA-1176 descriptive, predeclared-gate analysis. Fresh MUST stay sealed until eval."""
from __future__ import annotations
import csv, json, hashlib, statistics
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def median(values):
    values=list(values)
    return statistics.median(values) if values else None

def pr(x):
    return '—' if x is None else f'{x:.5f}'

def analyze():
    src=ROOT/'results'/'RESULTS_CORE.csv'
    if not src.exists() or not (ROOT/'results'/'EVAL_COMPLETE.json').exists():
        raise RuntimeError('Cannot analyze before checkpoint hash seal and fresh evaluation')
    rows=list(csv.DictReader(src.open('r',encoding='utf-8')))
    for r in rows:
        for key in ('seed','width','train_size','steps','init','loop_steps','parameter_count','packet_bytes','unique_train','examples_seen','forward_MAC_per_example','train_forward_MAC_proxy','extra_vector_ops_per_example'):
            r[key]=int(r[key])
        for key in ('train_nmse','dev_nmse','fresh_nmse','ood_nmse','training_seconds','latency_batch1_us','latency_batch32_us'):
            r[key]=float(r[key])
        for k in range(3):
            r[f'fresh_rule{k}_nmse']=float(r[f'fresh_rule{k}_nmse'])
        r['lanes']=r['lanes'].split('|')
        r['distill']=r['distill']=='True'
    assert len(rows)==189
    lanes=defaultdict(list)
    for r in rows:
        for l in r['lanes']:lanes[l].append(r)
    def find(lane,**kwargs):
        v=[r for r in lanes[lane] if all(r[k]==val for k,val in kwargs.items())]
        if len(v)!=1:raise AssertionError((lane,kwargs,len(v)))
        return v[0]
    worlds=['aligned_mirror_teacher','independent_mlp_teacher']
    seeds=[17,29,43]
    # Representation controlled comparisons, paired by identical teacher and train seed.
    rep={}
    for world in worlds:
        rep[world]={}
        for typ,w in [('shared',8),('film',8),('mirror',8),('independent',4)]:
            r=[find('representation',world=world,seed=s,model=typ,width=w) for s in seeds]
            rep[world][typ]={
                'fresh_median':median(x['fresh_nmse'] for x in r),
                'ood_median':median(x['ood_nmse'] for x in r),
                'params':r[0]['parameter_count'],
                'packet_bytes':r[0]['packet_bytes'],
                'MAC_per_sample':r[0]['forward_MAC_per_example'],
                'latency_b1_us':median(x['latency_batch1_us'] for x in r),
                'training_seconds_median':median(x['training_seconds'] for x in r)}
    mirror_pairs=[]
    for w in worlds:
        for s in seeds:
            m=find('representation',world=w,seed=s,model='mirror')
            f=find('representation',world=w,seed=s,model='film')
            mirror_pairs.append({'world':w,'seed':s,'mirror_nmse':m['fresh_nmse'],'film_nmse':f['fresh_nmse'],'relative_gain':1-m['fresh_nmse']/f['fresh_nmse'],'pass_5pct':m['fresh_nmse']<=0.95*f['fresh_nmse']})
    # Physical parameter width: separate budget from effective functions.
    parameter=[]
    for w in worlds:
        for typ in ('film','mirror'):
            for width in (4,8,12):
                v=[find('parameters',world=w,seed=s,model=typ,width=width) for s in seeds]
                parameter.append(dict(world=w,model=typ,width=width,params=v[0]['parameter_count'],packet_bytes=v[0]['packet_bytes'],MAC_per_sample=v[0]['forward_MAC_per_example'],fresh_median=median(r['fresh_nmse'] for r in v),ood_median=median(r['ood_nmse'] for r in v)))
    # Unique data quantity; optimizer updates held constant.
    unique=[]
    for w in worlds:
        for typ in ('film','mirror'):
            for n in (96,384,1536):
                v=[find('unique_data',world=w,seed=s,model=typ,train_size=n) for s in seeds]
                unique.append(dict(world=w,model=typ,unique_examples=n,train_examples_seen=v[0]['examples_seen'],fresh_median=median(r['fresh_nmse'] for r in v),ood_median=median(r['ood_nmse'] for r in v)))
    # Data quality: identical number of nominal samples, different distribution/noise.
    quality=[]
    quality_pairs=[]
    for w in worlds:
        for typ in ('film','mirror'):
            for regime in ('clean_balanced','skew','corrupted','oracle_filtered'):
                v=[find('data_quality',world=w,seed=s,model=typ,regime=regime) for s in seeds]
                quality.append(dict(world=w,model=typ,regime=regime,unique_examples_median=median(r['unique_train'] for r in v),fresh_median=median(r['fresh_nmse'] for r in v),ood_median=median(r['ood_nmse'] for r in v),rare_rule1_median=median(r['fresh_rule1_nmse'] for r in v),rare_rule2_median=median(r['fresh_rule2_nmse'] for r in v)))
            for s in seeds:
                c=find('data_quality',world=w,seed=s,model=typ,regime='corrupted')
                clean=find('data_quality',world=w,seed=s,model=typ,regime='oracle_filtered')
                quality_pairs.append(dict(world=w,seed=s,model=typ,unfiltered_nmse=c['fresh_nmse'],oracle_nmse=clean['fresh_nmse'],gain=1-clean['fresh_nmse']/c['fresh_nmse'],oracle_beats=clean['fresh_nmse']<c['fresh_nmse']))
    # Loop: distinct teacher family. T=1/2/3 equal learnable parameters, not equal compute.
    loop_rows=[]
    for s in seeds:
        a={}
        for t in (1,2,3):a[f'T{t}']=find('loop',world='loop_teacher',seed=s,model='loop',loop_steps=t,distill=False)
        a['onepass']=find('loop',world='loop_teacher',seed=s,model='onepass')
        a['distill1']=find('loop_distillation',world='loop_teacher',seed=s,model='loop',distill=True)
        a['seed']=s
        loop_rows.append(a)
    loops={label:dict(fresh_median=median(r[label]['fresh_nmse'] for r in loop_rows),ood_median=median(r[label]['ood_nmse'] for r in loop_rows),params=loop_rows[0][label]['parameter_count'],MAC_per_sample=loop_rows[0][label]['forward_MAC_per_example'],packet_bytes=loop_rows[0][label]['packet_bytes'],train_s_median=median(r[label]['training_seconds'] for r in loop_rows)) for label in ('T1','T2','T3','onepass','distill1')}
    loop_quality_pairs=[dict(seed=r['seed'],T1=r['T1']['fresh_nmse'],T3=r['T3']['fresh_nmse'],distill1=r['distill1']['fresh_nmse'],benefit=r['T3']['fresh_nmse']<=0.9*r['T1']['fresh_nmse'],distill_pass=(r['distill1']['fresh_nmse']<=1.05*r['T3']['fresh_nmse'] and r['distill1']['forward_MAC_per_example']<=0.5*r['T3']['forward_MAC_per_example'])) for r in loop_rows]
    # Search: dev picks, full search work charged, same examples/updates baseline.
    search=[]
    search_aggregates=[]
    for w in worlds:
        for s in seeds:
            cc=sorted([r for r in lanes['search_candidate'] if r['world']==w and r['seed']==s],key=lambda r:r['init'])
            assert len(cc)==8
            for budget in (1,4,8):
                candidate_set=cc[:budget]
                chosen=min(candidate_set,key=lambda x:x['dev_nmse'])
                control=find('search_no_search_baseline',world=w,seed=s,model='film',steps=120*budget)
                search.append(dict(world=w,seed=s,budget=budget,selected_arch=chosen['model'],selected_case=chosen['case_id'],selected_dev=chosen['dev_nmse'],selected_fresh=chosen['fresh_nmse'],baseline_fresh=control['fresh_nmse'],selected_packet_bytes=chosen['packet_bytes'],control_packet_bytes=control['packet_bytes'],trial_updates=sum(c['steps'] for c in candidate_set),baseline_updates=control['steps'],all_trials_s=sum(c['training_seconds'] for c in candidate_set),baseline_s=control['training_seconds'],all_trials_MAC_proxy=sum(c['train_forward_MAC_proxy'] for c in candidate_set),baseline_MAC_proxy=control['train_forward_MAC_proxy'],gain_over_equal_update_baseline=1-chosen['fresh_nmse']/control['fresh_nmse'],passes_5pct=chosen['fresh_nmse']<=0.95*control['fresh_nmse']))
    for budget in (1,4,8):
        v=[x for x in search if x['budget']==budget]
        search_aggregates.append(dict(budget=budget,selected_median=median(r['selected_fresh'] for r in v),baseline_median=median(r['baseline_fresh'] for r in v),search_trial_seconds_median=median(r['all_trials_s'] for r in v),baseline_seconds_median=median(r['baseline_s'] for r in v),selected_arch_counter=dict(Counter(r['selected_arch'] for r in v)),gains_5pct=sum(r['passes_5pct'] for r in v),pairs=len(v)))
    gates={
        'mirror_specific_5pct_pairs':sum(x['pass_5pct'] for x in mirror_pairs),
        'mirror_specific_gate':sum(x['pass_5pct'] for x in mirror_pairs)>=5,
        'loop_T3_gain_pairs':sum(x['benefit'] for x in loop_quality_pairs),
        'loop_gain_gate':sum(x['benefit'] for x in loop_quality_pairs)>=2,
        'loop_onepass_distill_pairs':sum(x['distill_pass'] for x in loop_quality_pairs),
        'loop_deployment_gate':sum(x['distill_pass'] for x in loop_quality_pairs)>=2,
        'data_quality_oracle_beats_noisy_mirror_pairs':sum(x['oracle_beats'] for x in quality_pairs if x['model']=='mirror'),
        'data_quality_oracle_beats_noisy_film_pairs':sum(x['oracle_beats'] for x in quality_pairs if x['model']=='film'),
        'data_quality_mirror_gate':sum(x['oracle_beats'] for x in quality_pairs if x['model']=='mirror')>=5,
        'data_quality_film_gate':sum(x['oracle_beats'] for x in quality_pairs if x['model']=='film')>=5,
        'search4_gain_pairs':sum(x['passes_5pct'] for x in search if x['budget']==4),
        'search_gate':sum(x['passes_5pct'] for x in search if x['budget']==4)>=4,
    }
    result={
        'id':'MA-1176','rows':len(rows),'source':'RESULTS_CORE.csv','representation':rep,
        'mirror_pairs':mirror_pairs,'parameter_axis':parameter,'unique_data_axis':unique,
        'quality_axis':quality,'quality_pairs':quality_pairs,'loops':loops,'loop_pairs':loop_quality_pairs,
        'search':search,'search_aggregates':search_aggregates,'gates':gates,
        'warning':'Tiny synthetic 4D regression, 3 student seeds. Quality comparisons are descriptive; exact matched time/FLOPs and causal LM performance not established.'
    }
    (ROOT/'results'/'ANALYSIS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return result


def report(r):
    out=[]
    out.append('# MA-1176 — 5つのスケーリング軸の極小実験・結果\n')
    out.append('**位置付け:** 4次元・3規則の合成回帰、各世界3 seedの小規模スクリーニング。Attention・自然言語学習・最適アーキテクチャの実証ではない。事前登録の合否条件を変更していない。\n')
    out.append('## 1. 何を独立にしたか\n')
    out.append('| 軸 | 独立変数 | 固定条件 | 目的 |\n|---|---|---|---|')
    out.extend(['| Loop | 同一32係数で反復1/2/3回 | 入力分布・係数数・320更新 | 計算の深さと推論蒸留 |',
                '| Mirror | 同幅FiLMと同数の88学習係数 | データ・240更新 | 共有核の機能変換の効用 |',
                '| 物理係数 | 幅4/8/12・学習係数44/88/132 | データ・240更新 | 実係数のスケーリング |',
                '| データ | 異なるユニーク例96/384/1536と汚染/偏り | 同一240更新×32例 | 情報量・質の効果 |',
                '| 探索 | 試行モデル1/4/8 | 合計候補更新数を対照へ投資 | 開発選択対追加学習 |\n'])
    out.append('## 2. Mirrorによる共有関数生成 vs 同予算制御\n')
    out.append('| 教師 | 方式 | 物理係数 | 記録B | Fresh NMSE↓ | OOD NMSE↓ | 1例の線形MAC下限 |\n|---|---|---:|---:|---:|---:|---:|')
    for world,pretty in [('aligned_mirror_teacher','Mirror整合教師'),('independent_mlp_teacher','独立MLP教師')]:
        for k in ('shared','film','mirror','independent'):
            v=r['representation'][world][k]
            out.append(f'| {pretty} | {k} | {v["params"]} | {v["packet_bytes"]} | {pr(v["fresh_median"])} | {pr(v["ood_median"])} | {v["MAC_per_sample"]} |')
    out.append('\nMirror自体のPASS判定: **{}/6、{}**。Mirror整合教師の改善だけをMirror固有の汎化優位とは呼ばない。\n'.format(r['gates']['mirror_specific_5pct_pairs'],'PASS' if r['gates']['mirror_specific_gate'] else 'FAIL'))
    out.append('## 3. 物理係数数のスケーリング\n')
    out.append('| 世界 | 方式 | 幅 | 物理係数 | Fresh NMSE↓ |\n|---|---|---:|---:|---:|')
    for v in r['parameter_axis']:
        out.append(f'| {v["world"]} | {v["model"]} | {v["width"]} | {v["params"]} | {pr(v["fresh_median"])} |')
    out.append('\n3点だけからスケーリング指数や単調な改善法則は主張しない。幅に比例してモデル係数と線形MACも増える。\n')
    out.append('## 4. データ量と情報品質（更新回数は同一）\n')
    out.append('| 教師 | 方式 | ユニーク学習例数 | Fresh NMSE↓ |\n|---|---|---:|---:|')
    for v in r['unique_data_axis']:
        out.append(f'| {v["world"]} | {v["model"]} | {v["unique_examples"]} | {pr(v["fresh_median"])} |')
    out.append('\n品質の操作： `clean_balanced`, `skew`（80/10/10）, `corrupted`（学習ラベル20%シャッフル）, `oracle_filtered`（汚染ラベルを正解メタデータで完全識別し削除）。Oracleは上限対照であり、現実のクリーニング手法ではない。\n')
    out.append('| 教師 | 方式 | データ条件 | 有効サンプル数中央値 | Fresh NMSE↓ | 希少rule1 NMSE↓ |\n|---|---|---|---:|---:|---:|')
    for v in r['quality_axis']:
        out.append(f'| {v["world"]} | {v["model"]} | {v["regime"]} | {v["unique_examples_median"]} | {pr(v["fresh_median"])} | {pr(v["rare_rule1_median"])} |')
    out.append('\nOracleフィルタの汚染データに対するペア勝ち: Mirror {}/6、FiLM {}/6。5/6の閾値に対してMirror {}, FiLM {}。\n'.format(r['gates']['data_quality_oracle_beats_noisy_mirror_pairs'],r['gates']['data_quality_oracle_beats_noisy_film_pairs'],'PASS' if r['gates']['data_quality_mirror_gate'] else 'FAIL','PASS' if r['gates']['data_quality_film_gate'] else 'FAIL'))
    out.append('## 5. Loop：学習の深さと実行時の深さを分ける\n')
    out.append('| モデル | 物理係数 | 1回推論の線形MAC | Fresh NMSE↓ | OOD NMSE↓ |\n|---|---:|---:|---:|---:|')
    for name in ('T1','T2','T3','onepass','distill1'):
        v=r['loops'][name]
        out.append(f'| {name} | {v["params"]} | {v["MAC_per_sample"]} | {pr(v["fresh_median"])} | {pr(v["ood_median"])} |')
    out.append('\nLoopの3段改善（T3がT1を10%以上上回る）：{}/3 → {}。蒸留T1がT3品質5%以内かつMAC半分以下：{}/3 → {}。蒸留教師の320更新費用は別に必要。\n'.format(r['gates']['loop_T3_gain_pairs'],'PASS' if r['gates']['loop_gain_gate'] else 'FAIL',r['gates']['loop_onepass_distill_pairs'],'PASS' if r['gates']['loop_deployment_gate'] else 'FAIL'))
    out.append('## 6. 探索スケーリング：候補探索 vs 単純な追加学習\n')
    out.append('| 試行候補数 | DEV最良を選んだFresh NMSE↓ | FiLM追加学習のFresh NMSE↓ | 候補学習秒中央値 | 対照学習秒中央値 | 選択で5%超勝ち |\n|---:|---:|---:|---:|---:|---:|')
    for v in r['search_aggregates']:
        out.append(f'| {v["budget"]} | {pr(v["selected_median"])} | {pr(v["baseline_median"])} | {v["search_trial_seconds_median"]:.3f} | {v["baseline_seconds_median"]:.3f} | {v["gains_5pct"]}/6 |')
    out.append('\n4候補探索が同じ合計更新数のFiLM継続学習に対して5%以上勝つ条件：{}/6 → **{}**。候補ごとに実MACと保存量が違うので、更新数一致は計算量一致を保証しない。なお探索候補は独立seedも含み、純粋なトポロジー探索ではない。\n'.format(r['gates']['search4_gain_pairs'],'PASS' if r['gates']['search_gate'] else 'FAIL'))
    out.append('## 7. 統合した研究上の判断\n')
    out.append('**事実／仮説の境界:** 今回確定するのは各軸の「物理係数数・実行段数・データ量・試行費用の分離」と、その小型合成課題における品質比較であり、5軸の最適な比率ではない。Mirror数は独立した学習パラメータ数ではない。Loopの層数は動的演算数であり、推論時ゼロ追加Loopに蒸留できるかは別判定。データの情報量は単純な件数と一致しない。\n')
    out.append('次の段階では、最も改善余地のある軸を2つだけ組み合わせた交互作用実験を行い、強い非Mirror・非Loop制御と総学習FLOPsを揃えて検証する。自然文Attentionへの導入前に、因果性・KVキャッシュ・学習時の総演算量を別ゲートで確認する。\n')
    out.append('## 8. 完全性・再現性\n')
    out.append('`PROTOCOL.json` は訓練前に凍結。`results/SEALED_TRAIN_MANIFEST.json` の全checkpoints SHA256を評価開始時に検証し、DEV出力を全件照合してからFresh/OODを評価。`RESULTS_CORE.csv` に189モデルの各結果、学習時間・MAC・物理係数・保存量を保存。3seed×2教師なので因果一般化や統計的有意差は主張しない。CPUクロック固定なし。\n')
    (ROOT/'REPORT_JA.md').write_text('\n'.join(out)+'\n',encoding='utf-8')

if __name__=='__main__':
    d=analyze()
    report(d)
    print('GATES',json.dumps(d['gates'],ensure_ascii=False))
    print('RESPONSE_MEDIANS',json.dumps(d['representation'],ensure_ascii=False))
    print('ANALYZED',d['rows'],'ROWS',flush=True)
