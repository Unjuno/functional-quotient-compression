from copy import deepcopy
from fqc.contracts import validate_pilot_contract, validate_transformer_extraction

PILOT={
 'package_version':'test-v1','model':{'model_id':'m','checkpoint_hash':'sha256:x','unique_paid_scalar_count_N':101},
 'baseline':{'bits_per_scalar':16,'baseline_bits':1616,'unique_storage_rule':'once','external_free_items':[]},
 'target':{'compression_factor':64,'B64_integer_bits':25,'metric_id':'m','quality_target':1.0},
 'decoder_protocol':{'protocol_id':'fqc','version':'1'},'numeric_contract':{'evaluation_dtype':'float64','randomness_policy':'deterministic'},
 'artifact_hashes':{'tensor_inventory':'sha256:a','operator_manifest':'sha256:b','paid_atom_manifest':'sha256:c','quality_contract':'sha256:d','replay_contract':'sha256:e'}}
ATOMS={'atoms':[
 {'atom_id':'root','class':'PAID','bits':8,'dependencies':[]},
 {'atom_id':'coeff','class':'PAID','bits':10,'dependencies':['root']},
 {'atom_id':'basis','class':'DERIVED','bits':0,'dependencies':['root']},
 {'atom_id':'arch','class':'EXTERNAL_FIXED','bits':0,'dependencies':[]}], 'required_decode_outputs':['basis','coeff']}
EXTRACTION={
 'model_identity':{'model_id':'x'},
 'tensor_inventory':[
  {'tensor_id':'embed','shape':[10,4],'dtype':'bf16','storage_group':'shared','baseline_included':True},
  {'tensor_id':'lm_head','shape':[10,4],'dtype':'bf16','storage_group':'shared','baseline_included':True},
  {'tensor_id':'WQ','shape':[4,4],'dtype':'bf16','storage_group':'q','baseline_included':True},
  {'tensor_id':'WK','shape':[4,2],'dtype':'bf16','storage_group':'k','baseline_included':True},
  {'tensor_id':'WV','shape':[4,2],'dtype':'bf16','storage_group':'v','baseline_included':True},
  {'tensor_id':'WO','shape':[4,4],'dtype':'bf16','storage_group':'o','baseline_included':True}],
 'modules':{'attention':[{'module_id':'a','q_heads':2,'kv_heads':1,'head_dim':2,'q_to_kv_map':[0,0],
  'projection_tensor_ids':{'WQ':'WQ','WK':'WK','WV':'WV','WO':'WO'},'position_operator':'rope','rope':{'rotary_dim':2},
  'qk_norm':'none','exact_reduction_policy':'rope_profile_operator_family'}]},
 'external_fixed_state':[],'derived_primitives':[{'primitive_id':'p','source_tensor_ids':['WQ','WK'],'analysis_only':True,'paid_if_serialized':True}], 'extraction_policy':{}}

def test_d56_example_accounting():
 r=validate_pilot_contract(PILOT,ATOMS); assert r.valid; assert (r.target_bits_64x,r.total_paid_atom_bits,r.required_paid_atom_bits)==(25,18,18)

def test_cross_class_cycle_rejected():
 a=deepcopy(ATOMS); a['atoms'][0]['dependencies']=['basis']; assert not validate_pilot_contract(PILOT,a).valid

def test_total_vs_required_paid_bits():
 a=deepcopy(ATOMS); a['atoms'].append({'atom_id':'unused','class':'PAID','bits':7,'dependencies':[]}); r=validate_pilot_contract(PILOT,a)
 assert r.valid and r.required_paid_atom_bits==18 and r.total_paid_atom_bits==25 and r.warnings

def test_storage_tie_counted_once_but_requires_proof():
 r=validate_transformer_extraction(EXTRACTION); assert r.valid and r.unique_baseline_scalar_count==88
 assert any('actual checkpoint storage identity' in w for w in r.warnings)

def test_duplicate_tensor_ids_rejected():
 m=deepcopy(EXTRACTION); m['tensor_inventory'].append(deepcopy(m['tensor_inventory'][0])); assert not validate_transformer_extraction(m).valid

def test_plain_bilinear_rejected_under_rope():
 m=deepcopy(EXTRACTION); m['modules']['attention'][0]['exact_reduction_policy']='plain_bilinear_family'; r=validate_transformer_extraction(m)
 assert not r.valid and any('RoPE' in e for e in r.errors)

def test_row_vector_attention_projection_shapes_are_checked():
 m=deepcopy(EXTRACTION)
 m['model_identity']['orientation']='row_vector_x_times_W'
 m['modules']['attention'][0]['d_model']=4
 m['modules']['attention'][0]['value_dim']=2
 assert validate_transformer_extraction(m).valid
 m['tensor_inventory'][2]['shape']=[4,3]
 r=validate_transformer_extraction(m)
 assert not r.valid and any('WQ shape' in e for e in r.errors)
