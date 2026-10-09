import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'run.py';s=importlib.util.spec_from_file_location('ma333',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_activation_specific_scale_sign_symmetries():
 x,w1,b1,w2,b2,g,b,p,pos,sg=m.params(33321)
 for name in m.ACTS:
  ref=m.evaluate(name,'baseline',x,w1,b1,w2,b2,g,b,p,pos,sg)
  assert m.nr(m.evaluate(name,'permutation_view',x,w1,b1,w2,b2,g,b,p,pos,sg),ref)<1e-6
  positive=m.nr(m.evaluate(name,'positive_scale_compensated',x,w1,b1,w2,b2,g,b,p,pos,sg),ref)
  flip=m.nr(m.evaluate(name,'sign_flip_compensated',x,w1,b1,w2,b2,g,b,p,pos,sg),ref)
  if name=='relu': assert positive<1e-6 and flip>1e-3
  if name=='tanh': assert flip<1e-6 and positive>1e-3
  if name in ('gelu','layernorm_relu'): assert positive>1e-3 and flip>1e-3
def test_serializer_reload_metadata_and_code():
 x,w1,b1,w2,b2,g,b,p,pos,sg=m.params(33322)
 for name in m.ACTS:
  for method in m.METHODS:
   blob=m.pack(name,method,w1,b1,w2,b2,g,b,p,pos,sg);meta,arr,code=m.load(blob)
   assert meta['activation']==name and meta['method']==method
   assert [a.shape for a in arr]==([w1.shape,b1.shape,w2.shape,b2.shape,g.shape,b.shape] if name=='layernorm_relu' else [w1.shape,b1.shape,w2.shape,b2.shape])
   if method=='permutation_view': assert (code==p).all()
