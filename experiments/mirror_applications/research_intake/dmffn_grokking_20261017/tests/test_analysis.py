from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze import grokking_events

def test_delayed_generalization_is_not_automatically_grokking():
    hist=[{'step':0,'train_accuracy':.03,'test_accuracy':.03},
          {'step':200,'train_accuracy':1.,'test_accuracy':.12},
          {'step':1000,'train_accuracy':1.,'test_accuracy':.30},
          {'step':1400,'train_accuracy':1.,'test_accuracy':.58},
          {'step':8000,'train_accuracy':1.,'test_accuracy':.65}]
    a=grokking_events(hist)
    assert a['delayed_generalization']
    assert not a['strong_grokking']
    assert a['train99_step']==200 and a['test_50_step']==1400

def test_strong_grokking_requires_high_test_and_lag():
    hist=[{'step':0,'train_accuracy':.03,'test_accuracy':.03},
          {'step':200,'train_accuracy':.99,'test_accuracy':.12},
          {'step':500,'train_accuracy':1.,'test_accuracy':.14},
          {'step':2600,'train_accuracy':1.,'test_accuracy':.95}]
    a=grokking_events(hist)
    assert a['strong_grokking']
    assert a['test_90_step']==2600

def test_unreached_thresholds_are_censored():
    hist=[{'step':0,'train_accuracy':.03,'test_accuracy':.03},
          {'step':200,'train_accuracy':1.,'test_accuracy':.04},
          {'step':8000,'train_accuracy':1.,'test_accuracy':.40}]
    a=grokking_events(hist)
    assert a['test_50_step'] is None and a['test_90_step'] is None
    assert not a['delayed_generalization'] and not a['strong_grokking']
