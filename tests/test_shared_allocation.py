from fqc.decoder_dag import Atom
from fqc.shared_allocation import BlockOption, coalition_rate_gain, evaluate_selection, exact_shared_allocate

def test_first_consumer_expensive_second_consumer_amortizes_root():
 atoms=[Atom('Q','PAID',20),Atom('D','PAID',12,('Q',))]
 blocks=[
  [BlockOption('P',15,1.0),BlockOption('S',5,0.4,('D',))],
  [BlockOption('P',15,1.0),BlockOption('S',5,0.4,('D',))]]
 private=evaluate_selection(blocks,atoms,[0,0]); first=evaluate_selection(blocks,atoms,[1,0]); both=evaluate_selection(blocks,atoms,[1,1])
 assert first.total_bits-private.total_bits==22
 assert both.total_bits-first.total_bits==-10

def test_exact_allocator_can_choose_shared_coalition():
 atoms=[Atom('Q','PAID',8),Atom('D','PAID',12,('Q',))]
 blocks=[[BlockOption('P',12,1.0),BlockOption('S',3,0.3,('D',))] for _ in range(3)]
 best=exact_shared_allocate(blocks,atoms,30)
 assert best is not None and best.option_ids==('S','S','S') and best.total_bits==29
 assert abs(best.error-0.9)<1e-12

def test_coalition_break_even_formula_matches_d63_example():
 assert coalition_rate_gain(32,[11,10,8])==-3
 assert coalition_rate_gain(32,[11,10,8,6])==3
