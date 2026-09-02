from fqc.allocation import AllocationOption, exact_multiple_choice_allocate, lagrangian_dual, minimum_bits_for_error_target, prune_dominated, variable_budget


def test_fixed_overhead_can_make_rate_impossible_before_allocation():
    assert variable_budget(100,40)==60
    assert variable_budget(25,30)==-5


def test_dominated_option_pruning():
    opts=[AllocationOption('a',4,.5),AllocationOption('b',5,.5),AllocationOption('c',6,.3)]
    assert [o.option_id for o in prune_dominated(opts)]==['a','c']


def test_density_greedy_counterexample_requires_exact_discrete_allocation():
    blocks=[
      [AllocationOption('A0',0,10),AllocationOption('A1',6,0)],
      [AllocationOption('B0',0,8),AllocationOption('B1',5,0)],
      [AllocationOption('C0',0,8),AllocationOption('C1',5,0)],
    ]
    best=exact_multiple_choice_allocate(blocks,10)
    assert best.option_ids==('A0','B1','C1')
    assert best.bits==10 and best.error==10


def test_lagrangian_dual_is_lower_bound_on_integral_optimum():
    blocks=[
      [AllocationOption('a0',0,1.0),AllocationOption('a1',4,.3)],
      [AllocationOption('b0',0,.8),AllocationOption('b1',3,.2)],
    ]
    opt=exact_multiple_choice_allocate(blocks,4)
    dual,_=lagrangian_dual(blocks,4,.15)
    assert opt is not None and dual<=opt.error+1e-12


def test_error_target_inversion():
    blocks=[
      [AllocationOption('a0',0,1),AllocationOption('a1',4,.2)],
      [AllocationOption('b0',0,1),AllocationOption('b1',5,.1)],
    ]
    r=minimum_bits_for_error_target(blocks,.4)
    assert r is not None and r.bits==9 and abs(r.error-.3)<1e-12
