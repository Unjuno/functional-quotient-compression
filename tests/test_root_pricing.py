from fqc.root_pricing import exact_finite_family_optimum, k_complete_root_sets, one_column_lookahead, coalition_break_even


def test_k_complete_enumeration_count_matches_d65_formula():
    roots=[f'R{i}' for i in range(100)]
    assert sum(1 for _ in k_complete_root_sets(roots,2)) == 1+100+4950


def test_one_column_stopping_is_not_safe_with_shared_prerequisite_complementarity():
    def solve_master(roots):
        roots=set(roots)
        bits=(4 if roots else 0)+len(roots)
        error=0.0
        for root in ('A','B'):
            if root in roots: bits+=1
            else: bits+=5; error+=5.0
        if bits>10: return None
        return {'bits':bits,'error':error,'roots':tuple(sorted(roots))}
    assert solve_master(())=={'bits':10,'error':10.0,'roots':()}
    assert one_column_lookahead((),['A','B'],solve_master) is None
    assert exact_finite_family_optimum(['A','B'],2,solve_master)=={'bits':8,'error':0.0,'roots':('A','B')}


def test_master_optimality_depends_on_candidate_family():
    table={():{'bits':10,'error':10.0},('A',):{'bits':9,'error':5.0},('B',):{'bits':9,'error':4.0},('A','B'):{'bits':8,'error':1.0}}
    solve=lambda roots: table.get(tuple(roots))
    assert exact_finite_family_optimum(['A'],2,solve)['error']==5.0
    assert exact_finite_family_optimum(['A','B'],2,solve)['error']==1.0


def test_coalition_screening_is_only_rate_screening():
    assert coalition_break_even(32,[15,14,13,12],[4,4,4,4],[0,1])==-11
    assert coalition_break_even(32,[15,14,13,12],[4,4,4,4],[0,1,2])==-2
    assert coalition_break_even(32,[15,14,13,12],[4,4,4,4],[0,1,2,3])==6

from fqc.root_pricing import (
    exact_replacement_score, optimistic_refund_lower_bound, family_lower_bound,
    safe_prune, grouped_lower_bound, closure_coupled_group_bound,
)


def test_d67_replacement_trap_reduced_objective():
    delta=exact_replacement_score(0.1,36,40,[0.3,0.3,0.3,0.3],[0.1,0.1,0.1,0.1])
    assert abs(delta+0.4)<1e-12


def test_optimistic_refund_bound_never_exceeds_concrete_delta_in_witness():
    current=[0.5,0.4,0.3,0.2]
    concrete=[0.2,0.6,0.1,0.5]
    exact=exact_replacement_score(0.1,36,44,current,concrete)
    lb=optimistic_refund_lower_bound(0.1,36,44,current,concrete)
    assert lb <= exact + 1e-12


def test_d68_family_bound_safe_prune_semantics():
    current=[0.5,0.5,0.5,0.5]
    lb=family_lower_bound(0.1,13,current,[0.5,0.6,0.4,0.5])
    assert abs(lb-1.3)<1e-12
    assert safe_prune(lb)
    assert not safe_prune(-0.01)


def test_d69_compatibility_grouping_can_prune_when_singletons_cannot():
    current=[0.0,0.0,0.0,0.0]
    coalitions=[
      {'closure_delta':13,'block_q':[-2.0,1.0,0.0,0.0]},
      {'closure_delta':13,'block_q':[1.0,-2.0,0.0,0.0]},
    ]
    singleton=grouped_lower_bound(0.1,13,current,coalitions,[(0,),(1,),(2,),(3,)])
    paired=grouped_lower_bound(0.1,13,current,coalitions,[(0,1),(2,3)])
    assert singleton < 0
    assert abs(paired-0.3)<1e-12
    assert safe_prune(paired)


def test_d69_closure_coupling_removes_independent_minimum_incompatibility():
    current=[0.0,0.0]
    coalitions=[
      {'closure_delta':0,'block_q':[1.0,0.0]},
      {'closure_delta':10,'block_q':[-1.0,0.0]},
    ]
    uncoupled=grouped_lower_bound(0.1,0,current,coalitions,[(0,),(1,)])
    coupled=closure_coupled_group_bound(0.1,current,coalitions,(0,),[(1,)])
    exact=min(0.1*c['closure_delta']+sum(c['block_q']) for c in coalitions)
    assert uncoupled==-1.0
    assert abs(coupled-exact)<1e-12
    assert safe_prune(coupled)
