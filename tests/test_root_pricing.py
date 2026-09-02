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
