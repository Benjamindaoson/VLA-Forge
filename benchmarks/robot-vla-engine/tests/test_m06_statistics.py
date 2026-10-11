from robot_vla.m06_statistics import (
    exact_one_sided_p,
    holm_adjust,
    paired_target,
    release_decision,
    task_cluster_difference,
    wilson,
)


def test_binomial_and_holm_are_matched_and_conservative():
    assert wilson(0, 10)['lower_95'] == 0
    assert exact_one_sided_p(0, 0) == 1
    assert exact_one_sided_p(10, 0) == 1/1024
    assert holm_adjust({'a': .01, 'b': .02, 'c': .5}) == {'a': .03, 'b': .04, 'c': .5}

def test_ten_target_cases_and_thirty_nine_legacy_clusters():
    target = paired_target([False]*10, [True]*10)
    assert target['paired_difference'] == 1
    assert target['one_sided_exact_p'] == 1/1024
    legacy = task_cluster_difference([0.0]*39)
    assert legacy['one_sided_lower_95'] == 0.0
    assert release_decision(target=target, legacy=legacy, holm_p=.003,
        candidate_chunk_p95_ms=10, frozen_chunk_p95_ms=10,
        illegal_actions=0, missing_or_corrupt=0)['decision'] == 'PROMOTE'
    assert release_decision(target=target, legacy=task_cluster_difference([-.05]*39), holm_p=.003,
        candidate_chunk_p95_ms=10, frozen_chunk_p95_ms=10,
        illegal_actions=0, missing_or_corrupt=0)['decision'] == 'REJECT'
    assert release_decision(target=target, legacy=legacy, holm_p=.06,
        candidate_chunk_p95_ms=10, frozen_chunk_p95_ms=10,
        illegal_actions=0, missing_or_corrupt=0)['decision'] == 'INSUFFICIENT_EVIDENCE'
