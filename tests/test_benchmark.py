from benchmarks.run_benchmark import character_error_rate


def test_cer_is_explicit_and_does_not_correct_spelling():
    assert character_error_rate("becuse", "because") > 0
    assert character_error_rate("because", "because") == 0
