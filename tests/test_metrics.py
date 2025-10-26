from ragcraft.metrics import mrr, ndcg, precision_at_k, recall_at_k, faithfulness_score


def test_mrr_first():
    assert mrr(["a", "b"], ["a", "c", "d"]) == 1.0

def test_mrr_second():
    assert mrr(["a"], ["c", "a", "d"]) == 0.5

def test_mrr_miss():
    assert mrr(["a"], ["b", "c"]) == 0.0

def test_precision():
    assert precision_at_k(["a", "b"], ["a", "c", "b", "d", "e"], 5) == 0.4

def test_faithfulness():
    score = faithfulness_score("the cat sat on the mat", ["the cat was sitting on a mat"])
    assert score > 0.5

