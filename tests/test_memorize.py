from secleak.audit.memorize import CallableClient, build_examples, compare, evaluate


def _lines():
    header = ",".join(f"c{i}" for i in range(20))
    body = [",".join(str((i * 7 + j * 3) % 97) for j in range(20)) for i in range(400)]
    return [header] + body


def test_memorising_model_beats_control_and_guessing_model_does_not():
    ex = build_examples(_lines(), "toy.csv", n=30, seed=0)
    truth = {e.prompt: e.truth for e in ex if e.condition == "real"}   # simulate memorisation
    mem = CallableClient(lambda p: truth.get(p, "0," * 19 + "0"))
    rep = {r["probe"]: r for r in compare(evaluate(mem, ex), n_perm=2000)}
    assert rep["row"]["diff"] > 0.5 and rep["row"]["p_one_sided"] < 0.01
    assert rep["feature"]["diff"] > 0.5

    guess = CallableClient(lambda p: "0")
    rep2 = {r["probe"]: r for r in compare(evaluate(guess, ex), n_perm=2000)}
    assert abs(rep2["row"]["diff"]) < 0.2 and rep2["row"]["p_one_sided"] > 0.05


def test_control_prompts_do_not_contain_true_next_row():
    ex = build_examples(_lines(), "toy.csv", n=20, seed=1)
    lines = set(_lines())
    ctrl_truth = [e.truth for e in ex if e.kind == "row" and e.condition == "control"]
    assert sum(t in lines for t in ctrl_truth) <= 1       # shuffled rows are (almost) never real rows
