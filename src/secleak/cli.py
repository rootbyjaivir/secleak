"""Command-line interface:  secleak {demo,audit,run,cross,report,memorize,census}"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load(args):
    from .data import load_dataset, subsample
    ds = load_dataset(args.dataset, root=args.root, max_rows_per_file=args.max_rows_per_file)
    return subsample(ds, args.max_rows) if args.max_rows and args.max_rows > 0 else ds


def cmd_demo(args):
    from .audit.card import build_card, card_markdown, save_card
    from .data import frame_to_dataset
    from .experiments import run_matrix
    from .report import write_report
    from .synthetic import make_synthetic
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(label_col="label", benign_values=["BENIGN"], time_col="timestamp",
               time_kind="iso", drop_cols=["timestamp"])
    ds = frame_to_dataset(make_synthetic(n=args.n, seed=0), "synthetic", cfg)
    card = build_card(ds)
    save_card(card, out / "card_synthetic.json", out / "card_synthetic.md")
    print(card_markdown(card))
    df = run_matrix(ds, models=["logreg", "rf", "gbdt"], seeds=range(args.seeds), tune=False)
    df.to_csv(out / "results.csv", index=False)
    print("report ->", write_report(out / "results.csv", out))


def cmd_audit(args):
    from .audit.card import build_card, card_markdown, save_card
    ds = _load(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    card = build_card(ds)
    save_card(card, out / f"card_{ds.name}.json", out / f"card_{ds.name}.md")
    print(card_markdown(card))


def cmd_run(args):
    from .experiments import run_matrix
    ds = _load(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = run_matrix(ds, protocols=args.protocols, models=args.models, seeds=range(args.seeds),
                    tune=not args.no_tune, max_train_rows=args.max_train_rows,
                    max_test_rows=args.max_test_rows or None, cap_mode=args.cap_mode)
    path = out / f"results_{ds.name}.csv"
    df.to_csv(path, index=False)
    print("saved", path)


def cmd_diagnose(args):
    from .diagnose import diagnose, write_diagnostics
    ds = _load(args)
    res = diagnose(ds, models=args.models, seeds=range(args.seeds), gaps=args.gaps,
                   max_train_rows=args.max_train_rows, max_test_rows=args.max_test_rows or None,
                   cap_mode=args.cap_mode)
    print("diagnostics ->", write_diagnostics(res, Path(args.out) / f"diagnose_{ds.name}", ds.name))


def cmd_cross(args):
    from .data import harmonise, load_config, load_dataset, subsample  # noqa: F401
    from .experiments import run_cross
    cfg = load_config()
    a = subsample(load_dataset(args.a, args.root), args.max_rows)
    b = subsample(load_dataset(args.b, args.root), args.max_rows)
    df = run_cross(a, b, cfg["harmonise"][args.a], cfg["harmonise"][args.b],
                   models=args.models, seeds=range(args.seeds))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"results_cross_{args.a}_{args.b}.csv", index=False)


def cmd_report(args):
    from .report import write_report
    files = sorted(Path(args.results).glob("results*.csv")) if Path(args.results).is_dir() \
        else [Path(args.results)]
    merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    tmp = Path(args.out); tmp.mkdir(parents=True, exist_ok=True)
    merged.to_csv(tmp / "all_results.csv", index=False)
    print("report ->", write_report(tmp / "all_results.csv", tmp))


def cmd_memorize(args):
    from .audit.memorize import AnthropicClient, build_examples, compare, evaluate, read_raw_lines
    lines = read_raw_lines(args.csv, n_max=args.max_lines)
    ex = build_examples(lines, Path(args.csv).name, n=args.n, seed=args.seed)
    print(f"{len(ex)} probe prompts; model={args.model or 'env SECLEAK_MODEL / default'}")
    res = evaluate(AnthropicClient(args.model), ex, verbose=True)
    rep = compare(res)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"memorize_{Path(args.csv).stem}.json").write_text(json.dumps(rep, indent=2, default=float))
    print(pd.DataFrame(rep).to_string(index=False))


def cmd_census(args):
    from .census import analyze
    analyze(args.csv, args.out, second=args.second)


def main(argv=None):
    p = argparse.ArgumentParser(prog="secleak")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(s):
        s.add_argument("dataset"); s.add_argument("--root", default=".")
        s.add_argument("--max-rows", type=int, default=500_000)
        s.add_argument("--max-rows-per-file", type=int, default=None)
        s.add_argument("--out", default="results")

    s = sub.add_parser("demo"); s.add_argument("--out", default="results/demo")
    s.add_argument("--n", type=int, default=20000); s.add_argument("--seeds", type=int, default=3)
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("audit"); data_args(s); s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("run"); data_args(s)
    s.add_argument("--protocols", nargs="+", default=["random", "random_dedup", "temporal_global", "temporal_per_class",
                                                    "temporal_per_label"])
    s.add_argument("--models", nargs="+", default=["logreg", "rf", "gbdt", "mlp"])
    s.add_argument("--seeds", type=int, default=5); s.add_argument("--no-tune", action="store_true")
    s.add_argument("--max-train-rows", type=int, default=200_000)
    s.add_argument("--max-test-rows", type=int, default=200_000, help="0 = evaluate on ALL test rows")
    s.add_argument("--cap-mode", choices=["uniform", "label_floor"], default="uniform")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("diagnose"); data_args(s)
    s.add_argument("--models", nargs="+", default=["rf"]); s.add_argument("--seeds", type=int, default=2)
    s.add_argument("--gaps", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.10])
    s.add_argument("--max-train-rows", type=int, default=200_000)
    s.add_argument("--max-test-rows", type=int, default=200_000, help="0 = evaluate on ALL test rows")
    s.add_argument("--cap-mode", choices=["uniform", "label_floor"], default="uniform")
    s.set_defaults(fn=cmd_diagnose)

    s = sub.add_parser("cross"); s.add_argument("a"); s.add_argument("b")
    s.add_argument("--root", default="."); s.add_argument("--max-rows", type=int, default=200_000)
    s.add_argument("--models", nargs="+", default=["logreg", "rf", "gbdt"])
    s.add_argument("--seeds", type=int, default=3); s.add_argument("--out", default="results")
    s.set_defaults(fn=cmd_cross)

    s = sub.add_parser("report"); s.add_argument("results"); s.add_argument("--out", default="results/report")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("memorize"); s.add_argument("csv"); s.add_argument("--model", default=None)
    s.add_argument("--n", type=int, default=40); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-lines", type=int, default=50_000); s.add_argument("--out", default="results")
    s.set_defaults(fn=cmd_memorize)

    s = sub.add_parser("census"); s.add_argument("csv"); s.add_argument("--second", default=None)
    s.add_argument("--out", default="results/census"); s.set_defaults(fn=cmd_census)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
