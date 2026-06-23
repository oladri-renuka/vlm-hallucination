"""
Analysis script -- domain x category x model breakdown + statistical tests.

Usage: python analyze_results.py results_llava.json results_internvl2.json
"""

import json
import sys
import csv
from collections import defaultdict, Counter
import numpy as np


def load_results(*paths):
    all_results = []
    for path in paths:
        with open(path) as f:
            all_results.extend(json.load(f))
    return all_results


def compute_cells(results):
    cells = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r["correct"] is None:
            continue
        key = (r["domain"], r["category"], r["model"])
        cells[key]["total"] += 1
        if r["correct"]:
            cells[key]["correct"] += 1
    return cells


def print_main_table(cells):
    domains = sorted(set(k[0] for k in cells))
    categories = sorted(set(k[1] for k in cells))
    models = sorted(set(k[2] for k in cells))

    header = ["domain", "category"] + [f"{m}_acc" for m in models] + [f"{m}_n" for m in models]
    print("\t".join(header))

    for domain in domains:
        for category in categories:
            row = [domain, category]
            for m in models:
                c = cells.get((domain, category, m))
                if c and c["total"] > 0:
                    row.append(f"{c['correct']/c['total']:.3f}")
                else:
                    row.append("-")
            for m in models:
                c = cells.get((domain, category, m))
                row.append(str(c["total"]) if c else "0")
            print("\t".join(row))


def domain_summary(cells):
    models = sorted(set(k[2] for k in cells))
    domains = sorted(set(k[0] for k in cells))

    print("\n=== Domain-level accuracy (all categories pooled) ===")
    print("domain\t" + "\t".join(f"{m}_acc\t{m}_n" for m in models))
    for domain in domains:
        row = [domain]
        for m in models:
            correct = sum(cells[(domain, cat, m)]["correct"]
                         for cat in set(k[1] for k in cells) if (domain, cat, m) in cells)
            total = sum(cells[(domain, cat, m)]["total"]
                       for cat in set(k[1] for k in cells) if (domain, cat, m) in cells)
            if total > 0:
                row.extend([f"{correct/total:.3f}", str(total)])
            else:
                row.extend(["-", "0"])
        print("\t".join(row))


def existence_breakdown(results):
    cells = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r["correct"] is None or r["category"] != "existence":
            continue
        dt = r.get("distractor_type", "unknown")
        key = (r["domain"], r["model"], dt)
        cells[key]["total"] += 1
        if r["correct"]:
            cells[key]["correct"] += 1

    print("\n=== Existence: Positive vs Adversarial ===")
    print("domain\tmodel\ttype\tacc\tn")
    for key in sorted(cells.keys()):
        c = cells[key]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        print(f"{key[0]}\t{key[1]}\t{key[2]}\t{acc:.3f}\t{c['total']}")


def unparseable_report(results):
    cells = defaultdict(int)
    totals = defaultdict(int)
    for r in results:
        if r["category"] == "confabulation":
            continue
        key = (r["model"], r["category"])
        totals[key] += 1
        if r.get("parsed_answer") == "unparseable":
            cells[key] += 1

    print("\n=== Unparseable responses (excluding confabulation) ===")
    print("model\tcategory\tunparseable\ttotal\trate")
    for key in sorted(totals.keys()):
        u = cells[key]
        t = totals[key]
        print(f"{key[0]}\t{key[1]}\t{u}\t{t}\t{u/t:.3f}")


def error_report(results):
    errs = [r for r in results if r.get("error")]
    if not errs:
        print("\n=== No runtime errors ===")
        return
    print(f"\n=== Runtime errors: {len(errs)} ===")
    for r in errs:
        print(f"  [{r['domain']}] {r['model']} image_id={r['image_id']}: {r['error']}")


def chi_square_interaction(results):
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        print("\n[scipy not installed -- skipping chi-square test]")
        return

    models = sorted(set(r["model"] for r in results))
    for model in models:
        model_results = [r for r in results if r["model"] == model and r["correct"] is not None]
        domains = sorted(set(r["domain"] for r in model_results))
        categories = sorted(set(r["category"] for r in model_results))

        table = np.zeros((len(domains), len(categories)), dtype=int)
        for r in model_results:
            if not r["correct"]:
                di = domains.index(r["domain"])
                ci = categories.index(r["category"])
                table[di][ci] += 1

        if table.sum() == 0 or np.any(table.sum(axis=1) == 0) or np.any(table.sum(axis=0) == 0):
            print(f"\n[{model}] Skipping chi-square -- insufficient errors in some cells")
            continue

        chi2, p, dof, _ = chi2_contingency(table)
        print(f"\n=== Chi-square: domain x category interaction for {model} ===")
        print(f"chi2={chi2:.2f}, p={p:.4f}, dof={dof}")
        print(f"Domains: {domains}")
        print(f"Categories: {categories}")
        print("Error counts by domain:")
        for i, d in enumerate(domains):
            print(f"  {d}: {dict(zip(categories, table[i]))}")


def export_csv(cells, path="results_summary.csv"):
    domains = sorted(set(k[0] for k in cells))
    categories = sorted(set(k[1] for k in cells))
    models = sorted(set(k[2] for k in cells))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "category", "model", "correct", "total", "accuracy"])
        for domain in domains:
            for category in categories:
                for model in models:
                    c = cells.get((domain, category, model))
                    if c and c["total"] > 0:
                        writer.writerow([
                            domain, category, model,
                            c["correct"], c["total"],
                            f"{c['correct']/c['total']:.4f}"
                        ])
    print(f"\nCSV written to {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py results_llava.json [results_internvl2.json]")
        sys.exit(1)

    results = load_results(*sys.argv[1:])
    print(f"Loaded {len(results)} result rows")
    print(f"Models: {Counter(r['model'] for r in results)}")

    cells = compute_cells(results)

    print("\n=== Accuracy by Domain x Category x Model ===")
    print_main_table(cells)
    domain_summary(cells)
    existence_breakdown(results)
    unparseable_report(results)
    error_report(results)
    chi_square_interaction(results)
    export_csv(cells)
