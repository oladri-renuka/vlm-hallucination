"""
Visualization suite for VLM hallucination paper.

Generates all figures from results_llava.json and results_internvl2.json.
Outputs PNG files to figures/ directory.

Usage: python visualize.py results_llava.json results_internvl2.json
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

MODELS = ['llava', 'internvl2']
MODEL_LABELS = {'llava': 'LLaVA-1.5-7B', 'internvl2': 'InternVL2-8B'}
MODEL_COLORS = {'llava': '#D85A30', 'internvl2': '#378ADD'}
DOMAIN_ORDER = ['natural', 'chart', 'medical', 'screenshot']
DOMAIN_LABELS = {'natural': 'Natural\n(COCO)', 'chart': 'Chart\n(ChartQA)',
                 'medical': 'Medical\n(SLAKE)', 'screenshot': 'Screenshot\n(RICO)'}
CATEGORY_ORDER = ['existence', 'attribute', 'count', 'spatial', 'ocr_numeric']
CATEGORY_LABELS = {'existence': 'Existence', 'attribute': 'Attribute',
                   'count': 'Count', 'spatial': 'Spatial', 'ocr_numeric': 'OCR/Numeric'}


def load_all(paths):
    results = []
    for p in paths:
        with open(p) as f:
            results.extend(json.load(f))
    return results


def compute_accuracy(results, domain, category, model):
    subset = [r for r in results if r['domain'] == domain
              and r['category'] == category and r['model'] == model
              and r['correct'] is not None]
    if not subset:
        return None, 0
    correct = sum(1 for r in subset if r['correct'])
    return correct / len(subset), len(subset)


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
    })


def fig1_heatmap(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        matrix = []
        annotations = []
        for domain in DOMAIN_ORDER:
            row = []
            ann_row = []
            for cat in CATEGORY_ORDER:
                acc, n = compute_accuracy(results, domain, cat, model)
                if acc is not None:
                    row.append(acc * 100)
                    ann_row.append(f'{acc*100:.1f}%\n(n={n})')
                else:
                    row.append(np.nan)
                    ann_row.append('—')
            matrix.append(row)
            annotations.append(ann_row)
        matrix = np.array(matrix, dtype=float)
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(len(CATEGORY_ORDER)))
        ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER], fontsize=10)
        ax.set_yticks(range(len(DOMAIN_ORDER)))
        ax.set_yticklabels([d.capitalize() for d in DOMAIN_ORDER], fontsize=10)
        ax.set_title(MODEL_LABELS[model], fontsize=13, fontweight=500, pad=10)
        for i in range(len(DOMAIN_ORDER)):
            for j in range(len(CATEGORY_ORDER)):
                text_color = 'white' if (not np.isnan(matrix[i, j]) and matrix[i, j] < 40) else 'black'
                if np.isnan(matrix[i, j]):
                    text_color = '#888'
                ax.text(j, i, annotations[i][j], ha='center', va='center', fontsize=8, color=text_color)
    fig.colorbar(im, ax=axes, shrink=0.8, label='Accuracy (%)')
    fig.suptitle('Accuracy by domain and failure category', fontsize=14, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig1_heatmap.png')
    plt.close()
    print('  fig1_heatmap.png')


def fig2_domain_overall(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DOMAIN_ORDER))
    width = 0.35
    for i, model in enumerate(MODELS):
        accs = []
        for domain in DOMAIN_ORDER:
            subset = [r for r in results if r['domain'] == domain
                      and r['model'] == model and r['correct'] is not None]
            acc = sum(1 for r in subset if r['correct']) / len(subset) if subset else 0
            accs.append(acc * 100)
        bars = ax.bar(x + i * width, accs, width, label=MODEL_LABELS[model],
                      color=MODEL_COLORS[model], edgecolor='white', linewidth=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAIN_ORDER])
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.set_title('Overall accuracy by domain', fontsize=13, fontweight=500)
    plt.tight_layout()
    plt.savefig('figures/fig2_domain_overall.png')
    plt.close()
    print('  fig2_domain_overall.png')


def fig3_category_delta(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    deltas = []
    labels = []
    colors = []
    for domain in DOMAIN_ORDER:
        for cat in CATEGORY_ORDER:
            acc_iv, n_iv = compute_accuracy(results, domain, cat, 'internvl2')
            acc_ll, n_ll = compute_accuracy(results, domain, cat, 'llava')
            if acc_iv is not None and acc_ll is not None:
                delta = (acc_iv - acc_ll) * 100
                deltas.append(delta)
                labels.append(f'{domain[:4]}/{cat[:5]}')
                colors.append('#378ADD' if delta >= 0 else '#D85A30')
    y = np.arange(len(deltas))
    ax.barh(y, deltas, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Accuracy difference (InternVL2 − LLaVA) in percentage points')
    ax.set_title('Where InternVL2 outperforms vs underperforms LLaVA', fontsize=13, fontweight=500)
    for i, d in enumerate(deltas):
        ha = 'left' if d >= 0 else 'right'
        offset = 1 if d >= 0 else -1
        ax.text(d + offset, i, f'{d:+.1f}', va='center', ha=ha, fontsize=8)
    plt.tight_layout()
    plt.savefig('figures/fig3_category_delta.png')
    plt.close()
    print('  fig3_category_delta.png')


def fig4_adversarial(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        domains_with_adv = []
        pos_accs = []
        adv_accs = []
        for domain in DOMAIN_ORDER:
            exist = [r for r in results if r['domain'] == domain
                     and r['category'] == 'existence' and r['model'] == model
                     and r['correct'] is not None]
            pos = [r for r in exist if r.get('distractor_type') == 'positive']
            adv = [r for r in exist if r.get('distractor_type') == 'adversarial']
            if pos and adv:
                domains_with_adv.append(domain)
                pos_accs.append(sum(1 for r in pos if r['correct']) / len(pos) * 100)
                adv_accs.append(sum(1 for r in adv if r['correct']) / len(adv) * 100)
        x = np.arange(len(domains_with_adv))
        width = 0.35
        ax.bar(x - width/2, pos_accs, width, label='Positive (present)',
               color='#5DCAA5', edgecolor='white')
        ax.bar(x + width/2, adv_accs, width, label='Adversarial (absent)',
               color='#E24B4A', edgecolor='white')
        for i, (p, a) in enumerate(zip(pos_accs, adv_accs)):
            ax.text(i - width/2, p + 1, f'{p:.0f}%', ha='center', fontsize=8)
            ax.text(i + width/2, a + 1, f'{a:.0f}%', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains_with_adv])
        ax.set_ylim(0, 110)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight=500)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel('Accuracy (%)')
    fig.suptitle('Existence accuracy: positive vs adversarial probes', fontsize=13, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig4_adversarial.png')
    plt.close()
    print('  fig4_adversarial.png')


def fig5_error_profile(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cat_colors = {
        'existence': '#378ADD', 'attribute': '#7F77DD', 'count': '#5DCAA5',
        'spatial': '#D85A30', 'ocr_numeric': '#EF9F27'
    }
    for ax, model in zip(axes, MODELS):
        domain_errors = {}
        for domain in DOMAIN_ORDER:
            errors = Counter()
            for cat in CATEGORY_ORDER:
                subset = [r for r in results if r['domain'] == domain
                          and r['category'] == cat and r['model'] == model
                          and r['correct'] is not None]
                wrong = sum(1 for r in subset if not r['correct'])
                if wrong > 0:
                    errors[cat] = wrong
            total_errors = sum(errors.values())
            if total_errors > 0:
                domain_errors[domain] = {k: v/total_errors*100 for k, v in errors.items()}
            else:
                domain_errors[domain] = {}
        x = np.arange(len(DOMAIN_ORDER))
        bottom = np.zeros(len(DOMAIN_ORDER))
        for cat in CATEGORY_ORDER:
            vals = [domain_errors.get(d, {}).get(cat, 0) for d in DOMAIN_ORDER]
            ax.bar(x, vals, bottom=bottom, label=CATEGORY_LABELS[cat],
                   color=cat_colors[cat], edgecolor='white', linewidth=0.5)
            bottom += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in DOMAIN_ORDER])
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight=500)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel('Share of total errors (%)')
    axes[1].legend(frameon=False, fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.suptitle('Error type distribution by domain', fontsize=13, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig5_error_profile.png')
    plt.close()
    print('  fig5_error_profile.png')


def fig6_spatial_balance(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        domains = []
        yes_accs = []
        no_accs = []
        for domain in DOMAIN_ORDER:
            spatial = [r for r in results if r['domain'] == domain
                       and r['category'] == 'spatial' and r['model'] == model
                       and r['correct'] is not None]
            if not spatial:
                continue
            yes_probes = [r for r in spatial if str(r['ground_truth']).lower() == 'yes']
            no_probes = [r for r in spatial if str(r['ground_truth']).lower() == 'no']
            if yes_probes and no_probes:
                domains.append(domain)
                yes_accs.append(sum(1 for r in yes_probes if r['correct']) / len(yes_probes) * 100)
                no_accs.append(sum(1 for r in no_probes if r['correct']) / len(no_probes) * 100)
        x = np.arange(len(domains))
        width = 0.35
        ax.bar(x - width/2, yes_accs, width, label='GT = yes', color='#5DCAA5', edgecolor='white')
        ax.bar(x + width/2, no_accs, width, label='GT = no', color='#E24B4A', edgecolor='white')
        for i, (y, n) in enumerate(zip(yes_accs, no_accs)):
            ax.text(i - width/2, y + 1, f'{y:.0f}%', ha='center', fontsize=8)
            ax.text(i + width/2, n + 1, f'{n:.0f}%', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains])
        ax.set_ylim(0, 115)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight=500)
        ax.axhline(50, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel('Accuracy (%)')
    fig.suptitle('Spatial accuracy by ground truth polarity', fontsize=13, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig6_spatial_balance.png')
    plt.close()
    print('  fig6_spatial_balance.png')


def fig7_unparseable(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = ['existence', 'count', 'spatial', 'attribute', 'ocr_numeric']
    x = np.arange(len(cats))
    width = 0.35
    for i, model in enumerate(MODELS):
        rates = []
        for cat in cats:
            subset = [r for r in results if r['category'] == cat and r['model'] == model]
            unp = sum(1 for r in subset if r.get('parsed_answer') == 'unparseable')
            rates.append(unp / len(subset) * 100 if subset else 0)
        ax.bar(x + i * width, rates, width, label=MODEL_LABELS[model],
               color=MODEL_COLORS[model], edgecolor='white')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats])
    ax.set_ylabel('Unparseable rate (%)')
    ax.set_title('Unparseable response rates by category', fontsize=13, fontweight=500)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('figures/fig7_unparseable.png')
    plt.close()
    print('  fig7_unparseable.png')


def fig8_model_radar(results):
    cats = ['existence', 'count', 'spatial', 'attribute', 'ocr_numeric']
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(polar=True))
    for ax, domain in zip(axes.flat, DOMAIN_ORDER):
        for model in MODELS:
            values = []
            for cat in cats:
                acc, n = compute_accuracy(results, domain, cat, model)
                values.append(acc * 100 if acc is not None else 0)
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=1.5, markersize=4,
                    label=MODEL_LABELS[model], color=MODEL_COLORS[model])
            ax.fill(angles, values, alpha=0.1, color=MODEL_COLORS[model])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
        ax.set_title(domain.capitalize(), fontsize=12, fontweight=500, pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8, frameon=False)
    fig.suptitle('Capability profiles by domain', fontsize=14, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig8_model_radar.png')
    plt.close()
    print('  fig8_model_radar.png')


def fig9_yes_bias(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    data = []
    for domain in DOMAIN_ORDER:
        for model in MODELS:
            exist = [r for r in results if r['domain'] == domain
                     and r['category'] == 'existence' and r['model'] == model
                     and r.get('parsed_answer') in ('yes', 'no')]
            if exist:
                yes_rate = sum(1 for r in exist if r['parsed_answer'] == 'yes') / len(exist) * 100
                data.append((domain, model, yes_rate))
    x = np.arange(len(DOMAIN_ORDER))
    width = 0.35
    for i, model in enumerate(MODELS):
        rates = [next((d[2] for d in data if d[0] == domain and d[1] == model), 0) for domain in DOMAIN_ORDER]
        bars = ax.bar(x + i * width, rates, width, label=MODEL_LABELS[model],
                      color=MODEL_COLORS[model], edgecolor='white')
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontsize=8)
    ax.axhline(50, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([d.capitalize() for d in DOMAIN_ORDER])
    ax.set_ylabel('"Yes" response rate (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Existence yes-bias: proportion of "yes" responses', fontsize=13, fontweight=500)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('figures/fig9_yes_bias.png')
    plt.close()
    print('  fig9_yes_bias.png')


def fig10_domain_shift_severity(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        categories = []
        best_accs = []
        worst_accs = []
        best_labels = []
        worst_labels = []
        for cat in CATEGORY_ORDER:
            accs = {}
            for domain in DOMAIN_ORDER:
                acc, n = compute_accuracy(results, domain, cat, model)
                if acc is not None and n >= 5:
                    accs[domain] = acc * 100
            if len(accs) >= 2:
                best_d = max(accs, key=accs.get)
                worst_d = min(accs, key=accs.get)
                categories.append(CATEGORY_LABELS[cat])
                best_accs.append(accs[best_d])
                worst_accs.append(accs[worst_d])
                best_labels.append(best_d[:4])
                worst_labels.append(worst_d[:4])
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, best_accs, width, label='Best domain', color='#5DCAA5', edgecolor='white')
        ax.bar(x + width/2, worst_accs, width, label='Worst domain', color='#E24B4A', edgecolor='white')
        for i, (bb, bw, bl, wl) in enumerate(zip(best_accs, worst_accs, best_labels, worst_labels)):
            ax.text(i - width/2, bb + 1, f'{bb:.0f}%\n({bl})', ha='center', fontsize=7)
            ax.text(i + width/2, bw + 1, f'{bw:.0f}%\n({wl})', ha='center', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight=500)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel('Accuracy (%)')
    fig.suptitle('Domain-shift severity: best vs worst domain per category',
                 fontsize=13, fontweight=500, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig10_domain_shift.png')
    plt.close()
    print('  fig10_domain_shift.png')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize.py results_llava.json results_internvl2.json")
        sys.exit(1)

    setup_style()
    results = load_all(sys.argv[1:])
    print(f"Loaded {len(results)} results ({Counter(r['model'] for r in results)})")
    print("\nGenerating figures...")

    fig1_heatmap(results)
    fig2_domain_overall(results)
    fig3_category_delta(results)
    fig4_adversarial(results)
    fig5_error_profile(results)
    fig6_spatial_balance(results)
    fig7_unparseable(results)
    fig8_model_radar(results)
    fig9_yes_bias(results)
    fig10_domain_shift_severity(results)

    print(f"\nAll figures saved to figures/")
    print(f"Total: {len(os.listdir('figures'))} files")
