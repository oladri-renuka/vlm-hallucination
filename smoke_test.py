"""
Smoke test: 3 images per domain, both probes per image.
Validates full pipeline end-to-end with minimal GPU time.
Use 4-bit quantization here -- NOT for real results.

Usage: python smoke_test.py llava
       python smoke_test.py internvl2
"""

import json
import sys
import os
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from run_pipeline import run_pipeline


def select_smoke_probes(probes_path="probes_all.json", n_images_per_domain=3):
    with open(probes_path) as f:
        probes = json.load(f)

    by_domain_image = defaultdict(list)
    for p in probes:
        by_domain_image[(p["domain"], str(p["image_id"]))].append(p)

    selected = []
    domain_counts = defaultdict(int)
    for (domain, img_id), img_probes in by_domain_image.items():
        if domain_counts[domain] >= n_images_per_domain:
            continue
        selected.extend(img_probes[:2])
        domain_counts[domain] += 1

    out_path = "probes_smoke.json"
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Smoke probe set: {len(selected)} probes across {sum(domain_counts.values())} images")
    print(f"  By domain: {dict(domain_counts)}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smoke_test.py [llava|internvl2]")
        sys.exit(1)

    model_name = sys.argv[1]
    smoke_probes = select_smoke_probes()

    if model_name == "llava":
        from llava_wrapper import LLaVAWrapper
        wrapper = LLaVAWrapper(quantize=True)
    elif model_name == "internvl2":
        from internvl_wrapper import InternVLWrapper
        wrapper = InternVLWrapper(max_num=6)
    else:
        print(f"Unknown model: {model_name}")
        sys.exit(1)

    results = run_pipeline(
        wrapper,
        model_name=model_name,
        probes_path=smoke_probes,
        output_path=f"results_{model_name}_smoke.json",
        limit=None,
        save_every=5,
    )

    parsed_ok = sum(1 for r in results if r["parsed_answer"] != "unparseable"
                    and r["parsed_answer"] != "error" and r["category"] != "confabulation")
    total_gradable = sum(1 for r in results if r["category"] != "confabulation")
    errors = sum(1 for r in results if r.get("error"))

    print(f"\n=== Smoke test summary ===")
    print(f"Total results: {len(results)}")
    print(f"Errors (image load / model crash): {errors}")
    print(f"Parseable (excl confabulation): {parsed_ok}/{total_gradable}")

    if errors > 0:
        print("\nERRORS found -- fix before full run:")
        for r in results:
            if r.get("error"):
                print(f"  [{r['domain']}] {r['image_id']}: {r['error']}")

    if total_gradable > 0 and parsed_ok < total_gradable * 0.5:
        print("\nWARNING: <50% parsed. Check parser logic.")
    else:
        print("\nSmoke test looks good. Proceed to full run with fp16.")
