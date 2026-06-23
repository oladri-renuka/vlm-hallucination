"""
Pipeline runner -- loops over probes_all.json, calls a model wrapper,
parses + grades each response, saves results incrementally.

Run ONE MODEL AT A TIME, not both in the same process -- this avoids the
GPU memory stacking issue hit when LLaVA and InternVL2 were loaded
together in one Kaggle session. Run this script once per model, each in
its own clean session/process, then merge the two results files.

Usage (inside a notebook cell, or adapt the __main__ block for a .py run):
    from run_pipeline import run_pipeline
    from llava_wrapper import LLaVAWrapper          # or internvl_wrapper
    wrapper = LLaVAWrapper(quantize=False)          # fp16 for real results
    run_pipeline(wrapper, model_name="llava", probes_path="probes_all.json",
                 output_path="results_llava.json", limit=None)
"""

import json
import os
import time

from parser import parse_response
from grader import grade
from image_loader import load_image_for_probe


def run_pipeline(wrapper, model_name, probes_path="probes_all.json",
                  output_path=None, limit=None, save_every=10):
    """
    wrapper: an object with .ask(image, prompt) -> str  (LLaVAWrapper or InternVLWrapper)
    model_name: "llava" or "internvl2" -- tagged onto every result row
    limit: if set, only run the first N probes (use for smoke testing)
    save_every: write partial results to disk every N probes, so a crash
                mid-run doesn't lose everything
    """
    if output_path is None:
        output_path = f"results_{model_name}.json"

    with open(probes_path) as f:
        probes = json.load(f)
    if limit:
        probes = probes[:limit]

    # resume support: if output_path already has partial results, skip
    # probes already completed rather than re-running them from scratch
    completed_ids = set()
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        completed_ids = set((r["image_id"], r["question"]) for r in results)
        print(f"Resuming: {len(results)} results already saved, skipping those")

    # cache loaded images per image_id within this run -- several probes
    # often share the same image_id, no reason to reload from disk/network
    # every single probe
    image_cache = {}

    total = len(probes)
    errors = 0
    for i, probe in enumerate(probes):
        key = (probe["image_id"], probe["question"])
        if key in completed_ids:
            continue

        try:
            if probe["image_id"] not in image_cache:
                image_cache[probe["image_id"]] = load_image_for_probe(probe)
            image = image_cache[probe["image_id"]]

            raw_response = wrapper.ask(image, probe["question"])
            parsed = parse_response(raw_response, probe["category"], probe["question"])
            correct = grade(parsed, probe["ground_truth"], probe["category"])

            results.append({
                "probe_index": i,
                "image_id": probe["image_id"],
                "domain": probe["domain"],
                "category": probe["category"],
                "question": probe["question"],
                "ground_truth": probe["ground_truth"],
                "distractor_type": probe.get("distractor_type"),
                "model": model_name,
                "raw_response": raw_response,
                "parsed_answer": parsed,
                "correct": correct,
            })

        except Exception as e:
            errors += 1
            print(f"ERROR on probe {i} ({probe['domain']}/{probe['category']}): {e}")
            results.append({
                "probe_index": i,
                "image_id": probe["image_id"],
                "domain": probe["domain"],
                "category": probe["category"],
                "question": probe["question"],
                "ground_truth": probe["ground_truth"],
                "distractor_type": probe.get("distractor_type"),
                "model": model_name,
                "raw_response": None,
                "parsed_answer": "error",
                "correct": None,
                "error": str(e),
            })

        if (i + 1) % save_every == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[{i+1}/{total}] saved checkpoint, {errors} errors so far")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done. {len(results)} results written to {output_path}. {errors} errors.")
    return results


if __name__ == "__main__":
    # adapt this block to whichever wrapper you're testing with
    import sys
    sys.path.insert(0, ".")

    from llava_wrapper import LLaVAWrapper

    wrapper = LLaVAWrapper(quantize=True)  # smoke test only -- fp16 for real run
    run_pipeline(
        wrapper, model_name="llava",
        probes_path="probes_all.json",
        output_path="results_llava_smoketest.json",
        limit=20,  # small slice for the smoke test
        save_every=5,
    )
