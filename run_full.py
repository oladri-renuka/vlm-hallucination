"""
Full pipeline orchestrator for RunPod.

Usage:
    python run_full.py llava        # run LLaVA only
    python run_full.py internvl2    # run InternVL2 only
    python run_full.py both         # run both sequentially in separate processes
"""

import sys
import os
import gc
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_model(model_name):
    if model_name == "llava":
        from llava_wrapper import LLaVAWrapper
        wrapper = LLaVAWrapper(quantize=False)
    elif model_name == "internvl2":
        from internvl_wrapper import InternVLWrapper
        wrapper = InternVLWrapper(max_num=12, quantize=False)
    else:
        print(f"Unknown model: {model_name}")
        sys.exit(1)

    from run_pipeline import run_pipeline
    run_pipeline(
        wrapper,
        model_name=model_name,
        probes_path="probes_all.json",
        output_path=f"results_{model_name}.json",
        limit=None,
        save_every=10,
    )

    del wrapper
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_full.py [llava|internvl2|both]")
        sys.exit(1)

    target = sys.argv[1]

    if target == "both":
        for model in ["llava", "internvl2"]:
            print(f"\n{'='*60}")
            print(f"Starting {model} in subprocess...")
            print(f"{'='*60}\n")
            result = subprocess.run(
                [sys.executable, __file__, model],
                check=False,
            )
            if result.returncode != 0:
                print(f"WARNING: {model} exited with code {result.returncode}")
    else:
        run_model(target)

    print("\nDone. Run: python analyze_results.py results_llava.json results_internvl2.json")
