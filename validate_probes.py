"""
Pre-flight validation. Run BEFORE paying for GPU time.
Checks every image in probes_all.json can be loaded, and validates probe schema.

Usage: python validate_probes.py
"""

import json
import sys
import os
from collections import Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from image_loader import load_image_for_probe

with open("probes_all.json") as f:
    probes = json.load(f)

unique_images = {}
for p in probes:
    key = (str(p["image_id"]), p["domain"])
    if key not in unique_images:
        unique_images[key] = p

print(f"Total probes: {len(probes)}")
print(f"Unique images to check: {len(unique_images)}")
print(f"By domain: {Counter(p['domain'] for p in probes)}")
print(f"By category: {Counter(p['category'] for p in probes)}")
print()

errors = []
for i, ((img_id, domain), probe) in enumerate(unique_images.items()):
    try:
        img = load_image_for_probe(probe)
        assert img.mode == "RGB", f"Expected RGB, got {img.mode}"
        assert img.size[0] > 0 and img.size[1] > 0
    except Exception as e:
        errors.append((domain, img_id, str(e)))
        print(f"  FAIL [{domain}] image_id={img_id}: {e}")

    if (i + 1) % 20 == 0:
        print(f"  checked {i+1}/{len(unique_images)}...")

print()
if errors:
    print(f"FAILED: {len(errors)} images could not be loaded:")
    for domain, img_id, err in errors:
        print(f"  [{domain}] {img_id}: {err}")
    print("\nFix these before running the pipeline on GPU.")
    sys.exit(1)
else:
    print("ALL IMAGES OK.")

required_keys = {"image_id", "domain", "category", "question", "ground_truth"}
for i, p in enumerate(probes):
    missing = required_keys - set(p.keys())
    if missing:
        print(f"Probe {i} missing keys: {missing}")
        sys.exit(1)

valid_categories = {"existence", "attribute", "count", "spatial", "ocr_numeric", "confabulation"}
bad_cats = [p["category"] for p in probes if p["category"] not in valid_categories]
if bad_cats:
    print(f"Invalid categories found: {set(bad_cats)}")
    sys.exit(1)

print("Schema validation passed.")
print("Safe to proceed to GPU run.")
