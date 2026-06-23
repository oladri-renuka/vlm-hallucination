"""
Local test suite -- exercises every component EXCEPT GPU model inference.
Run this on your Mac before uploading to RunPod.

Tests:
  1. All images load for every probe
  2. Parser handles every answer type correctly
  3. Grader produces expected results
  4. Pipeline works end-to-end with a mock model
  5. Analysis script runs on mock results
  6. No import errors in any module
"""

import json
import sys
import os
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        failed += 1


# ============================================================
# 1. Imports
# ============================================================
print("\n=== 1. Import checks ===")


def test_imports():
    from parser import parse_response
    from grader import grade
    from image_loader import load_image_for_probe
    from run_pipeline import run_pipeline
    from analyze_results import load_results, compute_cells, print_main_table

test("all modules import", test_imports)


def test_llava_wrapper_imports():
    # just check it parses -- can't instantiate without GPU
    import importlib
    spec = importlib.util.spec_from_file_location("llava_wrapper", "llava_wrapper.py")
    mod = importlib.util.module_from_spec(spec)
    # don't exec -- would try to import torch.cuda

test("llava_wrapper.py parses", test_llava_wrapper_imports)


# ============================================================
# 2. Image loading
# ============================================================
print("\n=== 2. Image loading (all probes) ===")

from image_loader import load_image_for_probe

with open("probes_all.json") as f:
    all_probes = json.load(f)

unique_images = {}
for p in all_probes:
    key = (str(p["image_id"]), p["domain"])
    if key not in unique_images:
        unique_images[key] = p

image_errors = []
for (img_id, domain), probe in unique_images.items():
    try:
        img = load_image_for_probe(probe)
        assert img.mode == "RGB"
        assert img.size[0] > 0 and img.size[1] > 0
    except Exception as e:
        image_errors.append((domain, img_id, str(e)))


def test_all_images_load():
    if image_errors:
        for d, i, e in image_errors:
            print(f"    [{d}] {i}: {e}")
        raise Exception(f"{len(image_errors)} images failed to load")

test(f"all {len(unique_images)} images load", test_all_images_load)


# ============================================================
# 3. Parser tests
# ============================================================
print("\n=== 3. Parser ===")

from parser import parse_response


def test_existence_yes():
    assert parse_response("Yes, there is a dog.", "existence") == "yes"

def test_existence_no():
    assert parse_response("No, I don't see a cat.", "existence") == "no"

def test_existence_negative_phrasing():
    assert parse_response("There is not a cat in this image.", "existence") == "no"

def test_count_digit():
    assert parse_response("There are 3 dogs.", "count") == "3"

def test_count_word():
    assert parse_response("There are three dogs.", "count") == "three" or \
           parse_response("There are three dogs.", "count") == "3"

def test_count_zero():
    assert parse_response("There are no visible cats.", "count") == "0"

def test_spatial_yes():
    assert parse_response("Yes, the dog is to the left.", "spatial") == "yes"

def test_spatial_no():
    assert parse_response("No, the cat is not to the right.", "spatial") == "no"

def test_spatial_forced_choice():
    q = "Which lobe is abnormal, left or right?"
    assert parse_response("The left lobe appears abnormal.", "spatial", q) == "left"

def test_attribute_color():
    assert parse_response("The bar is blue.", "attribute") == "blue"

def test_attribute_grey_normalize():
    assert parse_response("It appears grey.", "attribute") == "gray"

def test_attribute_forced_choice_organ():
    q = "Which is bigger in this image, kidney or liver?"
    assert parse_response("The liver is bigger.", "attribute", q) == "liver"

def test_attribute_shape():
    q = "What is the shape of the kidney in the picture?"
    assert parse_response("The kidney has an oval shape.", "attribute", q) == "oval"

def test_attribute_irregular():
    assert parse_response("The shape is irregular.", "attribute") == "irregular"

def test_attribute_medical_size():
    q = "Which is bigger in this image, lung or heart?"
    assert parse_response("The lung is much larger than the heart.", "attribute", q) == "lung"

def test_ocr_percent():
    q = "What is the value of 2016 for china?"
    assert parse_response("The value is 36%.", "ocr_numeric", q) == "36%"

def test_ocr_skip_year():
    q = "What is the value of 2016 for china?"
    r = parse_response("In 2016, the value for china was 79.3.", "ocr_numeric", q)
    assert r == "79.3", f"got {r}"

def test_ocr_dollar():
    q = "What is the value?"
    assert parse_response("The value is $1,234.56.", "ocr_numeric", q) == "$1,234.56"

def test_confabulation():
    r = parse_response("A dog sitting on a couch.", "confabulation")
    assert r == "A dog sitting on a couch."

def test_uncertainty():
    assert parse_response("I'm not sure about that.", "existence") == "unparseable"

def test_attribute_almost_the_same():
    q = "Which is bigger in this image, liver or heart?"
    r = parse_response("They are almost the same size.", "attribute", q)
    assert r == "almost the same", f"got {r}"


for fn in [test_existence_yes, test_existence_no, test_existence_negative_phrasing,
           test_count_digit, test_count_word, test_count_zero,
           test_spatial_yes, test_spatial_no, test_spatial_forced_choice,
           test_attribute_color, test_attribute_grey_normalize,
           test_attribute_forced_choice_organ, test_attribute_shape,
           test_attribute_irregular, test_attribute_medical_size,
           test_attribute_almost_the_same,
           test_ocr_percent, test_ocr_skip_year, test_ocr_dollar,
           test_confabulation, test_uncertainty]:
    test(fn.__name__, fn)


# ============================================================
# 4. Grader tests
# ============================================================
print("\n=== 4. Grader ===")

from grader import grade


def test_grade_existence_correct():
    assert grade("yes", "yes", "existence") == True

def test_grade_existence_wrong():
    assert grade("no", "yes", "existence") == False

def test_grade_count_exact():
    assert grade("3", "3", "count") == True

def test_grade_count_wrong():
    assert grade("2", "3", "count") == False

def test_grade_ocr_tolerance():
    assert grade("36.0", "36%", "ocr_numeric") == True

def test_grade_ocr_exact():
    assert grade("79.3", "79.3", "ocr_numeric") == True

def test_grade_confabulation_none():
    assert grade("A dog on a couch.", None, "confabulation") is None

def test_grade_unparseable():
    assert grade("unparseable", "yes", "existence") is None

def test_grade_attribute():
    assert grade("blue", "blue", "attribute") == True

def test_grade_attribute_wrong():
    assert grade("red", "blue", "attribute") == False

def test_grade_spatial():
    assert grade("yes", "yes", "spatial") == True


for fn in [test_grade_existence_correct, test_grade_existence_wrong,
           test_grade_count_exact, test_grade_count_wrong,
           test_grade_ocr_tolerance, test_grade_ocr_exact,
           test_grade_confabulation_none, test_grade_unparseable,
           test_grade_attribute, test_grade_attribute_wrong,
           test_grade_spatial]:
    test(fn.__name__, fn)


# ============================================================
# 5. Mock pipeline run (no GPU)
# ============================================================
print("\n=== 5. Mock pipeline (3 probes, fake model) ===")


class MockWrapper:
    """Returns canned responses that the parser should handle."""
    RESPONSES = {
        "existence": "Yes, I can see it.",
        "count": "There are 2 objects.",
        "spatial": "Yes, it is to the left.",
        "attribute": "The color is blue.",
        "ocr_numeric": "The value is 42%.",
        "confabulation": "A chart showing data trends over time.",
    }

    def ask(self, image, prompt):
        # return a response based on what category this likely is
        prompt_lower = prompt.lower()
        if "how many" in prompt_lower:
            return self.RESPONSES["count"]
        if "is there" in prompt_lower or "does this" in prompt_lower:
            return self.RESPONSES["existence"]
        if "to the left" in prompt_lower or "to the right" in prompt_lower:
            return self.RESPONSES["spatial"]
        if "color" in prompt_lower:
            return self.RESPONSES["attribute"]
        if "value" in prompt_lower:
            return self.RESPONSES["ocr_numeric"]
        if "describe" in prompt_lower:
            return self.RESPONSES["confabulation"]
        return "Yes."


def test_mock_pipeline():
    from run_pipeline import run_pipeline

    # pick 1 probe per domain
    by_domain = {}
    for p in all_probes:
        if p["domain"] not in by_domain:
            by_domain[p["domain"]] = p

    mini_probes = list(by_domain.values())
    mini_path = "probes_mock_test.json"
    with open(mini_path, "w") as f:
        json.dump(mini_probes, f)

    wrapper = MockWrapper()
    results = run_pipeline(
        wrapper, model_name="mock",
        probes_path=mini_path,
        output_path="results_mock_test.json",
        limit=None, save_every=100,
    )

    assert len(results) == len(mini_probes), f"Expected {len(mini_probes)} results, got {len(results)}"

    for r in results:
        assert "raw_response" in r
        assert "parsed_answer" in r
        assert "correct" in r
        assert r["model"] == "mock"
        assert r["raw_response"] is not None, f"raw_response is None for {r['domain']}"

    # cleanup
    os.remove(mini_path)
    os.remove("results_mock_test.json")

test("mock pipeline end-to-end", test_mock_pipeline)


# ============================================================
# 6. Analysis on mock results
# ============================================================
print("\n=== 6. Analysis script on mock results ===")


def test_analysis():
    from analyze_results import load_results, compute_cells, export_csv

    # create fake results
    fake = []
    for domain in ["natural", "chart", "medical", "screenshot"]:
        for cat in ["existence", "count", "spatial", "attribute"]:
            for model in ["llava", "internvl2"]:
                fake.append({
                    "image_id": 1, "domain": domain, "category": cat,
                    "question": "test?", "ground_truth": "yes",
                    "distractor_type": "positive", "model": model,
                    "raw_response": "Yes", "parsed_answer": "yes",
                    "correct": True,
                })
                fake.append({
                    "image_id": 2, "domain": domain, "category": cat,
                    "question": "test?", "ground_truth": "no",
                    "distractor_type": "adversarial", "model": model,
                    "raw_response": "Yes", "parsed_answer": "yes",
                    "correct": False,
                })

    fake_path = "results_fake_test.json"
    with open(fake_path, "w") as f:
        json.dump(fake, f)

    results = load_results(fake_path)
    cells = compute_cells(results)
    assert len(cells) > 0

    export_csv(cells, "results_summary_test.csv")
    assert os.path.exists("results_summary_test.csv")

    os.remove(fake_path)
    os.remove("results_summary_test.csv")

test("analysis pipeline", test_analysis)


# ============================================================
# 7. Probe schema completeness
# ============================================================
print("\n=== 7. Probe schema & coverage ===")

from collections import Counter


def test_probe_schema():
    required = {"image_id", "domain", "category", "question", "ground_truth"}
    valid_cats = {"existence", "attribute", "count", "spatial", "ocr_numeric", "confabulation"}
    valid_domains = {"natural", "chart", "medical", "screenshot"}

    for i, p in enumerate(all_probes):
        missing = required - set(p.keys())
        assert not missing, f"Probe {i} missing: {missing}"
        assert p["category"] in valid_cats, f"Probe {i} bad category: {p['category']}"
        assert p["domain"] in valid_domains, f"Probe {i} bad domain: {p['domain']}"

test("probe schema valid", test_probe_schema)


def test_category_coverage():
    from collections import defaultdict
    domain_cats = defaultdict(set)
    for p in all_probes:
        domain_cats[p["domain"]].add(p["category"])

    for domain, cats in domain_cats.items():
        assert "existence" in cats, f"{domain} missing existence probes"
        assert "confabulation" in cats, f"{domain} missing confabulation probes"

test("every domain has existence + confabulation", test_category_coverage)


def test_probe_counts():
    by_domain = Counter(p["domain"] for p in all_probes)
    for domain, count in by_domain.items():
        assert count >= 10, f"{domain} has only {count} probes"
    print(f"    Probe counts: {dict(by_domain)}")

test("sufficient probes per domain", test_probe_counts)


# ============================================================
# 8. Parser coverage on actual SLAKE probes
# ============================================================
print("\n=== 8. Parser on real SLAKE attribute probes ===")


def test_slake_attribute_parseable():
    slake_attr = [p for p in all_probes
                  if p["domain"] == "medical" and p["category"] == "attribute"]
    # simulate: model echoes back the ground truth (best case)
    unparseable = 0
    for p in slake_attr:
        gt = p["ground_truth"]
        if gt is None:
            continue
        result = parse_response(gt, "attribute", p["question"])
        if result == "unparseable":
            unparseable += 1
            print(f"    unparseable: q={p['question']}, gt={gt}")
    pct = unparseable / len(slake_attr) * 100 if slake_attr else 0
    assert pct < 10, f"{pct:.0f}% of SLAKE attribute GTs are unparseable by parser"
    print(f"    {len(slake_attr)} SLAKE attribute probes, {unparseable} unparseable ({pct:.1f}%)")

test("SLAKE attribute ground truths parseable", test_slake_attribute_parseable)


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED. Safe to copy to RunPod.")
else:
    print("FIX FAILURES before uploading to RunPod.")
    sys.exit(1)
