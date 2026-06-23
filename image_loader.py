import os
import json
from PIL import Image
from huggingface_hub import hf_hub_download


def load_coco_image(image_id, root="val2017"):
    path = os.path.join(root, f"{image_id:012d}.jpg")
    return Image.open(path).convert("RGB")


def load_chartqa_image(csv_path, root="chartqa_smoketest_images"):
    base = os.path.basename(csv_path).replace(".csv", ".png")
    return Image.open(os.path.join(root, base)).convert("RGB")


def load_slake_image(image_id):
    # cached locally by huggingface_hub after first download -- subsequent
    # calls for the same image_id are local disk reads, not network calls
    path = hf_hub_download(
        "Voxel51/SLAKE", filename=f"data/source_xmlab{image_id}.jpg", repo_type="dataset"
    )
    return Image.open(path).convert("RGB")


def load_screenqa_image(screen_id, root="screenqa_images"):
    path = os.path.join(root, f"{screen_id}.jpg")
    return Image.open(path).convert("RGB")


def load_image_for_probe(probe):
    """probe: one row from probes_all.json. All four domains now resolve
    to a local file read (SLAKE downloads once per unique image_id on
    first call, then reads from huggingface_hub's local cache after)."""
    domain = probe["domain"]

    if domain == "natural":
        return load_coco_image(probe["image_id"])
    if domain == "chart":
        return load_chartqa_image(probe["image_id"])
    if domain == "medical":
        return load_slake_image(probe["image_id"])
    if domain == "screenshot":
        return load_screenqa_image(probe["image_id"])

    raise ValueError(f"Unknown domain: {domain}")


if __name__ == "__main__":
    with open("probes_all.json") as f:
        all_probes = json.load(f)

    by_domain = {}
    for p in all_probes:
        by_domain.setdefault(p["domain"], p)

    for domain, probe in by_domain.items():
        img = load_image_for_probe(probe)
        print(domain, "->", img.size, img.mode)
