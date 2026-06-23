"""
InternVL2-8B wrapper -- shared .ask(image, prompt) -> str interface,
matching the LLaVA call pattern so the rest of the pipeline (parser,
grader) doesn't need to know which model it's talking to.

KNOWN RISK: InternVL2-8B's official example is pinned to transformers==4.37.2.
If your LLaVA setup already pulled a newer transformers, loading this model
may break in ways that look unrelated to your code -- a separate virtualenv
per model is the safe fix if you hit that.

KNOWN COST: this model dynamically splits each image into up to `max_num`
tiles + 1 thumbnail and runs all of them through the vision encoder per
call. Meaningfully heavier per-probe than LLaVA. If this OOMs on a Colab
T4 during the smoke test, drop max_num to 6 first before assuming
something's broken.

Requires: pip install torch torchvision transformers einops timm --break-system-packages
(flash-attn is optional and often fails to build cleanly -- omitted here;
add use_flash_attn=True to from_pretrained() yourself only if it's already
working in your environment, don't fight with installing it for this).
"""

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    cols = target_width // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            (i % cols + 1) * image_size,
            (i // cols + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image(image, input_size=448, max_num=12):
    """image: a PIL.Image, OR a path string (for ad-hoc testing)."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values


class InternVLWrapper:
    def __init__(self, model_path="OpenGVLab/InternVL2-8B", device="cuda", max_num=12,
                 quantize=False):
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
        if quantize:
            load_kwargs["load_in_4bit"] = True
        self.model = AutoModel.from_pretrained(model_path, **load_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        # greedy decoding -- matches the project's fp16/bf16 + temperature-0 decision (Section 8)
        self.generation_config = dict(max_new_tokens=200, do_sample=False)
        self.device = device
        self.max_num = max_num

    def ask(self, image, prompt):
        """image: a PIL.Image (e.g. from a HF dataset row). prompt: plain question text,
        no need to include '<image>' yourself -- this method adds it."""
        pixel_values = load_image(image, max_num=self.max_num).to(torch.bfloat16).to(self.device)
        question = f"<image>\n{prompt}"
        response, _ = self.model.chat(
            self.tokenizer, pixel_values, question, self.generation_config,
            history=None, return_history=True,
        )
        return response


if __name__ == "__main__":
    # quick smoke test -- same kind of check your teammate already ran on LLaVA
    import requests
    from io import BytesIO

    wrapper = InternVLWrapper(max_num=6)  # smaller tile count for a quick first check
    url = "https://images.unsplash.com/photo-1561037404-61cd46aa615b?w=600"
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    print(wrapper.ask(image, "Describe this image in detail."))
