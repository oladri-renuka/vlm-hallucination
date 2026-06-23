"""
LLaVA-1.5-7B wrapper -- shared .ask(image, prompt) -> str interface.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


class LLaVAWrapper:
    def __init__(self, model_path="llava-hf/llava-1.5-7b-hf", device="cuda",
                 quantize=False):
        dtype = torch.float16
        load_kwargs = dict(
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        if quantize:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device

    def ask(self, image, prompt):
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=200, do_sample=False
            )
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
