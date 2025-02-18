import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "mistral-community/pixtral-12b"
#processor = AutoProcessor.from_pretrained(model_id)
# target_model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
draft_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(draft_model.device, torch.float16)
inputs.pop("image_sizes", None)
print(inputs)
# Generate
generate_ids = draft_model.generate(**inputs, max_new_tokens=1)
processor.batch_decode(generate_ids, skip_special_tokens=True)