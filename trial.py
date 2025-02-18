import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "mistral-community/pixtral-12b"
# processor = AutoProcessor.from_pretrained(model_id)
# model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="cuda")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

print(model)

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
).to(model.device, torch.float16)
# inputs.pop("image_sizes", None)
# print(inputs)
# Generate

input_ids = inputs['input_ids'].to(model.device, dtype=torch.long)
attention_mask = inputs['attention_mask'].to(model.device, dtype=torch.long)
pixel_values = inputs['pixel_values'].to(model.device, dtype=torch.float32)

# generate_ids = model.generate(**inputs, max_new_tokens=36)
generate_ids = model.generate(input_ids=input_ids,
            #attention_mask=attention_mask,
            pixel_values=pixel_values, 
            max_new_tokens=36)
out = processor.batch_decode(generate_ids, skip_special_tokens=True)
print(out)