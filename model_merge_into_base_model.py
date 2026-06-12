import torch
from transformers import Qwen3OmniMoeForConditionalGeneration
from peft import PeftModel

# Replace with the exact base model repository you used
base_model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"  
adapter_path = "./LLaMA-Factory/outputs/stage1_lora"

print("Loading Model.......")
# 1. Load the FULL base model (Thinker + Talker)
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)

print("Loading adapter weights.......")
# 2. Load the LoRA adapter into the base model
model_with_lora = PeftModel.from_pretrained(model, adapter_path)

print("Merging Adapter Weights........")
# 3. Merge and unload
merged_model = model_with_lora.merge_and_unload()

print("Saving Model.......")
# 4. Save the complete, merged Omni model
merged_model.save_pretrained("./qwen3-omni-full-merged")

print("Model Saved at: ./qwen3-omni-full-merged")