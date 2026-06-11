"""
Quick inference test to verify the Stage 1 fine-tuned model responds correctly
to Hindi/Indic prompts.

Usage (after Stage 1 merge):
    python scripts/08_test_inference.py \
        --model_path ./LLaMA-Factory/outputs/stage1_merged \
        --prompt "भारत में कितने राज्य हैं?"

For 4-bit quantized inference (low VRAM):
    python scripts/08_test_inference.py \
        --model_path ./LLaMA-Factory/outputs/stage1_merged \
        --load_in_4bit
"""

import argparse


def build_chat_prompt(tokenizer, user_message: str, system_message: str) -> str:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _resolve_model_class(model_path: str):
    """Return the right CausalLM class for this checkpoint."""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    if "omni" in model_type.lower():
        try:
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                Qwen3OmniMoeThinkerForCausalLM,
            )
            print(f"Detected omni model type '{model_type}' — using Qwen3OmniMoeThinkerForCausalLM")
            return Qwen3OmniMoeThinkerForCausalLM
        except ImportError:
            pass
    return AutoModelForCausalLM


def run_inference(model_path: str, prompt: str, system: str, load_in_4bit: bool, max_new_tokens: int):
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("Loading model in 4-bit quantized mode ...")
    else:
        print("Loading model in bfloat16 (A100 native) ...")

    model_class = _resolve_model_class(model_path)
    model = model_class.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.eval()

    full_prompt = build_chat_prompt(tokenizer, prompt, system)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    print(f"\nPrompt: {prompt}\n")
    print("Generating response ...")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"Response:\n{response}")
    print(f"{'='*60}")
    return response


def main():
    parser = argparse.ArgumentParser(description="Test Stage 1 fine-tuned model inference")
    parser.add_argument("--model_path", required=True, help="Path to merged HF model")
    parser.add_argument("--prompt", default="नमस्ते! आप कैसे हैं? भारत के बारे में कुछ बताइए।")
    parser.add_argument("--system", default="आप एक सहायक और विनम्र AI सहायक हैं जो हिंदी में बात करते हैं।")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit for low VRAM")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        prompt=args.prompt,
        system=args.system,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
