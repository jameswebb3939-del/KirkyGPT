from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = "outputs/run1"

# Load tokenizer and model
print(f"Loading model from {MODEL_DIR}...")

tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)

prompt = "### User:\nGive me 3 follow-up questions about learning Python.\n\n### Assistant:\n"
inputs = tok(prompt, return_tensors="pt")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

print("\n--- Model Output ---\n")
print(tok.decode(out[0], skip_special_tokens=True))
