# backend/app/rag_local_llm.py
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

def load_llm(model_name=LLM_MODEL, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Note: some Qwen models use custom code; we set trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto" if device=="cuda" else None)
    return tokenizer, model

def llm_summarize(query, passages, tokenizer, model, max_new_tokens=256):
    prompt = "You are an assistant. Given the query and passages, produce a concise summary (3-6 sentences) and bullet list of key contributions. Cite passages as [1],[2],...\n\n"
    prompt += f"Query: {query}\n\n"
    for i, p in enumerate(passages, 1):
        prompt += f"[{i}] {p[:1200]}\n\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)

    outputs = model.generate(
        **inputs,
        generation_config=gen_config,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Отделяем prompt от сгенерированного текста более надежно
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    if text.startswith(prompt_text):
        summary = text[len(prompt_text):].strip()
    else:
        summary = text.strip()

    return summary
