# core/llm_engine.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMEngine:
    def __init__(self, model_path: str, device_map: str = "cpu"):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map=device_map,
            torch_dtype="auto"
        )

    def generate(self, messages: list, tools: list) -> str:
        # Build LLM input
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Extract only newly generated tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)