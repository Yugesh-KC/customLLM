from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import tiktoken
from llm import GPT, GPTConfig
import json




# --- Load model and tokenizer ---
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']



checkpoint = torch.load('model_weights.pt', map_location='cpu')

config = GPTConfig(**json.load(open('config.json')))
model = GPT(config)
model.load_state_dict(checkpoint)
model.eval()



app = FastAPI()


# --- Request schema ---
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int | None = 50


# --- Generation function ---
@torch.no_grad()
def generate(model, max_new_tokens, input='', temperature=1.0, top_k=None):
    start_token = torch.tensor([[eot]], device='cpu', dtype=torch.long)
    string_tokens = enc.encode(input, allowed_special=set())
    idx = torch.cat((start_token, torch.tensor(string_tokens, device='cpu', dtype=torch.long).unsqueeze(0)), dim=1)

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        if idx_next.item() == eot:
            break

    return enc.decode(idx.tolist()[0])


# --- FastAPI endpoint ---
@app.post("/generate")
def generate_text(req: GenerateRequest):
    output = generate(
        model=model,
        input=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    return {"generated_text": output}
