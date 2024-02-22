from mlx_lm import load, generate

if __name__ == "__main__":
    model, tokenizer = load("~/Llama-2-7b-chat-hf")
    response = generate(model, tokenizer, prompt="hello", verbose=True)
