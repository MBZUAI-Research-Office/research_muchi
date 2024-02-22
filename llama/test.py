import subprocess

if __name__ == "__main__":
    subprocess.call(
        'python -m mlx_lm.generate --model ~/Llama-2-7b-chat-hf --prompt "hello"'
    )
