# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
# Flash Attention temporarily disabled due to compatibility issues
try:
    from moellava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
    print("Flash Attention enabled successfully.")
except Exception as e:
    print(f"Flash Attention disabled due to: {e}")
    print("Falling back to standard attention (may actually improve performance as noted in README).")

from moellava.train.train import train

if __name__ == "__main__":
    train()
