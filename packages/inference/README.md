# Inference Package

Model serving, quantization, and deployment for AstroLLM.

## Serving Options

### Development: llama.cpp
- Runs quantized GGUF models on CPU or GPU
- Good for local testing and low-traffic serving
- Supports streaming, OpenAI-compatible API

### Production: vLLM
- Continuous batching for concurrent users
- PagedAttention for efficient GPU memory
- LoRA adapter hot-swapping for A/B testing
- OpenAI-compatible API

## Quantization

```bash
# Merge LoRA adapter with base model
python src/merge.py \
  --base Qwen/Qwen3.5-9B \
  --adapter models/latest/ \
  --output models/merged/

# Quantize to GGUF
python src/quantize.py \
  --input models/merged/ \
  --output models/astrollm-9b-q4_k_m.gguf \
  --quant q4_k_m
```

### Quantization Options

Sizes are for the fine-tuned Core model (Qwen3.5-9B; ~19.3 GB BF16 per HF) and approximate.

| Format | Size (9B, approx) | Quality | Speed | Use Case |
|--------|-------------------|---------|-------|----------|
| Q4_K_M | ≈ 5.5 GB | Good | Fast | Default recommendation |
| Q5_K_M | ≈ 6.5 GB | Better | Medium | Quality-sensitive |
| Q8_0 | ≈ 9.5 GB | Best quant | Slower | When quality matters most |
| BF16 | ≈ 19 GB | Full | GPU only | Production with GPU |

## Deployment

```bash
# Local (llama.cpp)
python src/serve.py --model models/astrollm-9b-q4_k_m.gguf --port 8080

# Production (vLLM)
python -m vllm.entrypoints.openai.api_server \
  --model models/merged/ \
  --port 8080 \
  --max-model-len 4096
```
