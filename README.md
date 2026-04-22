# codo

A fast, hackable CLI coding agent that works with local and cloud models.

## TL;DR

```bash
git clone https://github.com/ultrmgns/codo.git
cd codo
pip install -e .
codo
```

## vLLM setup

```bash
# In the codo REPL:
/config vllm_base_url=http://your-server:port/v1
/config vllm_api_key=your-key
/model vllm/your-model-name
```

Or via environment variables:

```bash
export VLLM_BASE_URL=http://your-server:port/v1
export VLLM_API_KEY=your-key
codo
```

## Other providers

Ollama, llama.cpp, OpenAI, Anthropic, Gemini, DeepSeek, and more are supported out of the box. Run `/model` to see all available providers and models.

## License

Apache 2.0
