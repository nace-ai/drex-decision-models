# Drex decision models

🏆 **Oct 14, 2026: Drex Max is out!** A multimodal 27B decision model that scores 67.59 on the public Decision Index 0.3, and it reads images too. [Model page](models/drex-max-v1) · [Weights](https://huggingface.co/nace-ai/drex-max-v1) · [Q8_0 GGUF](https://huggingface.co/nace-ai/drex-max-v1-Q8_0) · [Performance](models/drex-max-v1#performance)

🧠 **Oct 13, 2026: Drex v2.0 is out!** A 35B mixture-of-experts decision model that scores 61.25 on the public Decision Index 0.3.1, and it runs on llama.cpp and Ollama too. [Model page](models/drex-v2.0) · [Weights](https://huggingface.co/nace-ai/drex-v2.0) · [Q8_0 GGUF](https://huggingface.co/nace-ai/drex-v2.0-Q8_0) · [Performance](models/drex-v2.0#performance)

Open decision models from [Nace.AI](https://www.nace.ai/). Pass a `state` and named typed questions (yes/no, multiple choice or rating scale) and get a probability for every option back; nothing is generated. Drex v2.0, Drex v1.5 and Drex DLM serve the same `POST /v1/systemone` API as the hosted [Drex API](https://drex.nace.ai/docs) and run locally. Drex Max scores the same request format from Python (`python -m drex_max`) and also serves it on llama.cpp and Ollama.

[Hosted API docs](https://drex.nace.ai/docs) · [Agent skill](https://github.com/nace-ai/drex-agent-skill)

## Models

| Model | Backbone | Weights | llama.cpp and Ollama branch | License |
|---|---|---|---|---|
| [Drex Max](models/drex-max-v1) | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense 27B, multimodal) | [nace-ai/drex-max-v1](https://huggingface.co/nace-ai/drex-max-v1), [Q8_0 GGUF](https://huggingface.co/nace-ai/drex-max-v1-Q8_0) | `drex-max-v1` | [Nace.AI Open RAIL-M](models/drex-max-v1/LICENSE) |
| [Drex v2.0](models/drex-v2.0) | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B mixture of experts, about 3B active) | [nace-ai/drex-v2.0](https://huggingface.co/nace-ai/drex-v2.0), [Q8_0 GGUF](https://huggingface.co/nace-ai/drex-v2.0-Q8_0) | `drex-v2.0` | [Nace.AI Open RAIL-M](models/drex-v2.0/LICENSE) |
| [Drex v1.5](models/drex-v1.5) | [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) (distilled Qwen 3.5 9B) | [nace-ai/drex-v1.5](https://huggingface.co/nace-ai/drex-v1.5) | `drex-v1.5` | [Nace.AI Open RAIL-M](models/drex-v1.5/LICENSE) |
| [Drex DLM](models/drex-dlm) | [Efficient-DLM-8B](https://huggingface.co/nvidia/Efficient-DLM-8B) | [nace-ai/drex-dlm](https://huggingface.co/nace-ai/drex-dlm) | `edlm` (llama.cpp), `nace-edlm` (Ollama) | [CC BY-NC 4.0](models/drex-dlm/LICENSE.md) |

## Runners

Drex v2.0, Drex v1.5 and Drex DLM run with Python and answer the same `POST /v1/systemone` request; Drex Max runs with `python -m drex_max`. All four also run on llama.cpp and Ollama; for Drex Max they serve the text backbone.

| Runner | Hardware | Use it for |
|---|---|---|
| Python (`serve.py`) | NVIDIA GPU; Apple silicon for Drex DLM | The reference implementation, batched requests |
| [llama.cpp](https://github.com/nace-ai/llama.cpp) fork | CUDA, Metal; CPU for Drex v1.5; Metal and CPU for Drex v2.0 and Drex Max | A GGUF served by `llama-server`, including Q8_0 |
| [Ollama](https://github.com/nace-ai/ollama) fork | CUDA, Metal, CPU; Metal and CPU for Drex v2.0 and Drex Max | `ollama create` and a local daemon |

The commands below use Drex v1.5. Each model page has the exact steps, build presets and tested hardware.

## Python

```bash
git clone https://github.com/nace-ai/drex-decision-models.git
cd drex-decision-models
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf download nace-ai/drex-v1.5 --local-dir weights/drex-v1.5
python inference.py --model weights/drex-v1.5 --request examples/request.json
```

To serve it instead:

```bash
python serve.py --model weights/drex-v1.5 --port 8000
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' -d @examples/request.json
```

`GET /health` returns `{"status": "ok", "model": "drex-v1.5"}` once the weights are loaded. Replace `drex-v1.5` with any model above, for example `drex-v2.0`. To accept requests longer than 16,384 tokens, set `KEV_CONTEXT` ([Context length](docs/context-length.md)).

Checkpoint loading requires safetensors backbone weights and a `head.pt` containing
only tensors and primitive metadata. Python files and custom-code mappings bundled
with checkpoints are not executed; Drex DLM uses the implementation in this repository.
Checkpoints that require other custom Python implementations are unsupported.

## llama.cpp

Drex v2.0, Drex v1.5 and Drex DLM run on the llama.cpp fork. Build it, convert the weights to GGUF and serve them. Add `-DGGML_CUDA=ON` to the first `cmake` command on NVIDIA, and `-ngl 99` to `llama-server` to use the GPU.

```bash
git clone --branch drex-v2.0 --single-branch https://github.com/nace-ai/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build -j --target llama-server llama-quantize

python llama.cpp/convert_hf_to_gguf.py weights/drex-v1.5 --no-mtp --outtype bf16 \
  --outfile weights/drex-v1.5/drex-v1.5.gguf
llama.cpp/build/bin/llama-server -m weights/drex-v1.5/drex-v1.5.gguf \
  --embedding --pooling none -np 2 -c 32768 -b 16384 -ub 2048 --port 8000
```

The server takes the same request as `serve.py`. `llama-quantize` makes a Q8_0 GGUF; see the [model page](models/drex-v1.5/README.md#llamacpp).

## Ollama

The Nace fork of Ollama runs Drex v2.0, Drex v1.5 and Drex DLM as a `decision` capability and starts `llama-server` itself. Build the fork against the llama.cpp checkout above. These are the macOS commands; the [model page](models/drex-v1.5/README.md#ollama) has the CPU and CUDA presets.

```bash
git clone --branch drex-v2.0 --single-branch https://github.com/nace-ai/ollama.git
cd ollama
export OLLAMA_LLAMA_CPP_SOURCE="$PWD/../llama.cpp"
cmake -S llama/server --preset darwin
cmake --build build/llama-server-darwin --target llama-server --parallel 8
GOTOOLCHAIN=auto go build -trimpath -o ollama .
cd ..
```

Import the GGUF, start the daemon and leave it running:

```bash
cat > weights/drex-v1.5/Modelfile <<'EOF'
FROM ./drex-v1.5.gguf
CAPABILITY decision
PARAMETER num_ctx 16384
EOF
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_LLAMA_SERVER="$PWD/ollama/build/llama-server-darwin/bin/llama-server" ollama/ollama serve
```

In another terminal, import the model and send the example. Ollama needs `"model"` in the request, which the `jq` filter adds:

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama/ollama create drex-v1.5 -f weights/drex-v1.5/Modelfile
jq '. + {model: "drex-v1.5"}' examples/request.json | curl http://127.0.0.1:11434/v1/systemone \
  -H 'Content-Type: application/json' -d @-
```

## Docs

- [Request format](docs/request-format.md): fields, question types, responses and batches.
- [Context length](docs/context-length.md): default 16,384 tokens, up to 131,072 for Drex v1.5 and Drex v2.0, 32,768 for Drex DLM.

## License

Code: Apache-2.0, see [LICENSE](LICENSE). Model weights: see the License column in [Models](#models).

The Python runtime includes code adapted from [Kev](https://github.com/jaredpalmer/kev)
by Jared Palmer. See [NOTICE](NOTICE) and [upstream provenance](docs/upstream.md).
