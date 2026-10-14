# Drex Max

**A multimodal decision model: 67.59 on the public Decision Index 0.3.2, 6.3 points ahead of Drex v2.0 and 9.6 ahead of Jev 1.13.0.**

We introduce Drex Max from [Nace.AI](https://www.nace.ai/), a decision model that answers typed questions about a given context, in text, JSON and images. Pass the context as `state` ([request format](../../docs/request-format.md)) with one or more named questions, and get a probability for every option back in one forward pass. Nothing is generated. Drex Max is built on [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a dense 27B model, with a 255-way readout head, and is released under the Nace.AI Open RAIL-M license.

[Weights](https://huggingface.co/nace-ai/drex-max-v1) · [Drex v2.0](../drex-v2.0) · [Drex v1.5](../drex-v1.5)

It runs with Python on an NVIDIA GPU, and its text backbone runs on llama.cpp and Ollama on Apple silicon (Metal) and the CPU. The default limit is 16,384 tokens per request. `DREX_MAX_CONTEXT=N` or `--max-length N` raises it up to the 262,144-token window of the base model (see [context length](../../docs/context-length.md)). The bf16 weights take about 49 GB, and loading the backbone in 8-bit takes about 26 GB.

<details>
<summary>How it works</summary>

The backbone has 64 layers: 48 Gated DeltaNet linear-attention layers that stay left to right, and 16 full-attention layers (every fourth) that are non-causal, so every token can attend to the whole sequence. Images go through the native vision tower and are visible to every question on the request. A 255-way readout in `readout.safetensors` turns the last hidden state into probabilities over up to 255 options. The bundled loader applies the non-causal attention; a plain causal chat run does not reproduce the scores below.

</details>

## Performance

### Decision Index 0.3.2 (public only)

| Model | Public index | Raw | Knowledge & Reasoning | Language | Retrieval | Tools | Arts |
|---|---|---|---|---|---|---|---|
| **Drex Max** (27B) | **67.59** | 75.48 | 57.2 | 72.2 | 68.8 | 82.3 | 53.3 |
| Drex v2.0 (35B) | 61.25 | 70.55 | 47.9 | 65.3 | 63.0 | 76.5 | 53.9 |
| Jev 1.13.0 | 57.96 | 68.55 | 53.9 | 59.2 | 55.4 | 75.1 | 39.1 |

Public index only: the 37 public benchmarks in five areas, without the private tests, over 140,178 scored requests. Unanswered requests count as wrong. Area columns are chance-corrected skill × 100. The public text suite is the same as in Decision Index 0.3.1, so the Drex v2.0 and Jev rows are comparable.

<p><img src="assets/decision_index.png" alt="Decision Index" width="720"></p>

<p><img src="assets/areas.png" alt="Drex Max by area" width="720"></p>

<details>
<summary>JevBench (231 public items)</summary>

| Model | Accuracy | Easy | Standard | Hard |
|---|---|---|---|---|
| **Drex Max** | **92.2%** (213/231) | 100.0% | 100.0% | **83.8%** |
| Jev 1.13.0 | 86.6% (200/231) | 100.0% | 98.6% | 73.0% |
| Drex v2.0 | 85.7% (198/231) | 100.0% | 95.8% | 73.0% |

Choice = top option, yes/no = noul ≥ 0.5, ordinal = top level.

</details>

<details>
<summary>Decision Bench v4</summary>

Drex Max scores 94.30% (1010/1071).

</details>

<details>
<summary>Vision (400 held-out public items)</summary>

400 held-out items from the public Decision Index vision sets. Skill is chance-corrected accuracy. The private vision half is not scored here.

| | Accuracy | Skill |
|---|---|---|
| **Drex Max** | **71.5%** (286/400) | **61.9** |

| Benchmark | Items | Accuracy | Skill |
|---|---|---|---|
| CV-Bench | 80 | 93.8 | 88.9 |
| RealWorldQA | 80 | 82.5 | 73.2 |
| BLINK | 80 | 67.5 | 48.3 |
| MMMU-Pro vision | 80 | 57.5 | 51.7 |
| R-Bench-M | 80 | 56.3 | 47.5 |

<p><img src="assets/vision.png" alt="Vision hold-out" width="720"></p>

</details>

## Python

Download the weights, install the requirements and run a request. The example is [`examples/request.json`](../../examples/request.json).

```bash
hf download nace-ai/drex-max-v1 --local-dir weights/drex-max-v1
pip install -r models/drex-max-v1/requirements.txt
python -m drex_max --model weights/drex-max-v1 --request examples/request.json
```

Add `--8bit` to load the backbone in 8-bit (CUDA, about 26 GB). Add `images` with file paths or `data:image/` URLs to the request to pass images; each image can be up to about 4.2 MP. The command needs Python 3.12 or newer.

From Python:

```python
from drex_max.model import DecisionModel, answer

model = DecisionModel("weights/drex-max-v1")
question = {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {"billing": "Payments, charges, and refunds", "technical": "Bugs and outages", "other": "Anything else"},
}
row = {
    "state": {"ticket": "I was charged twice for the same order. Please refund the extra payment."},
    "question": question,
    # "images": ["receipt.jpg"],
}
print(answer(question, model.predict([row])[0]))
```

Response for `examples/request.json` from a bf16 run on an NVIDIA H100:

```json
{
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.9982,
      "probabilities": { "billing": 0.9988, "technical": 0.0003, "other": 0.0009 }
    },
    "refund": { "type": "noul", "noul": 0.9974 },
    "urgency": {
      "type": "score",
      "score": 1.6522,
      "legend": { "0": "Routine", "1": "Soon", "2": "Urgent" },
      "probabilities": { "0": 0.0152, "1": 0.3173, "2": 0.6674 },
      "confidence": 0.4783
    }
  }
}
```

<details>
<summary>Tested hardware</summary>

NVIDIA H100 80 GB with bf16 weights (48.6 GiB loaded), and NVIDIA L40S 48 GB with the backbone in 8-bit (25.9 GiB loaded). Both answered `examples/request.json` and an image request, and the 8-bit probabilities are within 0.03 of bf16 on the example.

</details>

## llama.cpp

Runs the text backbone on Apple silicon (Metal) and on the CPU; images need the Python path. Run these from the repository root:

```bash
git clone --branch drex-max-v1 --single-branch https://github.com/nace-ai/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build -j --target llama-server

python llama.cpp/convert_hf_to_gguf.py weights/drex-max-v1 --no-mtp --outtype q8_0 \
  --outfile weights/drex-max-v1/drex-max-v1-Q8_0.gguf

llama.cpp/build/bin/llama-server -m weights/drex-max-v1/drex-max-v1-Q8_0.gguf \
  -np 1 -c 16384 -b 16384 -ub 16384 --no-warmup --port 8000
```

The server answers `POST /v1/systemone` with the same request and response as the Python path, minus `images`. The Q8_0 GGUF is 27.3 GB. The conversion needs the bf16 weights, about 49 GB, on disk. To skip it, download the ready-made [Q8_0 GGUF](https://huggingface.co/nace-ai/drex-max-v1-Q8_0) (`hf download nace-ai/drex-max-v1-Q8_0 drex-max-v1-Q8_0.gguf --local-dir weights/drex-max-v1-Q8_0`) and pass its path to `llama-server`.

Drex Max reads each question as one whole prompt, so `-b` and `-ub` equal `-c` and one request holds up to that many tokens. For longer requests set `SYSTEMONE_CONTEXT=N` and use `-c N -b N -ub N` ([Context length](../../docs/context-length.md)).

<details>
<summary>Tested hardware</summary>

Apple M5 Pro (18-core CPU, 20-core GPU, 48 GB unified memory, macOS 26.5.2): the Q8_0 GGUF on Metal answers `examples/request.json` within 0.02 of the bf16 weights run with Python on an NVIDIA H100.

</details>

## Ollama

Use the Nace fork of Ollama with a `llama-server` built from the `drex-max-v1` branch above. Convert the GGUF first, as in the llama.cpp steps.

<details>
<summary>Build the fork</summary>

From the repository root:

```bash
git clone --branch drex-max-v1 --single-branch https://github.com/nace-ai/ollama.git
cd ollama
export OLLAMA_LLAMA_CPP_SOURCE="$PWD/../llama.cpp"
cmake -S llama/server --preset darwin
cmake --build build/llama-server-darwin --target llama-server --parallel 8
GOTOOLCHAIN=auto go build -trimpath -o ollama .
cd ..
```

For the CPU use the `cpu` preset and `build/llama-server-cpu` in place of `darwin` and `build/llama-server-darwin` here and below.

</details>

Write a `Modelfile` next to the GGUF:

```bash
cat > weights/drex-max-v1/Modelfile <<'EOF'
FROM ./drex-max-v1-Q8_0.gguf
CAPABILITY decision
PARAMETER num_ctx 16384
PARAMETER num_gpu 40
EOF
```

Start the daemon with the fork's binary and leave it running:

```bash
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_LLAMA_SERVER="$PWD/ollama/build/llama-server-darwin/bin/llama-server" ollama/ollama serve
```

In another terminal, import and call the model:

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama/ollama create drex-max-v1 -f weights/drex-max-v1/Modelfile
jq '. + {model: "drex-max-v1"}' examples/request.json | curl http://127.0.0.1:11434/v1/systemone \
  -H 'Content-Type: application/json' -d @-
```

Ollama needs `"model"` in the request, which the `jq` filter adds. `num_gpu 40` offloads 40 layers to Metal on a 48 GB Mac; use `PARAMETER num_gpu 0` for the CPU. The default is 16,384 tokens; for longer requests set `PARAMETER num_ctx N`, import again, and start the daemon with `SYSTEMONE_CONTEXT=N`.

<details>
<summary>Tested hardware</summary>

Through Ollama on the Apple M5 Pro above, the Q8_0 GGUF with `num_gpu 40` gives the same answers as `llama-server`.

</details>

## Files

| | |
|---|---|
| `model-*.safetensors`, `config.json` | Qwen3.8-27B backbone after the decision fine-tune (bf16) |
| `readout.safetensors` | The 255-way decision head |
| `decision_config.json` | Option codes, temperature and the non-causal attention setting |
| `code/drex_max/` | The loader: attention mask, readout and images |
| `inference.py` | Scores a request from a checkpoint directory |
