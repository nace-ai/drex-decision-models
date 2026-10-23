# Drex v2.0

**Nace's flagship decision model: 61.25 on the public Decision Index 0.3.1, 3.3 points ahead of Jev 1.13.0 and 3.2 ahead of Drex v1.5.**

We introduce Drex v2.0 from [Nace.AI](https://www.nace.ai/), a decision model that answers typed questions about a given context. Pass the context as `state` ([request format](../../docs/request-format.md)) with one or more named questions, and get a probability for every option back in one forward pass. Nothing is generated. Drex v2.0 beats Jev head to head in the eight-game arena (60.9%) and on email classification (AUC 0.984 against 0.981), and it runs on your own hardware with Python, llama.cpp or Ollama. It is built on [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), a mixture of experts with 35B parameters and about 3B active per token, with a [Kev](https://github.com/jaredpalmer/kev) pointer head, and is released under the Nace.AI Open RAIL-M license.

> **On your Mac:** a 4-bit build (21 GB) runs on any Apple silicon Mac with 36 GB or more, in Kapitsa, llama.cpp or Ollama, at 113 ms per decision with Kapitsa on an M5 Pro. See [Drex v2.0 on Apple silicon](apple-silicon.md).

[Weights](https://huggingface.co/nace-ai/drex-v2.0) · [Q8_0 GGUF](https://huggingface.co/nace-ai/drex-v2.0-Q8_0) · [Q4_K_M GGUF](https://huggingface.co/nace-ai/drex-v2.0-Q4_K_M) · [Apple silicon](apple-silicon.md) · [Drex v1.5](../drex-v1.5) · [llama.cpp fork](https://github.com/nace-ai/llama.cpp) (branch `drex-v2.0`) · [Ollama fork](https://github.com/nace-ai/ollama) (branch `drex-v2.0`)

It runs with Python, llama.cpp or Ollama, and on a Mac also with [Kapitsa](apple-silicon.md#kapitsa). The default limit is 16,384 tokens per request. With Python, llama.cpp or Ollama it can be raised to 131,072 ([Context length](../../docs/context-length.md)); Kapitsa has its own limits, also 16,384 tokens by default ([System One docs](https://github.com/nace-ai/kapitsa/blob/v0.2.14/docs/systemone.md#limits-and-options)). The bf16 weights take about 70 GB, the Q8_0 GGUF about 37 GB and the Q4_K_M GGUF about 21 GB.

<details>
<summary>How it works</summary>

The backbone is a mixture-of-experts `Qwen3_5MoeForConditionalGeneration` with 40 layers, 256 experts and 8 active per token: 35B parameters in total, about 3B active per token. Only its language model is used; the vision tower is dropped on load. Each question is scored in one forward pass over the state plus that question, and the pointer head in `head.pt` turns the result into probabilities. Nothing is generated, so the sampling values in `generation_config.json` do not apply.

</details>

## Performance

### Decision Index 0.3.1 (public only)

| Model | Public index | Raw | Knowledge & Reasoning | Language | Retrieval | Tools | Arts |
|---|---|---|---|---|---|---|---|
| **Drex v2.0** (35B) | **61.25** | 70.55 | 47.9 | 65.3 | 63.0 | 76.5 | 53.9 |
| Drex v1.5 (9B) | 58.08 | 68.08 | 44.6 | 60.4 | 61.7 | 75.0 | 48.8 |
| Jev 1.13.0 | 57.96 | 68.55 | 53.9 | 59.2 | 55.4 | 75.1 | 39.1 |
| Bespoke Nimble 9B v3 | 57.19 | 68.38 | 41.0 | 60.7 | 55.5 | 84.2 | 44.0 |
| Cloudflare clef-flash | 56.15 | 66.53 | 53.0 | 47.2 | 52.2 | 81.9 | 48.4 |
| ezjev 4B s2 | 50.82 | 63.16 | 31.0 | 60.5 | 56.3 | 69.9 | 31.1 |

Public index only: the 37 public benchmarks in five areas, without the private tests. Area columns are chance-corrected skill × 100. The Drex rows are from our own runs of the official kit on the full public suite; the other rows are from the [leaderboard](https://huggingface.co/spaces/multimodalart/jev-decision-index).

<details>
<summary>JevBench (231 public items)</summary>

| Model | Accuracy | Easy | Standard | Hard |
|---|---|---|---|---|
| Jev 1.13.0 | **86.6%** (200/231) | 100.0% | **98.6%** | **73.0%** |
| **Drex v2.0** | 85.7% (198/231) | 100.0% | 95.8% | **73.0%** |

Choice = top option, yes/no = noul ≥ 0.5, ordinal = top level.

</details>

<details>
<summary>DecideBench v1.1 (400 contrastive decisions)</summary>

Published decision models only (chat LLMs omitted). Few-shot uses the benchmark's examples-in-instructions protocol. Not an official leaderboard submission.

| Entry | Accuracy | Pair accuracy |
|---|---|---|
| Jev (AI Space) | **98.0%** | **96.0%** |
| **Drex v2.0** (few-shot) | 96.5% | 93.5% |
| Drex v2.0 (zero-shot) | 96.3% | 92.5% |
| imajev-4b | 95.0% | 90.5% |
| Clef 27B | 94.8% | 89.5% |
| yev0-4b | 94.3% | 88.5% |
| Bespoke-Nimble-9B | 94.0% | 88.0% |
| TEV | 92.8% | 86.0% |
| Drex v1.5 | 90.0% | 80.5% |
| JevK5 v0.3 | 88.8% | 78.5% |

</details>

<details>
<summary>Email classification</summary>

Held-out addresses. Score is AUC; caught @ 5% is recall at a 5% false-positive rate.

|  | Drex v2.0 | Jev 1.13.0 | Drex v1.5 |
|---|---|---|---|
| AUC | **0.984** | 0.981 | 0.972 |
| Caught @ 5% FP | **89.4%** | 86.4% | 80.3% |

</details>

<details>
<summary>Game arena against Jev</summary>

Eight OpenSpiel games, 32 matches each (16 openings × both colours). Score = (wins + ½ draws) ÷ matches.

|  | W – D – L | Score |
|---|---|---|
| **Drex v2.0 vs Jev** | **132 – 48 – 76** | **60.9%** |
| Drex v1.5 vs Jev | 122 – 47 – 87 | 56.8% |

| Game | Drex v2.0 vs Jev | Drex v1.5 vs Jev |
|---|---|---|
| Chess | 13 – 19 – 0 (70.3%) | 9 – 21 – 2 (60.9%) |
| Four in a Row | 11 – 0 – 21 (34.4%) | 11 – 0 – 21 (34.4%) |
| Reversi | 17 – 0 – 15 (53.1%) | 15 – 1 – 16 (48.4%) |
| Checkers | 20 – 5 – 7 (70.3%) | 21 – 3 – 8 (70.3%) |
| Clobber | 20 – 0 – 12 (62.5%) | 17 – 0 – 15 (53.1%) |
| Lines of Action | 21 – 11 – 0 (82.8%) | 26 – 3 – 3 (85.9%) |
| Dots and Boxes | 15 – 0 – 17 (46.9%) | 18 – 0 – 14 (56.3%) |
| Nine Men's Morris | 15 – 13 – 4 (67.2%) | 5 – 19 – 8 (45.3%) |

</details>

## Python

Needs a CUDA GPU; memory use grows with document length. The server answers `POST /v1/systemone`.

```bash
hf download nace-ai/drex-v2.0 --local-dir weights/drex-v2.0
python serve.py --model weights/drex-v2.0 --port 8000
```

To accept requests longer than 16,384 tokens, set `KEV_CONTEXT` (up to 131,072); see [Context length](../../docs/context-length.md).

## llama.cpp

Runs on Apple silicon (Metal) and on the CPU. Run these from the repository root:

```bash
git clone --branch drex-v2.0 --single-branch https://github.com/nace-ai/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build -j --target llama-server

python llama.cpp/convert_hf_to_gguf.py weights/drex-v2.0 --no-mtp --outtype q8_0 \
  --outfile weights/drex-v2.0/drex-v2.0-Q8_0.gguf

llama.cpp/build/bin/llama-server -m weights/drex-v2.0/drex-v2.0-Q8_0.gguf \
  --embedding --pooling none -np 2 -c 32768 -b 16384 -ub 2048 -ngl 28 --port 8000
```

The server answers `POST /v1/systemone` with the same request and response as the Python server. The Q8_0 GGUF is 36.9 GB, so on a 48 GB Mac `-ngl 28` offloads 28 layers to Metal and runs the rest on the CPU; `-ngl 0` runs entirely on the CPU. The conversion needs the bf16 weights, about 70 GB, on disk. To skip it, download the ready-made [Q8_0 GGUF](https://huggingface.co/nace-ai/drex-v2.0-Q8_0) instead (`hf download nace-ai/drex-v2.0-Q8_0 drex-v2.0-Q8_0.gguf --local-dir weights/drex-v2.0-Q8_0`) and pass its path to `llama-server`.

On a Mac with 36 GB or more, the [Q4_K_M GGUF](https://huggingface.co/nace-ai/drex-v2.0-Q4_K_M) fits entirely on Metal; see [Drex v2.0 on Apple silicon](apple-silicon.md#llamacpp) for its command.

`-np 2` encodes the state once and shares it across questions. `-c` is split across the two slots, so this command accepts requests up to 16,384 tokens. For up to 131,072 tokens, set `SYSTEMONE_CONTEXT=131072` and use `-c 262144` ([Context length](../../docs/context-length.md)).

<details>
<summary>Tested hardware</summary>

Apple M5 Pro (18-core CPU, 20-core GPU, 48 GB unified memory, macOS 26.5.2): the Q8_0 GGUF gives the same answers on Metal with `-ngl 28` and on the CPU, with probabilities within 0.03 of each other.

</details>

## Ollama

Use the Nace fork of Ollama with a `llama-server` built from the `drex-v2.0` branch above. Convert the GGUF first, as in the llama.cpp steps.

<details>
<summary>Build the fork</summary>

From the repository root:

```bash
git clone --branch drex-v2.0 --single-branch https://github.com/nace-ai/ollama.git
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
cat > weights/drex-v2.0/Modelfile <<'EOF'
FROM ./drex-v2.0-Q8_0.gguf
CAPABILITY decision
PARAMETER num_ctx 16384
PARAMETER num_gpu 28
EOF
```

Start the daemon with the fork's binary and leave it running:

```bash
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_LLAMA_SERVER="$PWD/ollama/build/llama-server-darwin/bin/llama-server" ollama/ollama serve
```

In another terminal, import and call the model:

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama/ollama create drex-v2.0 -f weights/drex-v2.0/Modelfile
jq '. + {model: "drex-v2.0"}' examples/request.json | curl http://127.0.0.1:11434/v1/systemone \
  -H 'Content-Type: application/json' -d @-
```

Ollama needs `"model"` in the request, which the `jq` filter adds. `num_gpu 28` offloads 28 layers to Metal on a 48 GB Mac; use `PARAMETER num_gpu 0` for the CPU. The default is 16,384 tokens; for up to 131,072, set `PARAMETER num_ctx 131072`, import again, and start the daemon with `SYSTEMONE_CONTEXT=131072`.

<details>
<summary>Tested hardware</summary>

Through Ollama on the Apple M5 Pro above, the Q8_0 GGUF with `num_gpu 28` gives the same answers as `llama-server`.

</details>

## License

Nace.AI Open RAIL-M license, see [LICENSE](LICENSE).
