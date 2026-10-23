# Drex v2.0 on Apple silicon

**A decision in under 200 ms, on your Mac.**

Drex v2.0 now runs fully on a Mac. The 4-bit build, [Drex v2.0 Q4_K_M](https://huggingface.co/nace-ai/drex-v2.0-Q4_K_M), is a 21 GB GGUF that fits any Apple silicon Mac with 36 GB of memory or more, and it runs in three engines: [Kapitsa](https://github.com/nace-ai/kapitsa), our open-source Mac engine, and the Nace forks of [llama.cpp](https://github.com/nace-ai/llama.cpp) and [Ollama](https://github.com/nace-ai/ollama). Nothing leaves the machine: setup, serving and every request run from the terminal. Open weights (Nace.AI Open RAIL-M) and an open-source runtime (Kapitsa, Apache-2.0).

[Q4_K_M GGUF](https://huggingface.co/nace-ai/drex-v2.0-Q4_K_M) · [Model page](README.md) · [Kapitsa](https://github.com/nace-ai/kapitsa) (release `v0.2.14` or later) · [llama.cpp fork](https://github.com/nace-ai/llama.cpp) (branch `drex-v2.0`) · [Ollama fork](https://github.com/nace-ai/ollama) (branch `drex-v2.0`)

## Benchmark

Apple M5 Pro, 48 GB. Median time per decision on a short support ticket, measured as client-side wall time over localhost HTTP, one request at a time.

| Engine | Per decision |
|---|---|
| **Kapitsa** | **113 ms** |
| llama.cpp | 248 ms |
| Ollama | 282 ms |

- A follow-up question on a document Kapitsa has already read takes 48 ms: Kapitsa keeps the document's state after the first read and only scores the new question.
- On an M3 Pro with 36 GB, llama.cpp takes 441 ms per decision.
- The 4-bit build gives the same answer as full precision on 97% of 2,710 questions.

## Hardware

Any Apple silicon Mac with 36 GB of memory or more (tested on 36 GB and 48 GB). The GGUF is 21 GB, and Kapitsa writes its own 24 GB copy of the weights on import, so keep room for both if you use Kapitsa.

## Download

Run every command on this page from the root of this repository.

```bash
hf download nace-ai/drex-v2.0-Q4_K_M drex-v2.0-Q4_K_M.gguf --local-dir weights/drex-v2.0-Q4_K_M
```

## Kapitsa

Kapitsa's `systemone` server answers `POST /v1/systemone` natively on Metal. It encodes the state once, scores each question from it, and keeps recently read states in memory, which is where the 48 ms follow-up comes from.

```bash
git clone --branch v0.2.14 --depth 1 https://github.com/nace-ai/kapitsa.git
cmake -S kapitsa -B kapitsa/build -DCMAKE_BUILD_TYPE=Release
cmake --build kapitsa/build -j 8 --target kapitsa systemone

kapitsa/build/kapitsa import weights/drex-v2.0-Q4_K_M/drex-v2.0-Q4_K_M.gguf --name drex-v2.0-q4
python3 kapitsa/tools/drex_pointer.py weights/drex-v2.0-Q4_K_M/drex-v2.0-Q4_K_M.gguf \
  ~/.cache/kapitsa/models/drex-v2.0-q4/pointer.bin

kapitsa/build/systemone --model ~/.cache/kapitsa/models/drex-v2.0-q4 --context 65536 --cache-mb 2048
```

Known issue: on Kapitsa v0.2.14, a single question whose instructions and option or criterion text together exceed 2,048 tokens crashes systemone instead of returning an error; keep long questions and long option lists short until the fix lands ([nace-ai/kapitsa#422](https://github.com/nace-ai/kapitsa/pull/422)).

`kapitsa import` writes the model to `~/.cache/kapitsa/models/drex-v2.0-q4`, and `drex_pointer.py` copies the pointer head from the GGUF into it, in f32. The server listens on `127.0.0.1:8000`. In another terminal:

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' -d @examples/request.json
```

`usage.input_tokens_details.cached_tokens` in the response shows how many tokens of the state came from the cache. `--cache-mb 0` serves every request cold. Kapitsa's [System One docs](https://github.com/nace-ai/kapitsa/blob/v0.2.14/docs/systemone.md) cover the flags and limits.

## llama.cpp

Build the `drex-v2.0` branch of the fork:

```bash
git clone --branch drex-v2.0 --single-branch https://github.com/nace-ai/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build -j --target llama-server

SYSTEMONE_CONTEXT=131072 llama.cpp/build/bin/llama-server \
  -m weights/drex-v2.0-Q4_K_M/drex-v2.0-Q4_K_M.gguf \
  --embedding --pooling none -np 2 -c 262144 -b 8192 -ub 8192 --no-repack -ngl 99 --port 8000
```

`-ngl 99` keeps every layer on Metal; `-ngl 0` runs entirely on the CPU. `-np 2` encodes the state once and shares it across questions, and `-c 262144` is the room two slots need for a 131,072-token state. The server takes the same request as Kapitsa and `serve.py`:

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' -d @examples/request.json
```

## Ollama

Build the Nace fork of Ollama against the llama.cpp checkout above, as on the [model page](README.md#ollama), then write a `Modelfile` next to the GGUF:

```bash
cat > weights/drex-v2.0-Q4_K_M/Modelfile <<'EOF'
FROM ./drex-v2.0-Q4_K_M.gguf
CAPABILITY decision
PARAMETER num_ctx 131072
PARAMETER num_gpu 99
PARAMETER parallel 2
EOF
```

Start the daemon with the fork's `llama-server` and leave it running:

```bash
OLLAMA_HOST=127.0.0.1:11434 SYSTEMONE_CONTEXT=131072 \
OLLAMA_LLAMA_SERVER="$PWD/ollama/build/llama-server-darwin/bin/llama-server" ollama/ollama serve
```

In another terminal, import the model and send the example. Ollama needs `"model"` in the request, which the `jq` filter adds:

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama/ollama create drex-v2.0 -f weights/drex-v2.0-Q4_K_M/Modelfile
jq '. + {model: "drex-v2.0"}' examples/request.json | curl http://127.0.0.1:11434/v1/systemone \
  -H 'Content-Type: application/json' -d @-
```

`num_gpu 99` keeps the 4-bit build on Metal; use `PARAMETER num_gpu 0` for the CPU.

## License

Weights: [Nace.AI Open RAIL-M](LICENSE). Kapitsa and the code in this repository: Apache-2.0. The llama.cpp and Ollama forks: MIT, as upstream.
