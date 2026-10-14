# Context length

A request (the state plus one question) is limited to 16,384 tokens by default. To accept longer ones, raise the limit for your runner. Larger requests return an error.

| Model | Default | Maximum |
|---|---|---|
| Drex Max | 16,384 | 262,144 |
| Drex v2.0 | 16,384 | 131,072 |
| Drex v1.5 | 16,384 | 131,072 |
| Drex DLM | 16,384 | 32,768 |

Below, `N` is the limit you want: up to 131,072 for Drex v2.0 and Drex v1.5 (and for Drex Max on llama-server and Ollama), and up to 32,768 for Drex DLM. A value above the model's maximum is rejected.

## Python

Drex Max uses `DREX_MAX_CONTEXT=N` (up to 262,144) or `--max-length N`; the flag wins. Longer requests need more GPU memory on top of the weights, so raise it in steps.

```bash
DREX_MAX_CONTEXT=N python -m drex_max --model weights/drex-max-v1 --request examples/request.json
```

The other models use `KEV_CONTEXT`:

```bash
KEV_CONTEXT=N python serve.py --model weights/<model> --port 8000
```

## llama-server

This section and the Ollama section cover every model. Drex Max reads each question as one prompt, so its text backbone uses the Drex DLM layout.

Set `SYSTEMONE_CONTEXT=N` and size `-c` for it. `-c` is shared by the slots, so each slot gets `-c` divided by `-np`.

| Model | Default (16,384) | Longer requests |
|---|---|---|
| Drex v2.0 and Drex v1.5 (`-np 2`) | `-c 32768 -b 16384 -ub 2048` | `SYSTEMONE_CONTEXT=N` with `-c` set to 2 x N, `-b 16384 -ub 2048` |
| Drex DLM and Drex Max (`-np 1`) | `-c 16384 -b 16384 -ub 16384` | `SYSTEMONE_CONTEXT=N` with `-c N -b N -ub N` |

For example, 131,072 tokens on Drex v2.0 or Drex v1.5 is `SYSTEMONE_CONTEXT=131072` with `-c 262144`.

## Ollama

Set `PARAMETER num_ctx N` in the Modelfile, run `ollama create` again, and start the daemon with `SYSTEMONE_CONTEXT=N`. Ollama sizes the slots from `num_ctx`.

## Individual limits

`KEV_SERVE_MAX_STATE`, `KEV_SERVE_MAX_BRANCH` and `KEV_SERVE_MAX_PACKED` (Python) and `SYSTEMONE_MAX_STATE`, `SYSTEMONE_MAX_BRANCH` and `SYSTEMONE_MAX_PACKED` (llama-server) override one limit each. They are not needed for the steps above.
