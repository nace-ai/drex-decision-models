"""Score a System One request with Drex Max.

    python -m drex_max --model weights/drex-max-v1
    python -m drex_max --model weights/drex-max-v1 --request examples/request.json
"""
import argparse
import json
from pathlib import Path

from drex_max.model import MAX_LENGTH, DecisionModel, answer

EXAMPLE = {
    "state": {"ticket": "I was charged twice for the same order. Please refund the extra payment."},
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments, charges, and refunds",
                "technical": "Bugs and outages",
                "other": "Anything else",
            },
        },
        "refund": {"type": "noul", "instructions": "Does the customer explicitly ask for a refund?"},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one System One request with Drex Max.")
    parser.add_argument("--model", required=True, help="Local checkpoint directory")
    parser.add_argument("--request", help="JSON file with one System One request; optional `images` list")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Longest request in tokens (up to 262,144; default 16,384 or DREX_MAX_CONTEXT)")
    parser.add_argument("--8bit", dest="load_in_8bit", action="store_true", help="Load the backbone in 8-bit (CUDA, bitsandbytes)")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text()) if args.request else EXAMPLE
    model = DecisionModel(args.model, device=args.device, load_in_8bit=args.load_in_8bit)
    images = request.get("images") or []
    out = {}
    for name, question in request["questions"].items():
        row = {"state": request["state"], "question": question}
        if images:
            row["images"] = images
        out[name] = answer(question, model.predict([row], max_length=args.max_length)[0])
    print(json.dumps({"answers": out}, indent=2))


if __name__ == "__main__":
    main()
