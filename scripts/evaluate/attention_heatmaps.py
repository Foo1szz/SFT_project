import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None  # optional

import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_num_layers(config: Any) -> int:
    for attr in ("num_hidden_layers", "num_layers", "n_layer", "num_transformer_layers"):
        val = getattr(config, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    raise ValueError("Could not infer number of layers from model config.")


def _select_layer_indices(num_layers: int) -> List[int]:
    if num_layers <= 2:
        return list(range(num_layers))
    if num_layers <= 4:
        # duplicate some to still produce up to 4
        base = list(range(num_layers))
        while len(base) < 4:
            base.append(base[-1])
        return base
    # first two, middle two, last two
    mid1 = (num_layers - 1) // 2
    mid2 = num_layers // 2
    indices = [0, 1, max(0, mid1), max(0, mid2), num_layers - 2, num_layers - 1]
    # ensure within bounds and preserve order while removing duplicates
    seen = set()
    ordered: List[int] = []
    for i in indices:
        if 0 <= i < num_layers and i not in seen:
            seen.add(i)
            ordered.append(i)
    # if fewer than 6 due to dedup/low layers, pad with last layer index
    while len(ordered) < 6 and num_layers > 0:
        ordered.append(ordered[-1])
    return ordered


def _clean_token(t: str, max_len: int = 16) -> str:
    # Common cleanup for BPE/Byte tokens
    t = t.replace("Ġ", " ").replace("▁", " ")
    # Trim newlines and tabs for tick labels
    t = t.replace("\n", "\\n").replace("\t", "\\t")
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _get_query(record: Dict[str, Any]) -> Optional[str]:
    # Try common field names in MATH-like datasets
    for key in ("problem", "question", "query", "prompt", "instruction", "input"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v
    # Sometimes MATH-500 nests under 'query' as dict with 'problem'
    q = record.get("query")
    if isinstance(q, dict):
        for key in ("problem", "question", "prompt"):
            v = q.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return None


def _group_by_level(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    buckets: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for r in records:
        lvl = r.get("level")
        try:
            lvl_int = int(lvl)
        except Exception:
            continue
        if 1 <= lvl_int <= 5:
            buckets[lvl_int].append(r)
    return buckets


def _prepare_inputs(tokenizer, text: str, device: torch.device, max_length: Optional[int] = None):
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length or tokenizer.model_max_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _plot_attention(
    attn: torch.Tensor,
    tokens: List[str],
    layer_idx: int,
    out_path: Path,
    title_prefix: str,
) -> None:
    # attn shape: (num_heads, L, L) for single batch
    attn_avg = attn.mean(dim=0).detach().cpu().numpy()
    L = attn_avg.shape[0]
    if L != len(tokens):
        L = min(L, len(tokens))
        attn_avg = attn_avg[:L, :L]
        tokens = tokens[:L]

    plt.figure(figsize=(min(12, max(6, L * 0.25)), min(10, max(5, L * 0.25))))
    if sns is not None:
        sns.heatmap(attn_avg, cmap="viridis", cbar=True)
    else:
        plt.imshow(attn_avg, cmap="viridis")
        plt.colorbar()
    plt.title(f"{title_prefix} — Layer {layer_idx}")
    # keep ticks sparse for readability
    step = max(1, len(tokens) // 32)
    xticks = list(range(0, len(tokens), step))
    yticks = list(range(0, len(tokens), step))
    plt.xticks(xticks, [tokens[i] for i in xticks], rotation=90, fontsize=6)
    plt.yticks(yticks, [tokens[i] for i in yticks], fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_attention_maps(
    model_path: str,
    data_file: str,
    output_dir: str,
    per_level: int = 5,
    seed: int = 42,
    max_plot_tokens: int = 128,
    device: Optional[str] = None,
) -> None:
    dev = torch.device(device) if device else _auto_device()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, attn_implementation="eager")
    model.to(dev)
    model.eval()

    num_layers = _get_num_layers(model.config)
    chosen_layers = _select_layer_indices(num_layers)

    records = _load_jsonl(data_file)
    by_level = _group_by_level(records)

    rng = random.Random(seed)
    base_out = Path(output_dir)
    _ensure_dir(base_out)

    for level in sorted(by_level.keys()):
        pool = [r for r in by_level[level] if _get_query(r)]
        if not pool:
            continue
        picked = pool if len(pool) <= per_level else rng.sample(pool, per_level)
        for idx, rec in enumerate(picked):
            query = _get_query(rec)
            assert query is not None
            inputs = _prepare_inputs(tokenizer, query, dev, max_length=max_plot_tokens)

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)

            attentions = outputs.attentions  # tuple of (num_layers) each (B, H, L, L)
            if attentions is None:
                raise RuntimeError("Model did not return attentions. Ensure output_attentions=True is supported.")

            input_ids = inputs["input_ids"][0].tolist()
            tokens = [
                _clean_token(t)
                for t in tokenizer.convert_ids_to_tokens(input_ids)
            ]
            # Limit tokens for plotting if very long
            if len(tokens) > max_plot_tokens:
                tokens = tokens[:max_plot_tokens]

            save_dir = base_out / f"level_{level}" / f"sample_{idx+1}"
            _ensure_dir(save_dir)
            # Persist a small metadata file for traceability
            meta = {
                "level": level,
                "index": idx + 1,
                "query": query,
                "tokens": tokens,
                "chosen_layers": chosen_layers,
                "model_path": model_path,
            }
            with (save_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            for li in chosen_layers:
                # attentions[li]: (B, H, L, L)
                attn = attentions[li][0]
                out_path = save_dir / f"layer_{li:02d}.png"
                _plot_attention(attn, tokens, li, out_path, title_prefix=f"Level {level} Sample {idx+1}")


def main():
    parser = argparse.ArgumentParser(description="Generate attention heatmaps for sampled MATH-500 queries.")
    parser.add_argument("--model-path", type=str, default="/mnt/sharedata/ssd_large/common/LLMs/deepseek-math-7b-rl", help="Path to HF model (local dir)")
    parser.add_argument("--data-file", type=str, default="/mnt/sharedata/ssd_large/common/datasets/MATH-500/test.jsonl", help="Path to MATH-500 test.jsonl")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("outputs") / "attention_maps"),
        help="Directory to save heatmaps",
    )
    parser.add_argument("--per-level", type=int, default=5, help="Samples per difficulty level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--max-plot-tokens", type=int, default=128, help="Max tokens shown in heatmaps")
    parser.add_argument("--device", type=str, default=None, help="Device string like 'cuda' or 'cpu'")

    args = parser.parse_args()

    # Validate paths early
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not Path(args.data_file).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    os.makedirs(args.output_dir, exist_ok=True)

    generate_attention_maps(
        model_path=args.model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
        per_level=args.per_level,
        seed=args.seed,
        max_plot_tokens=args.max_plot_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()

