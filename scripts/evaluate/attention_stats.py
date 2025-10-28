import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt
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
        base = list(range(num_layers))
        while len(base) < 4:
            base.append(base[-1])
        return base
    mid1 = (num_layers - 1) // 2
    mid2 = num_layers // 2
    indices = [0, 1, max(0, mid1), max(0, mid2), num_layers - 2, num_layers - 1]
    seen = set()
    ordered: List[int] = []
    for i in indices:
        if 0 <= i < num_layers and i not in seen:
            seen.add(i)
            ordered.append(i)
    while len(ordered) < 6 and num_layers > 0:
        ordered.append(ordered[-1])
    return ordered


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
    for key in ("problem", "question", "query", "prompt", "instruction", "input"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v
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


def _build_cot_messages(question: str) -> List[Dict[str, str]]:
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    content = f"{question}{suffix}"
    return [{"role": "user", "content": content}]


def _chat_inputs(tokenizer, messages: List[Dict[str, str]], device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {"input_ids": input_ids.to(device)}
        return inputs
    except Exception:
        # Fallback: naive concatenation if no chat template
        joined = "\n\n".join(m.get("content", "") for m in messages)
        enc = tokenizer(joined, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}


def _entropy(p: torch.Tensor) -> float:
    # p assumed to be 1D, sum to 1
    eps = 1e-12
    p = torch.clamp(p, min=eps)
    return float((-p * torch.log(p)).sum().item())


def _topk_mass(p: torch.Tensor, k: int) -> float:
    k = min(k, p.numel())
    vals, _ = torch.topk(p, k)
    return float(vals.sum().item())


def _metrics_from_attn_vector(p: torch.Tensor, k_values: Iterable[int]) -> Dict[str, float]:
    L = p.numel()
    ent = _entropy(p)
    ent_norm = ent / math.log(L) if L > 1 else 0.0
    out: Dict[str, float] = {
        "entropy": ent,
        "entropy_norm": ent_norm,
    }
    for k in k_values:
        mass = _topk_mass(p, k)
        out[f"top{k}"] = mass
        out[f"top{k}_pct"] = mass * 100.0
    return out


def _render_layer_bar_charts(
    base_out: Path,
    per_layer_metrics: Dict[int, Dict[int, Dict[str, List[float]]]],
    chosen_layers: List[int],
    k_values: Tuple[int, ...],
) -> None:
    if 10 not in k_values:
        raise ValueError("Cannot generate mean_top10 charts because k_values does not include 10.")

    plot_dir = base_out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    levels = [1, 2, 3, 4, 5]
    padded_layers = list(chosen_layers)
    while len(padded_layers) < 6 and padded_layers:
        padded_layers.append(padded_layers[-1])
    if len(padded_layers) < 6:
        # Fallback if chosen_layers was empty for some reason.
        padded_layers = [0, 1, 2, 3, 4, 5]

    layer_groups = [
        ("early", padded_layers[0:2]),
        ("middle", padded_layers[2:4]),
        ("late", padded_layers[4:6]),
    ]

    for group_name, layers in layer_groups:
        for slot_idx, layer_idx in enumerate(layers, start=1):
            layer_level_metrics = per_layer_metrics.get(layer_idx)
            if not layer_level_metrics:
                continue

            mean_entropy: List[float] = []
            mean_top10: List[float] = []

            for lvl in levels:
                metrics_for_level = layer_level_metrics.get(lvl, {})
                ent_list = metrics_for_level.get("entropy", [])
                top10_list = metrics_for_level.get("top10", [])
                ent_mean = float(sum(ent_list) / len(ent_list)) if ent_list else float("nan")
                top10_mean = float(sum(top10_list) / len(top10_list)) if top10_list else float("nan")
                mean_entropy.append(ent_mean)
                mean_top10.append(top10_mean)

            indices = list(range(len(levels)))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar([i - width / 2 for i in indices], mean_entropy, width=width, label="mean_entropy")
            ax.bar([i + width / 2 for i in indices], mean_top10, width=width, label="mean_top10")
            ax.set_xticks(indices)
            ax.set_xticklabels([str(lvl) for lvl in levels])
            ax.set_xlabel("Level")
            ax.set_ylabel("Value")
            ax.set_title(f"{group_name.capitalize()} #{slot_idx} | layer {layer_idx}")
            ax.legend()
            fig.tight_layout()

            out_path = plot_dir / f"layer_{layer_idx}_{group_name}_slot{slot_idx}_metrics.png"
            fig.savefig(out_path)
            plt.close(fig)


def _forward_last_token_attn(model, inputs: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], int]:
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
    if outputs.attentions is None:
        raise RuntimeError("Model did not return attentions. Ensure output_attentions=True is supported.")
    seq_len = int(inputs["input_ids"].shape[1])
    # Return per-layer (H, L, L) and target index (last token)
    per_layer = [att[0] for att in outputs.attentions]  # (H, L, L)
    return per_layer, seq_len - 1


def _generate_one_with_attn(model, inputs: Dict[str, torch.Tensor]):
    gen = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_attentions=True,
        use_cache=True,
    )
    # gen.attentions: tuple length = generated tokens (1), each is tuple over layers
    return gen


def compute_attention_stats(
    model_path: str,
    data_file: str,
    output_dir: str,
    per_level: int = 5,
    seed: int = 42,
    target: str = "query_last",  # or "first_gen"
    k_values: Tuple[int, ...] = (5, 10),
    device: Optional[str] = None,
    analysis: str = "sample",  # sample | balanced | full | all
) -> None:
    dev = torch.device(device) if device else _auto_device()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(dev)
    model.eval()

    num_layers = _get_num_layers(model.config)
    chosen_layers = _select_layer_indices(num_layers)

    records = _load_jsonl(data_file)
    by_level = _group_by_level(records)

    rng = random.Random(seed)

    # Helper to run one pass with a picking strategy and output subdir name
    def run_pass(pass_name: str, pick_counts: Dict[int, int]) -> None:
        base_out = Path(output_dir) / pass_name
        base_out.mkdir(parents=True, exist_ok=True)

        # For aggregate summary across levels
        level_agg: Dict[int, List[Dict[str, Any]]] = {}
        per_layer_level_metrics: Optional[Dict[int, Dict[int, Dict[str, List[float]]]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(list))) if pass_name == "full" else None
        )

        for level in sorted(by_level.keys()):
            pool = [r for r in by_level[level] if _get_query(r)]
            if not pool:
                continue
            want = pick_counts.get(level, 0)
            if want <= 0:
                continue
            if len(pool) <= want:
                picked = list(pool)
            else:
                picked = rng.sample(pool, want)

            details_rows: List[Dict[str, Any]] = []
            for idx, rec in enumerate(picked):
                query = _get_query(rec)
                assert query is not None
                messages = _build_cot_messages(query)
                inputs = _chat_inputs(tokenizer, messages, dev)

                if target == "first_gen":
                    try:
                        gen = _generate_one_with_attn(model, inputs)
                        # gen.attentions -> length 1 list over generated tokens
                        step_attn = gen.attentions[0]  # tuple over layers
                        # For each layer: (B, H, tgt_len=1, src_len)
                        per_layer = [a[0] for a in step_attn]  # (H, 1, src_len)
                        # We will use the only target index 0
                        tgt_index = 0
                        _ = per_layer[0].shape[-1]
                    except Exception:
                        # Fallback to last query token attentions
                        per_layer, tgt_pos = _forward_last_token_attn(model, inputs)
                        tgt_index = tgt_pos
                else:  # query_last
                    per_layer, tgt_index = _forward_last_token_attn(model, inputs)

                # Compute metrics for selected layers
                sample_metrics: Dict[str, Any] = {
                    "level": level,
                    "sample_index": idx + 1,
                    "chosen_layers": chosen_layers,
                    "target": target,
                    "k_values": list(k_values),
                }
                per_layer_metrics: Dict[int, Dict[str, float]] = {}
                aggregated_vals: Dict[str, List[float]] = {}

                for li in chosen_layers:
                    att = per_layer[li]
                    if att.dim() == 3:
                        # (H, L, L) -> select last/query pos
                        vec = att[:, tgt_index, :].mean(dim=0)
                    elif att.dim() == 4:
                        # (H, 1, src_len)
                        vec = att[:, 0, :].mean(dim=0)
                    else:
                        raise RuntimeError(f"Unexpected attention tensor rank: {att.shape}")
                    # Ensure proper normalization (some impls may have tiny drift)
                    vec = vec / (vec.sum() + 1e-12)
                    metrics = _metrics_from_attn_vector(vec, k_values)
                    per_layer_metrics[li] = metrics
                    for k, v in metrics.items():
                        aggregated_vals.setdefault(k, []).append(v)

                # Compute simple averages across the six layers
                averages = {f"avg_{k}": float(sum(vs) / len(vs)) for k, vs in aggregated_vals.items()}

                sample_metrics["per_layer"] = per_layer_metrics
                sample_metrics.update(averages)

                # Persist per-sample metrics
                save_dir = base_out / f"level_{level}" / f"sample_{idx+1}"
                save_dir.mkdir(parents=True, exist_ok=True)
                with (save_dir / "stats.json").open("w", encoding="utf-8") as f:
                    json.dump(sample_metrics, f, ensure_ascii=False, indent=2)

                # Add rows for CSV details
                for li, m in per_layer_metrics.items():
                    row = {
                        "level": level,
                        "sample": idx + 1,
                        "layer": li,
                        "entropy": m["entropy"],
                        "entropy_norm": m["entropy_norm"],
                    }
                    for k in k_values:
                        row[f"top{k}"] = m[f"top{k}"]
                        row[f"top{k}_pct"] = m[f"top{k}_pct"]
                    details_rows.append(row)

                    if per_layer_level_metrics is not None:
                        per_layer_level_metrics[li][level]["entropy"].append(m["entropy"])
                        if 10 in k_values:
                            per_layer_level_metrics[li][level]["top10"].append(m["top10"])

            # Write per-level details CSV
            level_dir = base_out / f"level_{level}"
            level_dir.mkdir(parents=True, exist_ok=True)
            details_path = level_dir / "details.csv"
            if details_rows:
                fieldnames = list(details_rows[0].keys())
                with details_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(details_rows)

            # Aggregate by level (mean across samples and chosen layers)
            if details_rows:
                keys = [k for k in details_rows[0].keys() if k not in ("level", "sample", "layer")]
                vals: Dict[str, List[float]] = {k: [] for k in keys}
                for r in details_rows:
                    for k in keys:
                        vals[k].append(float(r[k]))
                level_stats = {
                    "level": level,
                    **{f"mean_{k}": float(sum(v) / len(v)) for k, v in vals.items()},
                    **{f"std_{k}": float((sum((x - (sum(vs) / len(vs))) ** 2 for x in vs) / max(1, len(vs) - 1)) ** 0.5) for k, vs in vals.items()},
                }
                level_agg.setdefault(level, []).append(level_stats)

        # Write overall aggregate CSV across levels
        agg_rows: List[Dict[str, Any]] = []
        for lvl, stats_list in sorted(level_agg.items()):
            merged: Dict[str, List[float]] = {}
            for s in stats_list:
                for k, v in s.items():
                    if k == "level":
                        continue
                    if isinstance(v, (int, float)):
                        merged.setdefault(k, []).append(float(v))
            row: Dict[str, Any] = {"level": lvl}
            for k, vs in merged.items():
                row[k] = float(sum(vs) / len(vs))
            agg_rows.append(row)

        if agg_rows:
            agg_path = base_out / "summary_levels.csv"
            fieldnames = ["level"] + [k for k in agg_rows[0].keys() if k != "level"]
            with agg_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(agg_rows)

        if per_layer_level_metrics:
            _render_layer_bar_charts(base_out, per_layer_level_metrics, chosen_layers, k_values)

    # Determine pick counts based on analysis mode
    sizes = {lvl: len([r for r in by_level[lvl] if _get_query(r)]) for lvl in by_level}
    min_size = min([s for s in sizes.values() if s > 0]) if any(sizes.values()) else 0

    def pickmap(mode: str) -> Dict[int, int]:
        if mode == "sample":
            return {lvl: min(per_level, sizes.get(lvl, 0)) for lvl in by_level}
        if mode == "balanced":
            return {lvl: min(min_size, sizes.get(lvl, 0)) for lvl in by_level}
        if mode == "full":
            return {lvl: sizes.get(lvl, 0) for lvl in by_level}
        raise ValueError(f"Unknown mode {mode}")

    if analysis == "all":
        for mode in ("sample", "balanced", "full"):
            run_pass(mode, pickmap(mode))
    else:
        run_pass(analysis, pickmap(analysis))


def main():
    parser = argparse.ArgumentParser(description="Compute attention entropy and top-k mass across MATH-500 difficulty levels using DeepSeek chat template.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to HF model (local dir)")
    parser.add_argument("--data-file", type=str, required=True, help="Path to MATH-500 test.jsonl")
    parser.add_argument("--output-dir", type=str, default=str(Path("outputs") / "attention_stats"))
    parser.add_argument("--per-level", type=int, default=5, help="Samples per difficulty level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--target", type=str, choices=["query_last", "first_gen"], default="query_last", help="Which attention vector to analyze")
    parser.add_argument("--k-values", type=str, default="5,10", help="Comma-separated top-k values, e.g., '5,10'")
    parser.add_argument("--device", type=str, default=None, help="Device string like 'cuda' or 'cpu'")
    parser.add_argument("--analysis", type=str, choices=["sample", "balanced", "full", "all"], default="sample", help="Sampling/aggregation strategy")

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not Path(args.data_file).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    k_vals: Tuple[int, ...] = tuple(int(x) for x in args.k_values.split(",") if x.strip())

    os.makedirs(args.output_dir, exist_ok=True)

    compute_attention_stats(
        model_path=args.model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
        per_level=args.per_level,
        seed=args.seed,
        target=args.target,
        k_values=k_vals,
        device=args.device,
        analysis=args.analysis,
    )


if __name__ == "__main__":
    main()
