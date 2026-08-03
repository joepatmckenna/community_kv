"""CommunityKV evaluation execution over the public package API."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from community_kv import (
    CommunityKVAttention,
    CommunityKVConfig,
    CommunityKVRuntime,
    GraphAggregation,
    PartitionConfig,
)
from evals.datasets.dataset import Dataset
from evals.models import (
    ContextStrategy,
    HF_REVISION,
    load_causal_model,
    load_tokenizer,
    prepare_model,
)
from evals.resolutions import load_model_resolutions


RESULT_SCHEMA = "community-kv-eval-v1"


@dataclass(frozen=True, slots=True)
class EvalConfig:
    model: str
    output: Path
    resolutions: Path
    max_new_tokens: int = 128
    token_budget: int = 4096
    num_sink: int = 10
    lam: float = 0.5
    context_window: int = 131072
    context_strategy: ContextStrategy = ContextStrategy.TRUNCATE
    rope_factor: float | None = None
    max_samples: int | None = None
    device: str = "cuda:0"
    partition_devices: tuple[str, ...] = ()
    aggregation: GraphAggregation = GraphAggregation.PER_QUERY_HEAD

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if self.max_new_tokens < 2:
            raise ValueError("max_new_tokens must be at least 2")
        if self.token_budget <= 0 or self.token_budget % 64:
            raise ValueError("token_budget must be a positive multiple of 64")
        if self.num_sink < 0 or self.num_sink + 1 >= self.token_budget:
            raise ValueError("num_sink leaves no selected-token capacity")
        if not 0 <= self.lam <= 1:
            raise ValueError("lam must be in [0, 1]")
        if self.context_window <= self.max_new_tokens:
            raise ValueError("context_window must exceed max_new_tokens")
        if isinstance(self.context_strategy, str):
            object.__setattr__(
                self,
                "context_strategy",
                ContextStrategy(self.context_strategy),
            )
        if self.rope_factor is not None and (
            not math.isfinite(self.rope_factor) or self.rope_factor < 1
        ):
            raise ValueError("rope_factor must be finite and at least 1")
        if (
            self.context_strategy is ContextStrategy.TRUNCATE
            and self.rope_factor is not None
        ):
            raise ValueError("rope_factor requires context_strategy='extend'")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if isinstance(self.aggregation, str):
            object.__setattr__(
                self,
                "aggregation",
                GraphAggregation(self.aggregation),
            )


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class EvalRunner:
    def __init__(self, config: EvalConfig, dataset: Dataset) -> None:
        self.config = config
        self.dataset = dataset

    def _prepare_input(
        self,
        tokenizer: Any,
        sample: dict[str, Any],
        *,
        context_window: int | None = None,
        allow_truncation: bool = True,
    ) -> tuple[torch.Tensor, int, bool]:
        prompt_cap = (
            self.config.context_window if context_window is None else context_window
        ) - self.config.max_new_tokens
        input_ids = self.dataset.tokenize(tokenizer, sample)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("dataset tokenization must produce shape [1, sequence]")
        original_tokens = int(input_ids.shape[-1])
        truncated = False
        if original_tokens > prompt_cap:
            if not allow_truncation:
                raise ValueError(
                    f"prompt has {original_tokens} tokens but the configured "
                    f"context allows {prompt_cap}; increase --context-window"
                )
            truncated = self.dataset.fit_sample(
                sample,
                tokenizer,
                prompt_cap,
            )
            input_ids = self.dataset.tokenize(tokenizer, sample)
        if input_ids.shape[-1] > prompt_cap:
            raise ValueError(
                f"prompt has {input_ids.shape[-1]} tokens after fitting; "
                f"limit is {prompt_cap}"
            )
        if input_ids.shape[-1] < 18:
            raise ValueError("prompt is too short for CommunityKV graph construction")
        return input_ids, original_tokens, truncated

    def _run_sample(
        self,
        *,
        model: Any,
        tokenizer: Any,
        runtime: CommunityKVRuntime,
        resolutions: list[float],
        sample: dict[str, Any],
        context_window: int | None = None,
        allow_truncation: bool = True,
    ) -> dict[str, Any]:
        input_ids, original_tokens, truncated = self._prepare_input(
            tokenizer,
            sample,
            context_window=context_window,
            allow_truncation=allow_truncation,
        )
        input_ids = input_ids.to(self.config.device)
        prompt_tokens = int(input_ids.shape[-1])
        with torch.inference_mode():
            output = runtime.generate(
                model,
                input_ids=input_ids,
                resolutions=resolutions,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=False,
                output_scores=False,
            )
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 2
            or output.shape[0] != 1
        ):
            raise TypeError("runtime.generate must return one token sequence")
        generated = [
            int(token_id)
            for token_id in output[0, prompt_tokens:].detach().cpu().tolist()
        ]

        response = tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prediction = self.dataset.extract_answer(response, sample)
        score = float(self.dataset.score(response, sample))
        result = {
            "sample_id": self.dataset.sample_id(sample),
            "prompt_tokens": prompt_tokens,
            "original_prompt_tokens": original_tokens,
            "truncated": truncated,
            "generated_tokens": len(generated),
            "response": response,
            "prediction": prediction,
            "gold": self.dataset.gold(sample),
            "score": score,
        }
        del output, input_ids
        return result

    def run(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            raise RuntimeError("CommunityKV evaluation requires CUDA")
        device = torch.device(self.config.device)
        if device.type != "cuda":
            raise ValueError("model device must be CUDA")
        plan = prepare_model(
            self.config.model,
            required_tokens=self.config.context_window,
            rope_factor=self.config.rope_factor,
            context_strategy=self.config.context_strategy,
        )
        context_window = self.config.context_window
        if self.config.context_strategy is ContextStrategy.TRUNCATE:
            context_window = min(
                context_window,
                plan.truncation_context_window,
            )
        tokenizer = load_tokenizer(plan.profile)
        resolutions = load_model_resolutions(
            plan.profile.model_id,
            aggregation=self.config.aggregation,
            num_layers=plan.geometry.num_layers,
            fallback=1.0,
            path=self.config.resolutions,
        )
        samples = self.dataset.load_samples()
        if self.config.max_samples is not None:
            samples = samples[: self.config.max_samples]

        algorithm = CommunityKVConfig(
            token_budget=self.config.token_budget,
            num_sink=self.config.num_sink,
            lam=self.config.lam,
            aggregation=self.config.aggregation,
        )
        scheduling = PartitionConfig(
            devices=self.config.partition_devices,
        )
        runtime = CommunityKVRuntime(
            config=algorithm,
            partition=scheduling,
            num_layers=plan.geometry.num_layers,
            max_decode_tokens=self.config.max_new_tokens - 1,
        )
        implementation = CommunityKVAttention(runtime).register()
        model = load_causal_model(plan, device=device)
        model.config._attn_implementation = implementation

        payload: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "status": "running",
            "config": {
                "model": plan.profile.model_id,
                "model_profile": plan.profile.name,
                "model_revision": HF_REVISION,
                "dataset": self.dataset.lookup_name(),
                "max_new_tokens": self.config.max_new_tokens,
                "token_budget": self.config.token_budget,
                "num_sink": self.config.num_sink,
                "lam": self.config.lam,
                "context_window": context_window,
                "requested_context_window": self.config.context_window,
                "context_strategy": self.config.context_strategy.value,
                "max_samples": self.config.max_samples,
                "resolutions": str(self.config.resolutions),
                "aggregation": self.config.aggregation.value,
                "rope_factor": plan.rope_factor,
            },
            "model_geometry": asdict(plan.geometry),
            "samples": [],
            "summary": None,
        }
        atomic_json_dump(payload, self.config.output)
        try:
            for index, source_sample in enumerate(samples, start=1):
                sample = dict(source_sample)
                record = self._run_sample(
                    model=model,
                    tokenizer=tokenizer,
                    runtime=runtime,
                    resolutions=resolutions,
                    sample=sample,
                    context_window=context_window,
                    allow_truncation=(
                        self.config.context_strategy is ContextStrategy.TRUNCATE
                    ),
                )
                payload["samples"].append(record)
                atomic_json_dump(payload, self.config.output)
                print(
                    f"[{index}/{len(samples)}] {record['sample_id']}: "
                    f"score={record['score']:.4f}, "
                    f"tokens={record['prompt_tokens']}",
                    flush=True,
                )
            scores = [record["score"] for record in payload["samples"]]
            payload["status"] = "complete"
            payload["summary"] = {
                "sample_count": len(scores),
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
            }
            atomic_json_dump(payload, self.config.output)
            return payload
        except Exception as error:
            payload["status"] = "failed"
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            atomic_json_dump(payload, self.config.output)
            raise
        finally:
            runtime.close()


__all__ = [
    "EvalConfig",
    "EvalRunner",
    "RESULT_SCHEMA",
    "atomic_json_dump",
]
