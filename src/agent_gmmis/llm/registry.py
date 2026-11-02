"""Centralized Model Registry and Cluster Allocation Planner for Agent for GMMIS.

Defines declarative profiles for all supported on-premise (Mufasa A100 MIG) and cloud LLMs,
providing execution topology planning, vLLM command generation, and CLI inspection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelProfile:
    """Declarative specification of an LLM profile and its deployment constraints."""

    key: str                          # Short alias, e.g. "deepseek-r1-14b-awq"
    display_name: str                 # UI label, e.g. "DeepSeek-R1 Distill Qwen 14B (AWQ)"
    hf_repo_or_id: str                # Full Hugging Face repo ID or model name
    provider: str                     # "mufasa", "gemini", "ollama", "mock"
    family: str                       # "Qwen", "Llama", "Mistral", "DeepSeek", "GLM", "Moonshot", "Gemini"
    origin: str                       # "China", "USA", "Europe", "Local" (thesis metadata)
    vram_gb: float                    # Estimated weight footprint on 20GB MIG slice
    default_max_len_single: int       # Context window when running alone on 20GB GPU
    default_max_len_dual: int         # Context window when co-located with another model
    max_output_tokens: int = 2048     # Output token limit (4096 for reasoning models)
    is_reasoning: bool = False        # True for models with <think> reasoning traces
    needs_chat_template: Optional[str] = None  # e.g. "scripts/mufasa/mistral_chat_template.jinja"
    quantization_flag: Optional[str] = None    # e.g. "bitsandbytes" for BF16 models on 20GB MIG
    trust_remote_code: bool = True


# Built-in Catalog of Mufasa On-Premise Cached Models + Gemini Free Tier Models
MODEL_CATALOG: Dict[str, ModelProfile] = {
    "qwen2.5-7b-awq": ModelProfile(
        key="qwen2.5-7b-awq",
        display_name="Qwen 2.5 7B Instruct (AWQ)",
        hf_repo_or_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
        provider="mufasa",
        family="Qwen",
        origin="China",
        vram_gb=5.7,
        default_max_len_single=32768,
        default_max_len_dual=16384,
    ),
    "qwen2.5-14b-awq": ModelProfile(
        key="qwen2.5-14b-awq",
        display_name="Qwen 2.5 14B Instruct (AWQ)",
        hf_repo_or_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
        provider="mufasa",
        family="Qwen",
        origin="China",
        vram_gb=10.5,
        default_max_len_single=32768,
        default_max_len_dual=16384,
    ),
    "llama-3.1-8b-awq": ModelProfile(
        key="llama-3.1-8b-awq",
        display_name="Meta Llama 3.1 8B Instruct (AWQ)",
        hf_repo_or_id="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        provider="mufasa",
        family="Llama",
        origin="USA",
        vram_gb=5.5,
        default_max_len_single=32768,
        default_max_len_dual=16384,
    ),
    "mistral-7b-awq": ModelProfile(
        key="mistral-7b-awq",
        display_name="Mistral 7B Instruct v0.3 (AWQ)",
        hf_repo_or_id="solidrust/Mistral-7B-Instruct-v0.3-AWQ",
        provider="mufasa",
        family="Mistral",
        origin="Europe",
        vram_gb=4.2,
        default_max_len_single=32768,
        default_max_len_dual=16384,
        needs_chat_template="scripts/mufasa/mistral_chat_template.jinja",
    ),
    "mistral-7b-fp16": ModelProfile(
        key="mistral-7b-fp16",
        display_name="Mistral 7B Instruct v0.3 (FP16)",
        hf_repo_or_id="mistralai/Mistral-7B-Instruct-v0.3",
        provider="mufasa",
        family="Mistral",
        origin="Europe",
        vram_gb=14.5,
        default_max_len_single=24576,
        default_max_len_dual=16384,
        needs_chat_template="scripts/mufasa/mistral_chat_template.jinja",
    ),
    "codestral-22b-awq": ModelProfile(
        key="codestral-22b-awq",
        display_name="Codestral 22B v0.1 (AWQ)",
        hf_repo_or_id="TechxGenus/Codestral-22B-v0.1-AWQ",
        provider="mufasa",
        family="Mistral",
        origin="Europe",
        vram_gb=12.5,
        default_max_len_single=24576,
        default_max_len_dual=16384,
        needs_chat_template="scripts/mufasa/mistral_chat_template.jinja",
    ),
    "deepseek-r1-14b-awq": ModelProfile(
        key="deepseek-r1-14b-awq",
        display_name="DeepSeek-R1 Distill Qwen 14B (AWQ)",
        hf_repo_or_id="casperhansen/deepseek-r1-distill-qwen-14b-awq",
        provider="mufasa",
        family="DeepSeek",
        origin="China",
        vram_gb=9.5,
        default_max_len_single=24576,
        default_max_len_dual=16384,
        max_output_tokens=4096,
        is_reasoning=True,
    ),
    "glm-4-9b-chat": ModelProfile(
        key="glm-4-9b-chat",
        display_name="GLM-4 9B Chat (4-bit BnB)",
        hf_repo_or_id="THUDM/glm-4-9b-chat",
        provider="mufasa",
        family="GLM",
        origin="China",
        vram_gb=6.8,
        default_max_len_single=16384,
        default_max_len_dual=16384,
        quantization_flag="bitsandbytes",
        trust_remote_code=True,
    ),
    "moonlight-16b-a3b": ModelProfile(
        key="moonlight-16b-a3b",
        display_name="Moonlight 16B A3B Instruct (4-bit BnB)",
        hf_repo_or_id="moonshotai/Moonlight-16B-A3B-Instruct",
        provider="mufasa",
        family="Moonshot",
        origin="China",
        vram_gb=9.5,
        default_max_len_single=16384,
        default_max_len_dual=16384,
        quantization_flag="bitsandbytes",
        trust_remote_code=True,
    ),
    "gemini-3.1-flash-lite": ModelProfile(
        key="gemini-3.1-flash-lite",
        display_name="Google Gemini 3.1 Flash-Lite (Cloud)",
        hf_repo_or_id="gemini-3.1-flash-lite",
        provider="gemini",
        family="Gemini",
        origin="USA",
        vram_gb=0.0,
        default_max_len_single=1048576,
        default_max_len_dual=1048576,
    ),
    "gemini-3.5-flash-lite": ModelProfile(
        key="gemini-3.5-flash-lite",
        display_name="Google Gemini 3.5 Flash-Lite (Cloud)",
        hf_repo_or_id="gemini-3.5-flash-lite",
        provider="gemini",
        family="Gemini",
        origin="USA",
        vram_gb=0.0,
        default_max_len_single=1048576,
        default_max_len_dual=1048576,
    ),
    "gemini-2.5-flash": ModelProfile(
        key="gemini-2.5-flash",
        display_name="Google Gemini 2.5 Flash (Cloud)",
        hf_repo_or_id="gemini-2.5-flash",
        provider="gemini",
        family="Gemini",
        origin="USA",
        vram_gb=0.0,
        default_max_len_single=1048576,
        default_max_len_dual=1048576,
    ),
    "mock": ModelProfile(
        key="mock",
        display_name="Deterministic Offline Mock",
        hf_repo_or_id="mock",
        provider="mock",
        family="Mock",
        origin="Local",
        vram_gb=0.0,
        default_max_len_single=32768,
        default_max_len_dual=32768,
    ),
}


def resolve_model_profile(identifier: str, provider_hint: Optional[str] = None) -> ModelProfile:
    """
    Resolves a model profile from an alias key, full HF repo ID, or model name.

    If the identifier is unknown, constructs a fallback ModelProfile using conservative
    defaults, respecting any provider hint or auto-detecting cloud/mock providers.
    """
    raw = (identifier or "").strip().strip("'\"")
    if provider_hint:
        provider_hint = provider_hint.strip().strip("'\"")
    if not raw:
        return MODEL_CATALOG["gemini-3.5-flash-lite"]

    norm_key = raw.lower()

    # 1. Exact key match
    if norm_key in MODEL_CATALOG:
        return MODEL_CATALOG[norm_key]

    # 2. Match by hf_repo_or_id (case-insensitive)
    for prof in MODEL_CATALOG.values():
        if prof.hf_repo_or_id.lower() == norm_key:
            return prof

    # 3. Match by repo basename (e.g. "Qwen2.5-7B-Instruct-AWQ" matches "Qwen/Qwen2.5-7B-Instruct-AWQ")
    for prof in MODEL_CATALOG.values():
        basename = prof.hf_repo_or_id.split("/")[-1].lower()
        if basename == norm_key:
            return prof

    # 4. Fallback for custom or unlisted models
    provider = provider_hint
    family = "Custom"
    origin = "Unknown"
    vram_gb = 8.0
    is_reasoning = "r1" in norm_key or "reasoner" in norm_key or "reasoning" in norm_key
    max_output = 4096 if is_reasoning else 2048

    if norm_key.startswith("gemini") or provider_hint == "gemini":
        provider = "gemini"
        family = "Gemini"
        origin = "USA"
        vram_gb = 0.0
    elif norm_key == "mock" or provider_hint == "mock":
        provider = "mock"
        family = "Mock"
        origin = "Local"
        vram_gb = 0.0
    elif provider_hint == "ollama":
        provider = "ollama"
        origin = "Local"
    else:
        provider = provider or "mufasa"

    template = None
    if "mistral" in norm_key or "codestral" in norm_key:
        template = "scripts/mufasa/mistral_chat_template.jinja"

    return ModelProfile(
        key=norm_key.replace("/", "-"),
        display_name=raw,
        hf_repo_or_id=raw,
        provider=provider,
        family=family,
        origin=origin,
        vram_gb=vram_gb,
        default_max_len_single=16384,
        default_max_len_dual=16384,
        max_output_tokens=max_output,
        is_reasoning=is_reasoning,
        needs_chat_template=template,
        trust_remote_code=True,
    )


def build_vllm_serve_args(
    profile: ModelProfile,
    port: int,
    gpu_util: float,
    max_len: int,
) -> List[str]:
    """Constructs the command-line argument list for 'vllm serve'."""
    args = [
        profile.hf_repo_or_id,
        "--port", str(port),
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", f"{gpu_util:.2f}",
        "--enforce-eager",
    ]
    if profile.needs_chat_template:
        args.extend(["--chat-template", profile.needs_chat_template])
    if profile.quantization_flag == "bitsandbytes":
        args.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
    if profile.trust_remote_code:
        args.append("--trust-remote-code")
    return args


def plan_mufasa_allocation(
    model_a: str,
    model_b: Optional[str] = None,
    provider_a: Optional[str] = None,
    provider_b: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes optimal cluster resource allocation, execution topology, and vLLM parameters.

    Topologies:
      - "single": Single local vLLM server (port 8000, util 0.90) or cloud single agent.
      - "hybrid": One local vLLM server (port 8000, util 0.90) + Cloud/Mock Agent B.
      - "dual_simultaneous": Both local models fit in 20GB MIG (VRAM sum <= 12.5GB).
      - "dual_sequential": Models exceed 12.5GB combined VRAM; executed in 2 sequential stages.
    """
    prof_a = resolve_model_profile(model_a, provider_hint=provider_a)
    prof_b = resolve_model_profile(model_b, provider_hint=provider_b) if model_b else None

    # Case 1: Single Model
    if not prof_b or not model_b:
        if prof_a.provider == "mufasa":
            vllm_args_a = build_vllm_serve_args(prof_a, port=8000, gpu_util=0.90, max_len=prof_a.default_max_len_single)
            return {
                "topology": "single",
                "prof_a": prof_a,
                "prof_b": None,
                "util_a": 0.90,
                "util_b": 0.0,
                "port_a": 8000,
                "port_b": None,
                "max_len_a": prof_a.default_max_len_single,
                "max_len_b": 0,
                "vllm_args_a": vllm_args_a,
                "vllm_args_b": [],
                "needs_bitsandbytes": prof_a.quantization_flag == "bitsandbytes",
            }
        else:
            return {
                "topology": "single",
                "prof_a": prof_a,
                "prof_b": None,
                "util_a": 0.0,
                "util_b": 0.0,
                "port_a": None,
                "port_b": None,
                "max_len_a": prof_a.default_max_len_single,
                "max_len_b": 0,
                "vllm_args_a": [],
                "vllm_args_b": [],
                "needs_bitsandbytes": False,
            }

    # Case 2: Hybrid (One Mufasa, One Cloud/Mock)
    if prof_a.provider == "mufasa" and prof_b.provider != "mufasa":
        vllm_args_a = build_vllm_serve_args(prof_a, port=8000, gpu_util=0.90, max_len=prof_a.default_max_len_single)
        return {
            "topology": "hybrid",
            "prof_a": prof_a,
            "prof_b": prof_b,
            "util_a": 0.90,
            "util_b": 0.0,
            "port_a": 8000,
            "port_b": None,
            "max_len_a": prof_a.default_max_len_single,
            "max_len_b": prof_b.default_max_len_single,
            "vllm_args_a": vllm_args_a,
            "vllm_args_b": [],
            "needs_bitsandbytes": prof_a.quantization_flag == "bitsandbytes",
        }

    if prof_a.provider != "mufasa" and prof_b.provider == "mufasa":
        vllm_args_b = build_vllm_serve_args(prof_b, port=8001, gpu_util=0.90, max_len=prof_b.default_max_len_single)
        return {
            "topology": "hybrid",
            "prof_a": prof_a,
            "prof_b": prof_b,
            "util_a": 0.0,
            "util_b": 0.90,
            "port_a": None,
            "port_b": 8001,
            "max_len_a": prof_a.default_max_len_single,
            "max_len_b": prof_b.default_max_len_single,
            "vllm_args_a": [],
            "vllm_args_b": vllm_args_b,
            "needs_bitsandbytes": prof_b.quantization_flag == "bitsandbytes",
        }

    # Case 3: Cloud / Mock dual
    if prof_a.provider != "mufasa" and prof_b.provider != "mufasa":
        return {
            "topology": "dual_simultaneous",
            "prof_a": prof_a,
            "prof_b": prof_b,
            "util_a": 0.0,
            "util_b": 0.0,
            "port_a": None,
            "port_b": None,
            "max_len_a": prof_a.default_max_len_single,
            "max_len_b": prof_b.default_max_len_single,
            "vllm_args_a": [],
            "vllm_args_b": [],
            "needs_bitsandbytes": False,
        }

    # Case 4: Both on Mufasa GPU
    combined_vram = prof_a.vram_gb + prof_b.vram_gb
    needs_bnb = (prof_a.quantization_flag == "bitsandbytes") or (prof_b.quantization_flag == "bitsandbytes")

    # Threshold for simultaneous dual execution on 20GB MIG slice:
    # 12.5 GB model weights leaves >= 7.5 GB for KV-cache + CUDA context (total memory utilization <= 0.88)
    if combined_vram <= 12.5:
        # Dynamic memory allocation proportional to weight size, scaled to total 0.86
        ratio_a = prof_a.vram_gb / max(0.1, combined_vram)
        util_a = max(0.35, min(0.53, round(0.86 * ratio_a, 2)))
        util_b = round(0.86 - util_a, 2)

        vllm_args_a = build_vllm_serve_args(prof_a, port=8000, gpu_util=util_a, max_len=prof_a.default_max_len_dual)
        vllm_args_b = build_vllm_serve_args(prof_b, port=8001, gpu_util=util_b, max_len=prof_b.default_max_len_dual)

        return {
            "topology": "dual_simultaneous",
            "prof_a": prof_a,
            "prof_b": prof_b,
            "util_a": util_a,
            "util_b": util_b,
            "port_a": 8000,
            "port_b": 8001,
            "max_len_a": prof_a.default_max_len_dual,
            "max_len_b": prof_b.default_max_len_dual,
            "vllm_args_a": vllm_args_a,
            "vllm_args_b": vllm_args_b,
            "needs_bitsandbytes": needs_bnb,
        }
    else:
        # 2-Stage Sequential Dual Mode: each model runs alone with full 0.90 GPU slice
        vllm_args_a = build_vllm_serve_args(prof_a, port=8000, gpu_util=0.90, max_len=prof_a.default_max_len_single)
        vllm_args_b = build_vllm_serve_args(prof_b, port=8000, gpu_util=0.90, max_len=prof_b.default_max_len_single)

        return {
            "topology": "dual_sequential",
            "prof_a": prof_a,
            "prof_b": prof_b,
            "util_a": 0.90,
            "util_b": 0.90,
            "port_a": 8000,
            "port_b": 8000,
            "max_len_a": prof_a.default_max_len_single,
            "max_len_b": prof_b.default_max_len_single,
            "vllm_args_a": vllm_args_a,
            "vllm_args_b": vllm_args_b,
            "needs_bitsandbytes": needs_bnb,
        }


def format_bash_plan(plan: Dict[str, Any]) -> str:
    """Formats allocation plan as sourceable bash export commands."""
    prof_a = plan["prof_a"]
    prof_b = plan["prof_b"]
    vllm_a_str = " ".join(plan["vllm_args_a"])
    vllm_b_str = " ".join(plan["vllm_args_b"])

    lines = [
        f'TOPOLOGY="{plan["topology"]}"',
        f'MODEL_A_KEY="{prof_a.key}"',
        f'MODEL_A_REPO="{prof_a.hf_repo_or_id}"',
        f'PROVIDER_A="{prof_a.provider}"',
        f'PORT_A="{plan["port_a"] or 8000}"',
        f'MAX_LEN_A="{plan["max_len_a"]}"',
        f'IS_REASONING_A="{"true" if prof_a.is_reasoning else "false"}"',
        f'VLLM_ARGS_A="{vllm_a_str}"',
    ]

    if prof_b:
        lines.extend([
            f'MODEL_B_KEY="{prof_b.key}"',
            f'MODEL_B_REPO="{prof_b.hf_repo_or_id}"',
            f'PROVIDER_B="{prof_b.provider}"',
            f'PORT_B="{plan["port_b"] or 8001}"',
            f'MAX_LEN_B="{plan["max_len_b"]}"',
            f'IS_REASONING_B="{"true" if prof_b.is_reasoning else "false"}"',
            f'VLLM_ARGS_B="{vllm_b_str}"',
        ])
    else:
        lines.extend([
            'MODEL_B_KEY=""',
            'MODEL_B_REPO=""',
            'PROVIDER_B=""',
            'PORT_B=""',
            'MAX_LEN_B="0"',
            'IS_REASONING_B="false"',
            'VLLM_ARGS_B=""',
        ])

    lines.append(f'NEEDS_BITSANDBYTES="{"true" if plan["needs_bitsandbytes"] else "false"}"')
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Registry and Execution Topology Planner.")
    parser.add_argument("--list", action="store_true", help="List all registered models in catalog")
    parser.add_argument("--plan-bash", action="store_true", help="Output bash variables for eval in SLURM")
    parser.add_argument("--model-a", type=str, default="qwen2.5-7b-awq", help="Model A alias or HF repo ID")
    parser.add_argument("--model-b", type=str, default=None, help="Model B alias or HF repo ID")
    parser.add_argument("--provider-a", type=str, default=None, help="Provider for Model A")
    parser.add_argument("--provider-b", type=str, default=None, help="Provider for Model B")

    args = parser.parse_args()

    if args.list:
        print(f"{'Key':<22} | {'Provider':<8} | {'VRAM':<6} | {'Context':<7} | {'Display Name'}")
        print("-" * 75)
        for key, p in MODEL_CATALOG.items():
            print(f"{key:<22} | {p.provider:<8} | {p.vram_gb:<5.1f}G | {p.default_max_len_single:<7} | {p.display_name}")
        return

    clean_model_a = args.model_a.strip().strip("'\"") if args.model_a else "qwen2.5-7b-awq"
    clean_model_b = args.model_b.strip().strip("'\"") if args.model_b else None
    clean_provider_a = args.provider_a.strip().strip("'\"") if args.provider_a else None
    clean_provider_b = args.provider_b.strip().strip("'\"") if args.provider_b else None

    plan = plan_mufasa_allocation(
        model_a=clean_model_a,
        model_b=clean_model_b,
        provider_a=clean_provider_a,
        provider_b=clean_provider_b,
    )

    if args.plan_bash:
        print(format_bash_plan(plan))
    else:
        print(f"Topology: {plan['topology']}")
        print(f"Model A:  {plan['prof_a'].hf_repo_or_id} (Port: {plan['port_a']}, Util: {plan['util_a']})")
        if plan['prof_b']:
            print(f"Model B:  {plan['prof_b'].hf_repo_or_id} (Port: {plan['port_b']}, Util: {plan['util_b']})")
        print(f"Needs bitsandbytes: {plan['needs_bitsandbytes']}")


if __name__ == "__main__":
    main()

