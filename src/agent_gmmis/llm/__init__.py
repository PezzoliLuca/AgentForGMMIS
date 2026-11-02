"""Universal LLM client layer supporting Gemini Free Tier, Ollama, Mufasa HPC, and Mocks."""

from agent_gmmis.llm.client import LLMClient, LLMConfig
from agent_gmmis.llm.registry import (
    MODEL_CATALOG,
    ModelProfile,
    build_vllm_serve_args,
    plan_mufasa_allocation,
    resolve_model_profile,
)

__all__ = [
    "LLMClient",
    "LLMConfig",
    "MODEL_CATALOG",
    "ModelProfile",
    "build_vllm_serve_args",
    "plan_mufasa_allocation",
    "resolve_model_profile",
]
