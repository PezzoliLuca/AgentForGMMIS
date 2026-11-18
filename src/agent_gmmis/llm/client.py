"""Universal LLM client layer supporting Native Google GenAI, OpenAI, Ollama, and Mocks."""

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env if present
load_dotenv()

T = TypeVar("T", bound=BaseModel)

# Thread-safe rate limiter to strictly respect Google AI Studio Free Tier limits (~15-20 RPM)
_gemini_pacing_lock = threading.Lock()
_last_gemini_timestamp = 0.0


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""

    provider: str = os.getenv("LLM_PROVIDER", "gemini")
    model: str = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url: Optional[str] = os.getenv("LLM_BASE_URL", None)
    temperature: float = 0.0


class LLMClient:
    """
    Model-agnostic client supporting:
    - Google Gemini Free Tier (native google-genai SDK, 0.00 €)
    - OpenAI / Local Ollama / Mufasa cluster (via OpenAI SDK)
    - Offline / Mock provider for deterministic unit testing
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

        from agent_gmmis.llm.registry import resolve_model_profile
        self.profile = resolve_model_profile(self.config.model, provider_hint=self.config.provider)
        if self.config.provider != "mock" and self.profile.hf_repo_or_id:
            self.config.model = self.profile.hf_repo_or_id

        if self.config.provider == "mock":
            self.is_mock = True
            self._gemini_client = None
            self._openai_client = None
        elif self.config.provider == "gemini":
            if not self.config.api_key:
                self.is_mock = True
                self._gemini_client = None
                self._openai_client = None
            else:
                from google import genai
                self.is_mock = False
                self._gemini_client = genai.Client(api_key=self.config.api_key)
                self._openai_client = None
        else:
            # mufasa / ollama / openai
            from openai import OpenAI
            self.is_mock = False
            self._gemini_client = None
            dummy_key = self.config.api_key or "mufasa-local"

            if self.config.base_url:
                base_url = self.config.base_url
            elif self.config.provider == "ollama":
                base_url = "http://localhost:11434/v1"
            elif self.config.provider == "mufasa":
                base_url = "http://localhost:8000/v1"
            else:
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

            self._openai_client = OpenAI(
                api_key=dummy_key,
                base_url=base_url,
            )

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """Calls the LLM enforcing a strict Pydantic response format."""
        if self.is_mock:
            # Deterministic mock response for offline testing and verification
            if response_model.__name__ == "ExtractedPaperData":
                from agent_gmmis.data.models import InclusionCriteriaCheck
                from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore
                mock_data = ExtractedPaperData(
                    paper_title="Segment Anything in Medical Images",
                    proposed_model_id="MedSAM",
                    authors_affiliations="University of Toronto, Vector Institute",
                    code_repository_url="https://github.com/bowang-lab/MedSAM",
                    architecture="SAM, Transformer with Convolutions",
                    visual_backbone="ViT-B",
                    millions_of_parameters=93.7,
                    number_of_gflops=82.0,
                    resources="20, A100 80GB, Nvidia",
                    resources_total_vram_gb=1600,
                    screening=InclusionCriteriaCheck(
                        is_3d=True,
                        is_radiological=True,
                        is_public_dataset=True,
                        has_dice_metric=True,
                    ),
                    extracted_scores=[
                        RawExtractedScore(dataset_name="BraTS", organ_or_task="Brain Tumor", dice_score=0.7563, table_or_section_ref="Table 1"),
                        RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen Multi-Organ", dice_score=0.8030, table_or_section_ref="Table 1"),
                        RawExtractedScore(dataset_name="AMOS", organ_or_task="Abdomen", dice_score=0.8019, table_or_section_ref="Table 2"),
                        RawExtractedScore(dataset_name="FLARE", organ_or_task="Liver/Spleen/Kidney", dice_score=0.7950, table_or_section_ref="Table 2"),
                        RawExtractedScore(dataset_name="KiTS", organ_or_task="Kidney and Tumor", dice_score=0.7954, table_or_section_ref="Table 3"),
                    ],
                )
                return mock_data  # type: ignore

            if response_model.__name__ == "ReinterrogationScoreResponse":
                import re
                from agent_gmmis.extraction.schemas import ReinterrogationScoreResponse
                m_ds = re.search(r"Dataset Name:\s*([^\n\r]+)", prompt) or re.search(r"dataset '([^']+)'", prompt)
                ds_name = m_ds.group(1).strip() if m_ds else "BraTS"
                return ReinterrogationScoreResponse(
                    dataset_name=ds_name,
                    organ_or_task="Overall",
                    is_valid_benchmark=True,
                    dice_score=0.8850,
                    prompt_type="bounding box",
                    table_or_section_ref="Table 1",
                    maintained_original=True,
                    rationale="Verified peak score in Table 1 under bounding box prompt.",
                )  # type: ignore

            if response_model.__name__ == "ReinterrogationOrganBreakdownResponse":
                import re
                from agent_gmmis.extraction.schemas import ReinterrogationOrganBreakdownResponse
                m_ds = re.search(r"On dataset '([^']+)'", prompt) or re.search(r"Dataset Name:\s*([^\n\r]+)", prompt)
                ds_name = m_ds.group(1).strip() if m_ds else "BTCV"
                return ReinterrogationOrganBreakdownResponse(
                    dataset_name=ds_name,
                    organ_scores=[],
                    table_or_section_ref="Table 1",
                    only_macro_average_reported=True,
                    rationale="Mock offline mode: no additional per-organ breakdown inferred.",
                )  # type: ignore

            if response_model.__name__ == "ScoutQueryPlan":
                from agent_gmmis.scouting.schemas import GeneratedSearchQuery, ScoutQueryPlan, ScoutTier
                return ScoutQueryPlan(
                    thought_process="Targeting emerging 3D foundation models (SAM2/Mamba/VLM) and volumetric radiological benchmarks.",
                    queries=[
                        GeneratedSearchQuery(
                            query_string="3D medical image segmentation foundation model SAM2 volumetric CT MRI",
                            target_tier=ScoutTier.TIER_1_GENERALIST,
                            rationale="Discover novel 3D promptable SAM2 adaptations for volumetric CT/MRI.",
                        ),
                        GeneratedSearchQuery(
                            query_string="text-prompted universal 3D medical segmentation TotalSegmentator AMOS",
                            target_tier=ScoutTier.TIER_1_GENERALIST,
                            rationale="Discover vision-language generalist 3D segmentation models.",
                        ),
                        GeneratedSearchQuery(
                            query_string="3D Mamba volumetric medical segmentation BraTS BTCV FLARE",
                            target_tier=ScoutTier.TIER_2_BENCHMARK_SOTA,
                            rationale="Track state-space SOTA 3D architectures on canonical benchmarks.",
                        ),
                    ],
                )  # type: ignore

            if response_model.__name__ == "ScoutScreeningResult":
                from agent_gmmis.scouting.schemas import (
                    ScoutScreeningResult,
                    ScoutThoughtProcess,
                    ScoutTier,
                    TriStateCriterion,
                )
                p_low = prompt.lower()
                # Heuristic mock evaluation for deterministic testing
                is_survey = any(w in p_low for w in ["a survey", "comprehensive review", "systematic review", "scoping review", "literature review"])
                is_2d_only = any(w in p_low for w in ["retinal fundus", "dermoscopy", "histopathology", "whole slide", "2d colonoscopy", "optical colonoscopy", "2d only"]) and not any(w in p_low for w in ["3d", "volumetric", "ct", "mri", "pet"])
                is_non_seg = any(w in p_low for w in ["image synthesis", "report generation", "pure classification", "without performing semantic segmentation", "super-resolution"]) and "segmentation model" not in p_low
                is_private_only = any(
                    w in p_low
                    for w in [
                        "exclusively on an internal",
                        "exclusively on an in-house",
                        "exclusively on a proprietary",
                        "solely on a proprietary",
                        "confidential private cohort",
                        "proprietary dataset only",
                    ]
                )

                if is_survey or is_2d_only or is_non_seg or is_private_only:
                    return ScoutScreeningResult(
                        thought_process=ScoutThoughtProcess(
                            task_and_novelty_analysis="Excluded: survey/review or non-segmentation task." if (is_survey or is_non_seg) else "Segmentation task detected.",
                            dimensionality_and_modality_analysis="Excluded: restricted to 2D or non-radiological optical modality." if is_2d_only else "3D/radiological modality.",
                            benchmark_and_paradigm_analysis="Excluded: private clinical cohort only." if is_private_only else "Out of scope.",
                        ),
                        reasoning="Mock screening identified explicit exclusion criteria in abstract.",
                        proposed_model_name=None,
                        is_novel_segmentation_model=not (is_survey or is_non_seg),
                        is_3d_volumetric=TriStateCriterion.NO if is_2d_only else TriStateCriterion.YES,
                        is_radiological=TriStateCriterion.NO if is_2d_only else TriStateCriterion.YES,
                        uses_public_benchmarks=TriStateCriterion.NO if is_private_only else TriStateCriterion.IMPLICIT_UNKNOWN,
                        tier=ScoutTier.REJECTED,
                        mentioned_datasets=[],
                        imaging_modalities=[],
                        discovered_external_leads=[],
                    )  # type: ignore

                is_generalist = any(w in p_low for w in ["foundation model", "generalist", "segment anything", "sam", "promptable", "universal", "zero-shot", "text-driven", "text-prompted", "multi-modality"])
                tier = ScoutTier.TIER_1_GENERALIST if is_generalist else ScoutTier.TIER_2_BENCHMARK_SOTA
                datasets = [ds for ds in ["AMOS", "BTCV", "BraTS", "TotalSegmentator", "FLARE", "KiTS", "LiTS"] if ds.lower() in p_low]
                return ScoutScreeningResult(
                    thought_process=ScoutThoughtProcess(
                        task_and_novelty_analysis="Proposes a novel medical image segmentation architecture.",
                        dimensionality_and_modality_analysis="Evaluates on 3D volumetric radiological imaging (CT/MRI/PET) or implicit multi-modal datasets.",
                        benchmark_and_paradigm_analysis=f"Classified as {tier.value} based on generalist/promptable vs task-specific cues.",
                    ),
                    reasoning="Valid 3D radiological segmentation model identified from abstract.",
                    proposed_model_name="MockModel3D",
                    is_novel_segmentation_model=True,
                    is_3d_volumetric=TriStateCriterion.YES if any(w in p_low for w in ["3d", "volumetric", "ct", "mri", "pet"]) else TriStateCriterion.IMPLICIT_UNKNOWN,
                    is_radiological=TriStateCriterion.YES if any(w in p_low for w in ["ct", "mri", "pet", "ultrasound"]) else TriStateCriterion.IMPLICIT_UNKNOWN,
                    uses_public_benchmarks=TriStateCriterion.YES if datasets or "public" in p_low or "benchmark" in p_low else TriStateCriterion.IMPLICIT_UNKNOWN,
                    tier=tier,
                    mentioned_datasets=datasets,
                    imaging_modalities=[m for m in ["CT", "MRI", "PET"] if m.lower() in p_low],
                    discovered_external_leads=[],
                )  # type: ignore

            return response_model.model_construct()

        # Native Google GenAI Provider
        if self._gemini_client is not None:
            from google.genai import types
            schema_json = json.dumps(response_model.model_json_schema(), indent=2)
            augmented_system = f"{system_prompt or ''}\n\nYou MUST respond with a valid JSON object strictly conforming to this JSON Schema:\n{schema_json}"

            config_obj = types.GenerateContentConfig(
                response_mime_type="application/json",
                system_instruction=augmented_system,
                temperature=self.config.temperature,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

            import time
            response = None
            max_retries = 8
            backoff = 2.0
            primary_model = self.config.model
            current_model = primary_model
            fallback_models = [primary_model, "gemini-flash-lite-latest"]
            # Deduplicate while preserving order
            fallback_models = list(dict.fromkeys(m for m in fallback_models if m))
            for attempt in range(max_retries):
                try:
                    global _gemini_pacing_lock, _last_gemini_timestamp
                    with _gemini_pacing_lock:
                        now = time.time()
                        elapsed = now - _last_gemini_timestamp
                        if elapsed < 3.0:
                            time.sleep(3.0 - elapsed)
                        _last_gemini_timestamp = time.time()

                    response = self._gemini_client.models.generate_content(
                        model=current_model,
                        contents=prompt,
                        config=config_obj,
                    )
                    break
                except Exception as exc:
                    err_msg = str(exc)
                    if any(x in err_msg for x in ["503", "UNAVAILABLE", "high demand", "429", "RESOURCE_EXHAUSTED", "Quota exceeded", "404", "NOT_FOUND"]):
                        if attempt < max_retries - 1:
                            current_model = fallback_models[(attempt + 1) % len(fallback_models)]
                            print(f"   ⚠️ [Google AI Studio API Retry] Switching to model '{current_model}'...")
                            if "404" not in err_msg and "NOT_FOUND" not in err_msg:
                                print(f"   ⚠️ [Google AI Studio API Retry] Retrying {current_model} in {backoff:.1f}s (attempt {attempt+1}/{max_retries})...")
                                time.sleep(backoff)
                                backoff = min(backoff * 1.5, 20.0)
                            continue
                    raise exc
            import re
            raw_text = response.text or "{}"
            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            # Extract JSON substring if wrapped in markdown code blocks
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            return response_model.model_validate_json(raw_text)

        # OpenAI-compatible Provider (Ollama / Mufasa / OpenAI)
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Calculate realistic output token budget for structured JSON response:
        # Granular per-organ extraction can emit 30+ RawExtractedScore rows across multi-organ benchmarks.
        resp_name = getattr(response_model, "__name__", "")
        is_reasoning = bool(getattr(self, "profile", None) and self.profile.is_reasoning)
        if is_reasoning:
            max_needed_output = 8192 if resp_name == "ExtractedPaperData" else 3072
        else:
            max_needed_output = 4096 if resp_name == "ExtractedPaperData" else 1536

        # Clean, inline and compact schema representation to preserve context tokens and eliminate $defs
        raw_schema = response_model.model_json_schema()
        defs = raw_schema.get("$defs", {})

        def _inline_and_compact(d: Any) -> Any:
            if isinstance(d, dict):
                if "$ref" in d:
                    ref_key = d["$ref"].split("/")[-1]
                    if ref_key in defs:
                        return _inline_and_compact(defs[ref_key])
                return {k: _inline_and_compact(v) for k, v in d.items() if k not in ("title", "default", "$defs")}
            elif isinstance(d, list):
                return [_inline_and_compact(v) for v in d]
            return d

        compact_schema = _inline_and_compact(raw_schema)
        compact_schema.pop("$defs", None)
        schema_json = json.dumps(compact_schema, separators=(",", ":"))

        total_chars = len(prompt) + (len(system_prompt) if system_prompt else 0) + len(schema_json)
        if self.config.provider == "mufasa":
            default_mufasa_ctx = getattr(getattr(self, "profile", None), "default_max_len_dual", 24576) or 24576
            context_limit = int(os.getenv("MUFASA_MAX_MODEL_LEN", str(default_mufasa_ctx)))
        elif self.config.provider == "ollama":
            context_limit = 16384
        else:
            context_limit = 32768

        # Conservative character-to-token estimation (numeric tables tokenize denser than prose)
        chars_per_token = 1.85 if context_limit <= 16384 else 2.2
        approx_tokens = int(total_chars / chars_per_token) + 800

        # Guarantee at least max_needed_output tokens of generation headroom by trimming prompt if needed
        required_headroom = max_needed_output + 800
        if approx_tokens > context_limit - required_headroom and len(prompt) > 4000:
            excess_tokens = approx_tokens - (context_limit - required_headroom)
            excess_chars = int(excess_tokens * chars_per_token) + 800

            # Avoid cutting the middle where comparative tables are located!
            # If the prompt contains the manuscript content header, trim trailing content (ablation/appendices)
            marker = "=== MANUSCRIPT CONTENT"
            if marker in prompt:
                m_idx = prompt.find(marker)
                target_len = max(m_idx + 2000, len(prompt) - excess_chars)
                prompt = prompt[:target_len] + "\n\n[... trailing manuscript text trimmed to respect context limit ...]\n"
            else:
                mid = len(prompt) // 2
                half_trim = min(mid - 500, max(500, excess_chars // 2))
                prompt = prompt[: mid - half_trim] + "\n\n[... middle text trimmed to respect context limit ...]\n\n" + prompt[mid + half_trim :]

            messages[-1]["content"] = prompt
            total_chars = len(prompt) + (len(system_prompt) if system_prompt else 0) + len(schema_json)
            approx_tokens = int(total_chars / chars_per_token) + 800

        safe_max_tokens = min(max_needed_output, max(800, context_limit - approx_tokens - 500))

        # Direct structured completion for vLLM / Ollama (avoids slow redundant beta.parse timeout)
        use_beta_parse = (not is_reasoning) and (self.config.provider not in ("mufasa", "ollama"))

        if use_beta_parse:
            try:
                completion = self._openai_client.beta.chat.completions.parse(
                    model=self.config.model,
                    messages=messages,
                    response_format=response_model,
                    temperature=self.config.temperature,
                    max_tokens=safe_max_tokens,
                )
                parsed = completion.choices[0].message.parsed
                if parsed is not None:
                    return parsed
            except Exception:
                pass

        top_properties = list(raw_schema.get("properties", {}).keys())
        top_keys_hint = ", ".join(f'"{p}"' for p in top_properties) if top_properties else ""
        augmented_messages = [dict(m) for m in messages]
        augmented_messages[0]["content"] += (
            "\n\nCRITICAL INSTRUCTIONS FOR STRUCTURED JSON OUTPUT:\n"
            "1. You MUST respond ONLY with a raw, valid JSON object.\n"
            "2. Do NOT echo or output schema definitions, do NOT output '$defs', and do NOT output any conversational text or markdown explanation.\n"
            f"3. Your output JSON MUST begin with '{{' and contain the instance data with top-level keys: [{top_keys_hint}].\n"
            f"4. Target JSON Schema:\n{schema_json}"
        )

        call_kwargs = {
            "model": self.config.model,
            "messages": augmented_messages,
            "temperature": self.config.temperature,
            "max_tokens": safe_max_tokens,
        }
        if not is_reasoning:
            call_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._openai_client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            # Automatic recovery: if vLLM throws 400 context limit error, trim manuscript and retry
            err_str = str(exc)
            if "maximum context length is" in err_str:
                user_content = augmented_messages[-1]["content"]
                trim_chars = 12000
                if len(user_content) > trim_chars + 2000:
                    marker = "=== MANUSCRIPT CONTENT"
                    if marker in user_content:
                        m_idx = user_content.find(marker)
                        target_len = max(m_idx + 2000, len(user_content) - trim_chars)
                        augmented_messages[-1]["content"] = (
                            user_content[:target_len]
                            + "\n\n[... trailing manuscript text trimmed to respect context limit ...]\n"
                        )
                    else:
                        mid = len(user_content) // 2
                        half_trim = trim_chars // 2
                        augmented_messages[-1]["content"] = (
                            user_content[: mid - half_trim]
                            + "\n\n[... middle text trimmed to respect context limit ...]\n\n"
                            + user_content[mid + half_trim :]
                        )
                    retry_tokens = min(max_needed_output, 1500)
                    retry_kwargs = {
                        "model": self.config.model,
                        "messages": augmented_messages,
                        "temperature": self.config.temperature,
                        "max_tokens": retry_tokens,
                    }
                    if not is_reasoning:
                        retry_kwargs["response_format"] = {"type": "json_object"}
                    response = self._openai_client.chat.completions.create(**retry_kwargs)
                else:
                    raise exc
            else:
                raise exc

        import re
        raw_text = response.choices[0].message.content or "{}"
        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        # Extract outer JSON structure if model included extra preamble or conversational text
        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            extracted_json = raw_text[first_brace : last_brace + 1]
            try:
                return response_model.model_validate_json(extracted_json)
            except Exception:
                pass

        try:
            return response_model.model_validate_json(raw_text)
        except Exception:
            # Bulletproof healing: repair unclosed quotes, drop incomplete objects, and balance bracket stack
            s = raw_text.strip()
            if first_brace != -1:
                s = raw_text[first_brace:].strip()

            # Step 1: If truncated inside an array of objects (like extracted_scores),
            # rewind to the last completely closed object `}` after the last `[`,
            # or drop incomplete element if first element was truncated
            last_sq = s.rfind("[")
            if last_sq != -1:
                last_brace_after_sq = s.rfind("}", last_sq)
                if last_brace_after_sq != -1:
                    candidate = s[: last_brace_after_sq + 1] + "]}"
                    try:
                        return response_model.model_validate_json(candidate)
                    except Exception:
                        pass
                else:
                    candidate = s[: last_sq + 1] + "]}"
                    try:
                        return response_model.model_validate_json(candidate)
                    except Exception:
                        pass

            # Step 2: Handle unclosed string literals
            in_string = False
            escape = False
            for char in s:
                if char == "\\" and not escape:
                    escape = True
                    continue
                if char == '"' and not escape:
                    in_string = not in_string
                escape = False

            if in_string:
                s += '"'

            # Step 3: Clean up dangling colon (key without value, e.g. "key":) or incomplete key
            s = re.sub(r":\s*$", ": null", s.strip())
            s = re.sub(r",\s*\"[^\":]*\"\s*$", "", s.strip())
            s = re.sub(r",\s*([\]}])", r"\1", s)
            s = re.sub(r",\s*$", "", s.strip())

            # Step 4: Stack-based bracket balancer
            stack = []
            in_str = False
            esc = False
            for char in s:
                if char == "\\" and not esc:
                    esc = True
                    continue
                if char == '"' and not esc:
                    in_str = not in_str
                elif not in_str:
                    if char in ("{", "["):
                        stack.append(char)
                    elif char == "}" and stack and stack[-1] == "{":
                        stack.pop()
                    elif char == "]" and stack and stack[-1] == "[":
                        stack.pop()
                esc = False

            while stack:
                opener = stack.pop()
                s += "}" if opener == "{" else "]"

            return response_model.model_validate_json(s)

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Standard free-form text generation."""
        if self.is_mock:
            return "Mock response for offline testing."

        if self._gemini_client is not None:
            from google.genai import types
            config_obj = types.GenerateContentConfig(
                temperature=self.config.temperature,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            if system_prompt:
                config_obj.system_instruction = system_prompt
            import time
            current_model = self.config.model
            fallback_models = ["gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-flash-lite-latest"]
            backoff = 4.0
            for attempt in range(5):
                try:
                    global _gemini_pacing_lock, _last_gemini_timestamp
                    with _gemini_pacing_lock:
                        now = time.time()
                        elapsed = now - _last_gemini_timestamp
                        if elapsed < 3.0:
                            time.sleep(3.0 - elapsed)
                        _last_gemini_timestamp = time.time()

                    response = self._gemini_client.models.generate_content(
                        model=current_model,
                        contents=prompt,
                        config=config_obj,
                    )
                    return response.text or ""
                except Exception as exc:
                    err_msg = str(exc)
                    if any(x in err_msg for x in ["503", "UNAVAILABLE", "high demand", "429", "RESOURCE_EXHAUSTED", "Quota exceeded", "404", "NOT_FOUND"]):
                        if attempt < 4:
                            if attempt >= 1 and attempt < len(fallback_models):
                                current_model = fallback_models[attempt]
                            time.sleep(backoff)
                            backoff = min(backoff * 1.5, 15.0)
                            continue
                    raise exc
            return ""

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._openai_client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content or ""
