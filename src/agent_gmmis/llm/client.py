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
            max_retries = 6
            backoff = 5.0
            current_model = self.config.model
            fallback_models = ["gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-flash-lite-latest"]
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
                            if attempt >= 1 and attempt < len(fallback_models):
                                current_model = fallback_models[attempt]
                                print(f"   ⚠️ [Google AI Studio API Retry] Switching to model '{current_model}'...")
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
        # ExtractedPaperData needs at most ~2800 tokens (including thought_process and 30+ benchmark rows)
        # Reinterrogation responses need at most ~800 tokens.
        resp_name = getattr(response_model, "__name__", "")
        max_needed_output = 3500 if resp_name == "ExtractedPaperData" else 800

        # Clean and compact schema representation to preserve context tokens
        raw_schema = response_model.model_json_schema()
        def _compact_schema_dict(d: Any) -> Any:
            if isinstance(d, dict):
                return {k: _compact_schema_dict(v) for k, v in d.items() if k not in ("title", "default")}
            elif isinstance(d, list):
                return [_compact_schema_dict(v) for v in d]
            return d

        compact_schema = _compact_schema_dict(raw_schema)
        schema_json = json.dumps(compact_schema, separators=(",", ":"))

        total_chars = len(prompt) + (len(system_prompt) if system_prompt else 0) + len(schema_json)
        approx_tokens = int(total_chars / 2.0) + 400
        context_limit = 16384 if self.config.provider in ("mufasa", "ollama") else 32768

        # Guarantee at least max_needed_output tokens of generation headroom by trimming prompt if needed
        required_headroom = max_needed_output + 300
        if approx_tokens > context_limit - required_headroom and len(prompt) > 4000:
            excess_tokens = approx_tokens - (context_limit - required_headroom)
            excess_chars = int(excess_tokens * 2.2) + 400
            mid = len(prompt) // 2
            half_trim = min(mid - 500, max(500, excess_chars // 2))
            prompt = prompt[: mid - half_trim] + "\n\n[... middle text trimmed to respect context limit ...]\n\n" + prompt[mid + half_trim :]
            messages[-1]["content"] = prompt
            total_chars = len(prompt) + (len(system_prompt) if system_prompt else 0) + len(schema_json)
            approx_tokens = int(total_chars / 2.0) + 400

        safe_max_tokens = min(max_needed_output, max(800, context_limit - approx_tokens - 100))

        # Direct structured completion for vLLM / Ollama (avoids slow redundant beta.parse timeout)
        use_beta_parse = self.config.provider not in ("mufasa", "ollama")

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

        augmented_messages = [dict(m) for m in messages]
        augmented_messages[0]["content"] += f"\n\nYou MUST respond with a valid JSON object strictly conforming to this JSON Schema:\n{schema_json}"
        
        try:
            response = self._openai_client.chat.completions.create(
                model=self.config.model,
                messages=augmented_messages,
                response_format={"type": "json_object"},
                temperature=self.config.temperature,
                max_tokens=safe_max_tokens,
            )
        except Exception as exc:
            # Automatic recovery: if vLLM throws 400 context limit error, dynamically adapt output tokens or trim
            err_str = str(exc)
            if "maximum context length is" in err_str:
                import re
                m_ctx = re.search(r"maximum context length is (\d+)", err_str)
                m_inp = re.search(r"prompt contains at least (\d+)", err_str) or re.search(r"value=(\d+)", err_str)
                ctx_len = int(m_ctx.group(1)) if m_ctx else 16384
                inp_len = int(m_inp.group(1)) if m_inp else approx_tokens

                min_safe_headroom = min(max_needed_output, 2500)
                if inp_len + min_safe_headroom <= ctx_len:
                    retry_tokens = max(min_safe_headroom, ctx_len - inp_len - 64)
                    response = self._openai_client.chat.completions.create(
                        model=self.config.model,
                        messages=augmented_messages,
                        response_format={"type": "json_object"},
                        temperature=self.config.temperature,
                        max_tokens=retry_tokens,
                    )
                else:
                    # Trim middle of the prompt to fit within context length with full headroom
                    user_content = augmented_messages[-1]["content"]
                    excess_chars = int((inp_len - (ctx_len - (max_needed_output + 400))) * 2.5) + 300
                    if len(user_content) > excess_chars + 1000:
                        mid = len(user_content) // 2
                        half_trim = excess_chars // 2
                        augmented_messages[-1]["content"] = (
                            user_content[: mid - half_trim]
                            + "\n\n[... middle text trimmed to respect context limit ...]\n\n"
                            + user_content[mid + half_trim :]
                        )
                        response = self._openai_client.chat.completions.create(
                            model=self.config.model,
                            messages=augmented_messages,
                            response_format={"type": "json_object"},
                            temperature=self.config.temperature,
                            max_tokens=max_needed_output,
                        )
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

        try:
            return response_model.model_validate_json(raw_text)
        except Exception:
            # Bulletproof healing: repair unclosed quotes, drop incomplete objects, and balance bracket stack
            s = raw_text.strip()

            # Step 1: If truncated inside an array of objects (like extracted_scores),
            # rewind to the last completely closed object `}` after the last `[`
            last_sq = s.rfind("[")
            if last_sq != -1:
                last_brace_after_sq = s.rfind("}", last_sq)
                if last_brace_after_sq != -1:
                    candidate = s[:last_brace_after_sq + 1] + "]}"
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
