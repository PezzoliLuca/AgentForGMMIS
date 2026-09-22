"""Universal LLM client layer supporting Native Google GenAI, OpenAI, Ollama, and Mocks."""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env if present
load_dotenv()

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""

    provider: str = os.getenv("LLM_PROVIDER", "gemini")
    model: str = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url: Optional[str] = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
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

        if self.config.provider == "mock" or not self.config.api_key:
            self.is_mock = True
            self._gemini_client = None
            self._openai_client = None
        elif self.config.provider == "gemini":
            from google import genai
            self.is_mock = False
            self._gemini_client = genai.Client(api_key=self.config.api_key)
            self._openai_client = None
        else:
            from openai import OpenAI
            self.is_mock = False
            self._gemini_client = None
            self._openai_client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
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

            return response_model.model_construct()

        # Native Google GenAI Provider
        if self._gemini_client is not None:
            schema_json = json.dumps(response_model.model_json_schema(), indent=2)
            augmented_system = f"{system_prompt or ''}\n\nYou MUST respond with a valid JSON object strictly conforming to this JSON Schema:\n{schema_json}"

            config_dict: Dict[str, Any] = {
                "response_mime_type": "application/json",
                "system_instruction": augmented_system,
                "temperature": self.config.temperature,
            }

            import time
            response = None
            max_retries = 5
            backoff = 4.0
            for attempt in range(max_retries):
                try:
                    response = self._gemini_client.models.generate_content(
                        model=self.config.model,
                        contents=prompt,
                        config=config_dict,
                    )
                    break
                except Exception as exc:
                    if "503" in str(exc) or "UNAVAILABLE" in str(exc) or "high demand" in str(exc).lower():
                        if attempt < max_retries - 1:
                            time.sleep(backoff)
                            backoff *= 2.0
                            continue
                    raise exc
            raw_text = response.text or "{}"
            # Extract JSON substring if wrapped in markdown code blocks
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
            return response_model.model_validate_json(raw_text)

        # OpenAI-compatible Provider (Ollama / Mufasa / OpenAI)
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = self._openai_client.beta.chat.completions.parse(
            model=self.config.model,
            messages=messages,
            response_format=response_model,
            temperature=self.config.temperature,
        )
        return completion.choices[0].message.parsed

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Standard free-form text generation."""
        if self.is_mock:
            return "Mock response for offline testing."

        if self._gemini_client is not None:
            config_dict: Dict[str, Any] = {"temperature": self.config.temperature}
            if system_prompt:
                config_dict["system_instruction"] = system_prompt
            response = self._gemini_client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=config_dict,
            )
            return response.text or ""

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
