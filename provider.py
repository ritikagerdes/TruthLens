"""
LLM Provider Abstraction Layer
-------------------------------
Swap between Gemini and Groq without touching agent code.
All agents talk to LLMProvider — never to a specific SDK directly.
"""

from enum import Enum
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from config import get_settings


class LLMRole(str, Enum):
    """
    Semantic roles map to appropriate models.
    PRIMARY  → complex reasoning, long context  (Gemini)
    FAST     → quick classification, extraction (Groq)
    """
    PRIMARY = "primary"
    FAST = "fast"


class LLMProvider:
    """
    Central factory for LLM instances.
    Agents request a model by ROLE, not by provider name.
    This is your swap point — change providers here, nothing else changes.
    """

    def __init__(self):
        self._settings = get_settings()
        self._instances: dict[LLMRole, BaseChatModel] = {}

    def get(self, role: LLMRole = LLMRole.PRIMARY, temperature: float = 0.0) -> BaseChatModel:
        """
        Returns a cached LLM instance for the given role.
        temperature=0.0 by default for deterministic fact-checking.
        """
        cache_key = (role, temperature)
        if cache_key not in self._instances:
            self._instances[cache_key] = self._build(role, temperature)
        return self._instances[cache_key]

    def _build(self, role: LLMRole, temperature: float) -> BaseChatModel:
        if role == LLMRole.PRIMARY:
            return ChatGoogleGenerativeAI(
                model=self._settings.gemini_model,
                google_api_key=self._settings.gemini_api_key,
                temperature=temperature,
            )
        elif role == LLMRole.FAST:
            return ChatGroq(
                model=self._settings.groq_model,
                groq_api_key=self._settings.groq_api_key,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown LLM role: {role}")


# Singleton — import this everywhere
llm_provider = LLMProvider()
