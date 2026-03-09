"""
ClaimExtractorAgent
--------------------
Takes ingested articles and extracts discrete, checkable claims from each one.
Uses Gemini (PRIMARY) for deep reading. Returns claims appended to pipeline state.

A "claim" is:
  - A specific factual assertion that can be verified
  - Not an opinion, prediction, or vague statement
  - Atomic — one fact per claim

Example input:  "The President signed a bill that will cut taxes by 40% for middle-class families."
Example output: ["The President signed a bill.", "The bill cuts taxes by 40% for middle-class families."]
"""

import json
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from graph.state import PipelineState, ArticleState, ClaimState
from llm.provider import LLMRole

SYSTEM_PROMPT = """You are a precise claim extraction engine for a fact-checking system.

Your job is to extract discrete, verifiable factual claims from news articles.

Rules:
1. Extract ONLY specific, checkable factual assertions — not opinions, predictions, or vague statements.
2. Each claim must be atomic (one fact per claim).
3. Keep the original wording as close as possible — do not paraphrase heavily.
4. Ignore editorializing, emotional language, and conjecture.
5. Return ONLY a valid JSON array of strings. No preamble, no explanation, no markdown.

Example output:
["The unemployment rate fell to 3.7% in October.", "Congress passed the Infrastructure Investment Act.", "NASA launched the Artemis II mission on March 15."]

If no verifiable claims are found, return an empty array: []
"""


class ClaimExtractorAgent(BaseAgent):
    name = "ClaimExtractorAgent"
    llm_role = LLMRole.PRIMARY   # Gemini — needs deep reading

    async def _execute(self, state: PipelineState) -> PipelineState:
        logger = self._get_logger(state.run_id)
        all_claims: List[ClaimState] = []

        for article in state.articles:
            claims = await self._extract_from_article(article)
            await logger.info(
                f"Extracted {len(claims)} claims from article",
                context={"article_id": str(article.id), "title": article.title[:80]}
            )
            all_claims.extend(claims)

        state.claims.extend(all_claims)
        return state

    async def _extract_from_article(self, article: ArticleState) -> List[ClaimState]:
        """Call LLM to extract claims from a single article."""
        body_preview = (article.body or article.title)[:3000]  # stay within context limits

        user_prompt = f"""Article Title: {article.title}
Article Body:
{body_preview}

Extract all verifiable factual claims from this article."""

        response = await self.llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        raw_claims = self._parse_response(response.content)

        return [
            ClaimState(
                article_id=article.id,
                raw_text=claim_text,
            )
            for claim_text in raw_claims
            if claim_text.strip()
        ]

    def _parse_response(self, content: str) -> List[str]:
        """
        Safely parse the LLM JSON response.
        Handles cases where the model wraps output in markdown fences.
        """
        try:
            # Strip markdown fences if present
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])

            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return []
        except (json.JSONDecodeError, ValueError):
            # Log-worthy but not fatal — return empty list
            return []
