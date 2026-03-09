"""
Tests for ClaimExtractorAgent
-------------------------------
Arrange-Act-Assert pattern (consistent with your .NET NUnit style).
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from agents.claim_extractor import ClaimExtractorAgent
from graph.state import PipelineState, ArticleState


def make_article(**kwargs) -> ArticleState:
    defaults = dict(
        id=uuid.uuid4(),
        url="https://example.com/article",
        title="Test Article",
        body="The unemployment rate fell to 3.7% in October. The Fed raised interest rates by 0.25%.",
        source_name="Test Source",
    )
    return ArticleState(**{**defaults, **kwargs})


def make_state(articles=None) -> PipelineState:
    return PipelineState(
        run_id=uuid.uuid4(),
        articles=articles or [],
    )


@pytest.mark.asyncio
async def test_extract_claims_returns_claims_for_valid_article():
    # Arrange
    agent = ClaimExtractorAgent()
    article = make_article()
    state = make_state(articles=[article])

    mock_response = MagicMock()
    mock_response.content = '["The unemployment rate fell to 3.7% in October.", "The Fed raised interest rates by 0.25%."]'

    with patch.object(agent, "llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Act
        result = await agent.run(state)

    # Assert
    assert len(result.claims) == 2
    assert result.claims[0].raw_text == "The unemployment rate fell to 3.7% in October."
    assert result.claims[0].article_id == article.id
    assert "ClaimExtractorAgent" in result.completed_agents


@pytest.mark.asyncio
async def test_extract_claims_handles_empty_response_gracefully():
    # Arrange
    agent = ClaimExtractorAgent()
    article = make_article(body="This is an editorial with no checkable facts.")
    state = make_state(articles=[article])

    mock_response = MagicMock()
    mock_response.content = "[]"

    with patch.object(agent, "llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Act
        result = await agent.run(state)

    # Assert
    assert len(result.claims) == 0
    assert len(result.errors) == 0  # No errors — empty is valid


@pytest.mark.asyncio
async def test_extract_claims_handles_malformed_llm_response():
    # Arrange
    agent = ClaimExtractorAgent()
    article = make_article()
    state = make_state(articles=[article])

    mock_response = MagicMock()
    mock_response.content = "Sorry, I cannot process this article."  # Non-JSON response

    with patch.object(agent, "llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Act
        result = await agent.run(state)

    # Assert — agent degrades gracefully, no crash, no error propagated
    assert len(result.claims) == 0
    assert "ClaimExtractorAgent" in result.completed_agents


@pytest.mark.asyncio
async def test_agent_failure_does_not_kill_pipeline():
    # Arrange
    agent = ClaimExtractorAgent()
    article = make_article()
    state = make_state(articles=[article])

    with patch.object(agent, "llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API timeout"))

        # Act
        result = await agent.run(state)

    # Assert — error captured in state, pipeline continues
    assert len(result.errors) == 1
    assert "ClaimExtractorAgent failed" in result.errors[0]
    assert "ClaimExtractorAgent" not in result.completed_agents


@pytest.mark.asyncio
async def test_parse_response_strips_markdown_fences():
    # Arrange
    agent = ClaimExtractorAgent()
    fenced_json = '```json\n["Claim one.", "Claim two."]\n```'

    # Act
    result = agent._parse_response(fenced_json)

    # Assert
    assert result == ["Claim one.", "Claim two."]
