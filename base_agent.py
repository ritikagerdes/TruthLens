"""
Base Agent
-----------
All agents inherit from this. Enforces the contract:
- Every agent gets a logger and LLM provider
- Every agent implements run(state) -> state
- Errors are caught, logged, and added to state (pipeline continues)
"""

from abc import ABC, abstractmethod
from typing import Optional
import uuid

from graph.state import PipelineState
from llm.provider import LLMProvider, LLMRole, llm_provider
from logger.async_logger import AsyncLogger


class BaseAgent(ABC):
    """
    Abstract base for all TruthLens agents.

    Subclasses must implement:
        async def _execute(self, state: PipelineState) -> PipelineState

    The run() method wraps _execute() with error handling + logging.
    """

    name: str = "BaseAgent"
    llm_role: LLMRole = LLMRole.PRIMARY

    def __init__(self, provider: LLMProvider = llm_provider):
        self._provider = provider
        self._logger: Optional[AsyncLogger] = None

    @property
    def llm(self):
        return self._provider.get(role=self.llm_role)

    def _get_logger(self, run_id: Optional[uuid.UUID] = None) -> AsyncLogger:
        if self._logger is None or self._logger.run_id != run_id:
            self._logger = AsyncLogger(agent_name=self.name, run_id=run_id)
        return self._logger

    async def run(self, state: PipelineState) -> PipelineState:
        """
        Entry point called by LangGraph.
        Wraps _execute with error handling — a failing agent doesn't kill the pipeline.
        """
        logger = self._get_logger(state.run_id)
        await logger.info(f"{self.name} starting")

        try:
            updated_state = await self._execute(state)
            updated_state.completed_agents.append(self.name)
            await logger.info(f"{self.name} completed")
            return updated_state
        except Exception as e:
            error_msg = f"{self.name} failed: {str(e)}"
            await logger.error(error_msg, context={"exception": str(e)})
            state.errors.append(error_msg)
            return state  # Return unchanged state — pipeline continues

    @abstractmethod
    async def _execute(self, state: PipelineState) -> PipelineState:
        """Agent-specific logic. Must be implemented by subclasses."""
        ...
