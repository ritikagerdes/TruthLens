"""
TruthLens Pipeline
-------------------
LangGraph graph wiring all 7 agents in sequence.
Every agent is a node. State flows through each one.
Swap agent order or add branches here — agents don't know about each other.
"""

from langgraph.graph import StateGraph, END

from graph.state import PipelineState
from agents.stubs import (
    NewsIngestionAgent,
    ClaimClassifierAgent,
    FactCheckerAgent,
    EvidenceRankerAgent,
    VerdictAgent,
    SummaryAgent,
)
from agents.claim_extractor import ClaimExtractorAgent


def build_pipeline() -> StateGraph:
    """
    Constructs and compiles the TruthLens LangGraph pipeline.
    Returns a compiled graph ready to invoke.
    """

    # Instantiate agents
    ingestion = NewsIngestionAgent()
    extractor = ClaimExtractorAgent()
    classifier = ClaimClassifierAgent()
    fact_checker = FactCheckerAgent()
    ranker = EvidenceRankerAgent()
    verdict = VerdictAgent()
    summary = SummaryAgent()

    # Build graph
    graph = StateGraph(PipelineState)

    # Register nodes — each node is an agent's run() method
    graph.add_node("ingest", ingestion.run)
    graph.add_node("extract_claims", extractor.run)
    graph.add_node("classify_claims", classifier.run)
    graph.add_node("fact_check", fact_checker.run)
    graph.add_node("rank_evidence", ranker.run)
    graph.add_node("verdict", verdict.run)
    graph.add_node("summary", summary.run)

    # Linear pipeline — sequential edges
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "extract_claims")
    graph.add_edge("extract_claims", "classify_claims")
    graph.add_edge("classify_claims", "fact_check")
    graph.add_edge("fact_check", "rank_evidence")
    graph.add_edge("rank_evidence", "verdict")
    graph.add_edge("verdict", "summary")
    graph.add_edge("summary", END)

    return graph.compile()


# Singleton compiled pipeline
pipeline = build_pipeline()
