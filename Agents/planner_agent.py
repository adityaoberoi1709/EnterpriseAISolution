import logging
from typing import TypedDict, Annotated, List, Sequence
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from config.LLM_Factory import get_llm
from config.settings import settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    plan: List[str]
    current_step: int
    step_results: List[str]
    final_answer: str
    context_docs: List[dict]


class ExecutionPlan(BaseModel):
    steps: list[str] = Field(..., description="Ordered list of sub-tasks to complete the query", min_length=1, max_length=8)
    reasoning: str = Field(..., description="Why this plan was chosen")


PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an enterprise AI planning agent. 
     Given a user query, break it into concrete sub-tasks.
     Each step should be independently executable.(retrieve info, lookup entity, summarize, calculate).
     Keep steps concise (max 8). Output a JSON plan."""),
    ("human", "Query: {query}"),
])


def plan_node(state: AgentState) -> AgentState:
    llm = get_llm()
    structured_llm = llm.with_structured_output(ExecutionPlan)
    chain = PLAN_PROMPT | structured_llm
    try:
        plan = chain.invoke({"query": state["query"]})
        logger.info(f"Plan generated: {plan.steps}")
        return {
            **state,
            "plan": plan.steps,
            "current_step": 0,
            "step_results": [],
            "messages": [AIMessage(content=f"Plan: {plan.steps}\nReasoning: {plan.reasoning}")],
        }
    except Exception as e:
        logger.warning(f"Planning failed, using single-step fallback: {e}")
        return {
            **state,
            "plan": [state["query"]],
            "current_step": 0,
            "step_results": [],
        }


STEP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an execution agent. Complete the given sub-task using the provided context.
Be concise and factual. If the context doesn't contain the answer, say so."""),
    ("human", """Sub-task: {step}
    Retrieved context:
    {context}
    Completed steps so far:
    {prior_results}
    Complete the sub-task."""),
])


def execute_step_node(state: AgentState) -> AgentState:
    from Agents.rag_agent import retrieve_context
    step_idx = state["current_step"]
    step = state["plan"][step_idx]
    logger.info(f"Executing step {step_idx + 1}/{len(state['plan'])}: {step}")
    context_docs = retrieve_context(step)
    context_text = "\n\n--\n\n".join([d.page_content for d in context_docs[:5]])
    prior = "\n".join([f"{i+1}. {r}" for i, r in enumerate(state["step_results"])])
    llm = get_llm()
    chain = STEP_PROMPT | llm

    try:
        response = chain.invoke({
            "step": step,
            "context": context_text or "No relevant context found.",
            "prior_results": prior or "None yet.",
        })
        result = response.content.strip()
    except Exception as e:
        result = f"Step failed: {e}"

    return {
        **state,
        "step_results": state["step_results"] + [result],
        "current_step": step_idx + 1,
        "context_docs": [{"text": d.page_content, "meta": d.metadata} for d in context_docs],
        "messages": [AIMessage(content=f"Step {step_idx+1} result: {result}")],
    }


SYNTH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a synthesis agent for an enterprise knowledge system.
     Combine the result of all completed steps into single, coherent, well-structured answer.
     Be factual cite the steps used and avoid hallucination.
     If information is missing, acknowledge it explicitly"""),
    ("human", """Original Query: {query}
      Step Results: {step_results}
      Provide the final comprehensive answer.""")
])


def synthesize_node(state: AgentState) -> AgentState:
    llm = get_llm()
    chain = SYNTH_PROMPT | llm
    step_text = "\n".join([
        f"Step {i+1} ({state['plan'][i]}): {r}"
        for i, r in enumerate(state["step_results"])
    ])
    try:
        response = chain.invoke({
            "query": state["query"],
            "step_results": step_text
        })
        final = response.content.strip()
    except Exception as e:
        final = f"Synthesis failed: {e}. Partial Results: {step_text}"
    return {
        **state,
        "final_answer": final,
        "messages": [AIMessage(content=final)]
    }


def should_continue(state: AgentState) -> str:
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    if state["current_step"] >= settings.MAX_AGENT_ITERATION:
        logger.warning("Max iterations reached, forcing synthesis")
        return "synthesize"
    return "execute_step"


def build_planner_graph() -> StateGraph:
    memory = MemorySaver()
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute_step")
    graph.add_conditional_edges("execute_step", should_continue, {"execute_step": "execute_step", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=memory)


_graph = None


def get_planner():
    global _graph
    if _graph is None:
        _graph = build_planner_graph()
    return _graph


def run_planner(query: str, thread_id: str = "default") -> dict:
    planner = get_planner()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "final_answer": "",
        "context_docs": []
    }
    final_state = planner.invoke(initial_state, config=config)
    return {
        "answer": final_state["final_answer"],
        "plan": final_state["plan"],
        "step_results": final_state["step_results"],
        "context_docs": final_state.get("context_docs", [])
    }
