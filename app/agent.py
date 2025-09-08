from typing import TypedDict, List, Any
import os
import json

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from dotenv import load_dotenv

from app.tools import get_property_details, get_suburb_trends


class AgentState(TypedDict):
    """Tracks the evolving state for the real estate agent."""

    query: str
    result: str
    tool_calls: List[Any]


# Tool and LLM setup
tools = [get_property_details, get_suburb_trends]

# Load environment from .env if present
load_dotenv()

# Initialize LLM (requires OPENAI_API_KEY)
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Create a .env with OPENAI_API_KEY=... or export it."
    )

llm = ChatOpenAI(model="gpt-4-turbo")

# Bind tools to create the agent-capable model
agent_model = llm.bind_tools(tools)

__all__ = [
    "AgentState",
    "tools",
    "llm",
    "agent_model",
    "StateGraph",
    "END",
]

# ----------------------------
# Graph node functions
# ----------------------------

def call_model(state: AgentState):
    """Invoke the tool-enabled LLM with the user's query.

    Returns a dict updating the state's `tool_calls` with the model message (which may contain tool calls).
    """
    query = state["query"]
    ai_message: AIMessage = agent_model.invoke(query)
    return {"tool_calls": [ai_message]}


def call_tool(state: AgentState):
    """Execute the last requested tool call and append its ToolMessage to state."""
    if not state.get("tool_calls"):
        return {"tool_calls": []}

    last_message = state["tool_calls"][-1]
    # Expecting the last message to be an AIMessage with tool_calls
    if not isinstance(last_message, AIMessage) or not getattr(last_message, "tool_calls", None):
        return {"tool_calls": []}

    tool_call = last_message.tool_calls[-1]
    tool_name: str = tool_call.get("name")
    tool_args = tool_call.get("args") or {}
    tool_call_id = tool_call.get("id")

    # Find the matching tool by name
    selected_tool = None
    for t in tools:
        if getattr(t, "name", None) == tool_name:
            selected_tool = t
            break

    if selected_tool is None:
        # Return a tool message indicating failure to locate tool
        content = json.dumps({"error": f"Tool not found: {tool_name}"})
        return {"tool_calls": [ToolMessage(content=content, tool_call_id=tool_call_id or "")]}

    # Invoke tool with provided arguments
    output = selected_tool.invoke(tool_args)

    # Tool outputs should be JSON-serializable; ensure string content
    if not isinstance(output, str):
        try:
            content = json.dumps(output)
        except Exception:
            content = str(output)
    else:
        content = output

    tool_message = ToolMessage(content=content, tool_call_id=tool_call_id or "")
    return {"tool_calls": [tool_message]}


def should_continue(state: AgentState):
    """Router to decide whether to continue with tool execution or end.

    If the last message contains tool calls, we should continue; otherwise, end.
    """
    if not state.get("tool_calls"):
        return "end"
    last_message = state["tool_calls"][-1]
    # Continue if AI requests tool calls; end if it's a final LLM answer or a ToolMessage
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"

# ----------------------------
# Graph wiring
# ----------------------------

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("action", call_tool)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
graph.add_edge("action", "agent")

app = graph.compile()

# Update exports
__all__.extend(["graph", "app"])


