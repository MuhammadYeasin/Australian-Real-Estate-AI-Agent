from app.agent import app
from typing import List

try:
    from langchain_core.messages import ToolMessage, AIMessage
except Exception:  # Fallback types if not available
    ToolMessage = type("ToolMessage", (), {})  # type: ignore
    AIMessage = type("AIMessage", (), {})  # type: ignore


def _extract_final_output(state: dict) -> str:
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return ""
    last_message = tool_calls[-1]
    # If message objects from LangChain are present, use their content
    if isinstance(last_message, (ToolMessage, AIMessage)):
        content = getattr(last_message, "content", "")
        return content if isinstance(content, str) else str(content)
    # Otherwise, stringify
    return str(last_message)


def main() -> None:
    history: List[object] = []
    while True:
        user_input = input("Ask a question about Melbourne real estate (type 'exit' to quit): ")
        if user_input.strip().lower() == "exit":
            break

        try:
            # Persist conversation by passing prior tool_calls history
            state = app.invoke({"query": user_input, "tool_calls": history})
            output = _extract_final_output(state)
            print("\n=== Agent Response ===")
            print(output if output else "No response produced.")
            print("======================\n")
            # Update history for next turn
            history = state.get("tool_calls", history)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()


