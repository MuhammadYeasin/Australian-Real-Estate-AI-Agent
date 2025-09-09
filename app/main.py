from app.agent import app, langsmith_client
from typing import List
import os
import uuid
from datetime import datetime

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
    session_id = str(uuid.uuid4())
    
    print("🏠 Australian Real Estate AI Agent")
    print("=" * 40)
    if langsmith_client:
        print("✅ LangSmith tracing enabled")
        print(f"📊 Session ID: {session_id}")
        print(f"🔗 View traces at: https://smith.langchain.com")
    else:
        print("⚠️  LangSmith tracing disabled (set LANGCHAIN_API_KEY to enable)")
    print("=" * 40)
    
    while True:
        user_input = input("\nAsk a question about Melbourne real estate (type 'exit' to quit): ")
        if user_input.strip().lower() == "exit":
            break

        try:
            # Log conversation start to LangSmith if available
            if langsmith_client:
                try:
                    langsmith_client.create_run(
                        name="real_estate_conversation",
                        run_type="chain",
                        inputs={"user_query": user_input, "session_id": session_id},
                        project_name=os.getenv("LANGCHAIN_PROJECT", "Australian-Real-Estate-Agent")
                    )
                except Exception as e:
                    print(f"⚠️  LangSmith logging error: {e}")
            
            # Persist conversation by passing prior tool_calls history
            state = app.invoke({"query": user_input, "tool_calls": history})
            output = _extract_final_output(state)
            
            print("\n=== Agent Response ===")
            print(output if output else "No response produced.")
            print("======================\n")
            
            # Log conversation completion to LangSmith if available
            if langsmith_client:
                try:
                    langsmith_client.create_run(
                        name="real_estate_conversation_complete",
                        run_type="chain",
                        outputs={"agent_response": output, "session_id": session_id},
                        project_name=os.getenv("LANGCHAIN_PROJECT", "Australian-Real-Estate-Agent")
                    )
                except Exception as e:
                    print(f"⚠️  LangSmith logging error: {e}")
            
            # Update history for next turn
            history = state.get("tool_calls", history)
        except Exception as e:
            print(f"Error: {e}")
            # Log error to LangSmith if available
            if langsmith_client:
                try:
                    langsmith_client.create_run(
                        name="real_estate_conversation_error",
                        run_type="chain",
                        outputs={"error": str(e), "session_id": session_id},
                        project_name=os.getenv("LANGCHAIN_PROJECT", "Australian-Real-Estate-Agent")
                    )
                except Exception as langsmith_error:
                    print(f"⚠️  LangSmith logging error: {langsmith_error}")
    
    print(f"\n👋 Session ended. Session ID: {session_id}")
    if langsmith_client:
        print("📊 Check your LangSmith dashboard for detailed traces and analytics")


if __name__ == "__main__":
    main()


