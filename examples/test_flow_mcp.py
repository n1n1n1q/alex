from __future__ import annotations
import asyncio
import json
import os
import sys
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from dataclasses import is_dataclass
from typing import Any, Dict, Optional
from alex.llm.extractor import state_to_json_str

load_dotenv()

DEFAULT_TIMEOUT = float(os.getenv("MCP_FLOW_TIMEOUT", "45"))
MODEL_NAME = os.getenv("MCP_GEMINI_MODEL", "gemini-2.0-flash-exp")
_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["alex/llm/mcp_server.py"],
    env=None,
)
_MODEL: Optional[genai.GenerativeModel] = None


def _ensure_model() -> Optional[genai.GenerativeModel]:
    """Lazy init and cache Gemini model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Gemini API key not set")
        return None
    try:
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        }
        _MODEL = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
        )
    except Exception as exc:
        print(f"Failed to initialise Gemini model: {exc}")
        return None
    return _MODEL


def _normalize_game_state(game_state: Optional[Any]) -> Dict[str, Any]:
    """Convert different game state formats to JSON-serializable dict."""
    if isinstance(game_state, dict):
        return game_state
    if is_dataclass(game_state):
        return json.loads(state_to_json_str(game_state))
    raise TypeError(
        f"Unsupported game_state type: {type(game_state)!r}. "
        "Expected dict, GameState, or dataclass-compatible object."
    )


async def demonstrate_flow(game_state=None):
    """Main async flow for MCP + Gemini."""
    model = _ensure_model()
    if model is None:
        return

    game_state_dict = _normalize_game_state(game_state)

    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Step 1: Get prompt from MCP server
            result = await session.call_tool(
                "plan_actions",
                arguments={"game_state": game_state_dict}
            )
            prompt = result.content[0].text

            # Step 2: Send to Gemini model
            print("Sending prompt to agent...")
            response = model.generate_content(prompt)
            plan_json = response.text

            # Step 3: Parse and validate
            try:
                plan = json.loads(plan_json)
                print("Response is valid JSON")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON from Gemini: {e}")
                return

            validation_result = await session.call_tool(
                "validate_action_plan",
                arguments={"plan_json": plan_json}
            )
            validation = json.loads(validation_result.content[0].text)

            if not validation.get("valid", False):
                print("Plan validation failed")
                print(f"Errors: {validation.get('errors')}")
                return

            # Step 4: Print summary
            print("\nFINAL OUTPUT:")
            print("-" * 60)
            # print(f"Reasoning: {plan.get('reasoning', 'N/A')}")
            # print(f"Immediate Action: {plan.get('immediate_action', 'N/A')}")
            print(plan)

            return {"plan": plan, "validation": validation, "prompt": prompt}


def process_game_state(game_state: Any):
    """
    Safe, synchronous entry point for repeated calls.
    Each call runs in a fresh asyncio loop (no reuse issues).
    """
    try:
        asyncio.run(demonstrate_flow(game_state))
    except RuntimeError as e:
        # Handles "event loop is closed" errors gracefully
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(demonstrate_flow(game_state))
        loop.close()


def main():
    """If called directly (CLI), read JSON from stdin."""
    try:
        input_data = sys.stdin.read()
        if input_data.strip():
            game_state = json.loads(input_data)
        else:
            game_state = None
        asyncio.run(demonstrate_flow(game_state))
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
