import os
import json
import asyncio
import sys
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
import json
load_dotenv()


async def demonstrate_flow():
    """
    demonstration of the MCP + Gemini flow
    """
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Gemini API key not set")
        return
    
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.7,
        "max_output_tokens": 2048,
        "response_mime_type": "application/json",
    }
    
    # Set up Gemini model
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config
    )
    
    # MCP server parameters
    server_params = StdioServerParameters(
    command=sys.executable,
    args=["alex/llm/mcp_server.py"],
    env=None
)
    # Example game state
    with open('examples/example_state.json', 'r') as file:
        example_game_state = json.load(file)

    game_state = example_game_state

  
    print("Connecting to MCP server")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP server")
            
            # Call the MCP tool to get the prompt
            result = await session.call_tool(
                "plan_actions",
                arguments={"game_state": game_state}
            )
            
            prompt = result.content[0].text
            
            print(f"\nReceived prompt from MCP server")
            print(f"Length: {len(prompt)} characters")
            
            
            print("Sending prompt to agent")
            print(f"Model: gemini-2.0-flash-exp")
            print(f"Config: JSON output, temp=0.7")
            
            response = model.generate_content(prompt)
            
            print("\nReceived response from Gemini")
            
            plan_json = response.text
            
            print(f"\nRaw Gemini Response:")
            print("-" * 70)
            print(plan_json)
            print("-" * 70)
            
            try:
                plan = json.loads(plan_json)
                print("\nResponse is valid JSON")
            except json.JSONDecodeError as e:
                print(f"\nInvalid JSON from Gemini: {e}")
                return
            
            print("Validating plan via MCP server")
            
            validation_result = await session.call_tool(
                "validate_action_plan",
                arguments={"plan_json": plan_json}
            )
            
            validation = json.loads(validation_result.content[0].text)
            
            if not validation["valid"]:
                print("\nPlan validation failed")
                print(f"Errors: {validation['errors']}")
                return
            
            if validation["warnings"]:
                print(f"\nWarnings: {validation['warnings']}")
            else:
                print("\nPlan is valid with no warnings")
            
            
            subgoals = []
            for sg_dict in plan.get("subgoals", []):
                subgoal = {
                    "name": sg_dict["name"],
                    "params": sg_dict.get("params", {}),
                    "priority": sg_dict.get("priority", 50)
                }
                subgoals.append(subgoal)
            
            
            print("\nFINAL OUTPUT:")
            print("-" * 70)
            print(f"Reasoning: {plan.get('reasoning', 'N/A')}")
            print(f"\nImmediate Action: {plan.get('immediate_action', 'N/A')}")
            print(f"\nSubgoals:")
            for i, sg in enumerate(subgoals, 1):
                print(f"  {i}. {sg['name']}")
                print(f"     Priority: {sg['priority']}")
                print(f"     Params: {sg['params']}")
            
            if "context_notes" in plan:
                print(f"\nContext Notes:")
                for note in plan["context_notes"]:
                    print(f"  • {note}")
            

def main():
    """
    Main entry point
    """
    import sys
    
    asyncio.run(demonstrate_flow())


if __name__ == "__main__":
    main()