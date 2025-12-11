import json
import asyncio
import sys
import torch
from transformers import pipeline
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from alex.core.config import get_config


async def demonstrate_flow():

    cfg = get_config()
    model_name = cfg.hf_model_name
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: {model_name} on {device}...")
    
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    
    generation_config = {
        "max_new_tokens": cfg.hf_max_tokens,
        "temperature": cfg.hf_temperature,
        "top_p": 0.95,
        "do_sample": True,
        "return_full_text": False,
    }
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[cfg.mcp_server_path],
        env=None
    )
    
    with open('examples/example_state.json', 'r') as file:
        example_game_state = json.load(file)

    game_state = example_game_state

    print("Connecting to MCP server")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP server")
            
            result = await session.call_tool(
                "plan_actions",
                arguments={"game_state": game_state}
            )
            
            prompt = result.content[0].text
            
            print(f"\nReceived prompt from MCP server")
            print(f"Length: {len(prompt)} characters")
            
            print("Sending prompt to model")
            print(f"Model: {model_name}")
            print(f"Config: JSON output, temp=0.7")
            
            messages = [
                {"role": "system", "content": "You are an expert Minecraft agent planner. You must output ONLY valid JSON. Do not include markdown formatting or explanations."},
                {"role": "user", "content": prompt}
            ]
            
            outputs = pipe(messages, **generation_config)
            response_text = outputs[0]["generated_text"]
            
            print("\nReceived response from model")
            
            plan_json = response_text.strip()
            
            import re
            pattern = r"```(?:json)?\s*(.*?)```"
            match = re.search(pattern, plan_json, re.DOTALL)
            if match:
                plan_json = match.group(1).strip()
            
            print(f"\nRaw Model Response:")
            print("-" * 70)
            print(plan_json)
            print("-" * 70)
            
            try:
                plan = json.loads(plan_json)
                print("\nResponse is valid JSON")
            except json.JSONDecodeError as e:
                print(f"\nInvalid JSON from model: {e}")
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
                    print(f"  - {note}")


def main():

    import sys
    
    asyncio.run(demonstrate_flow())


if __name__ == "__main__":
    main()
