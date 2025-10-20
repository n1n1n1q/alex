import json
from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP
from few_shot_prompts import FEW_SHOT_EXAMPLES


mcp = FastMCP("minecraft-planner")

SYSTEM_PROMPT = """You are an expert Minecraft strategist and planner. Your job is to analyze the current game state and create a structured action plan that will help the player survive and progress.

**CRITICAL: You must respond with ONLY valid JSON. No markdown, no code blocks, no explanations outside the JSON.**

Output Format:
{
    "reasoning": "Brief explanation of your strategy (2-3 sentences)",
    "subgoals": [
        {"name": "action_name", "params": {...}, "priority": 0-100}
    ],
    "immediate_action": "what to do right now",
    "context_notes": ["observation1", "observation2"]
}

Priority Guidelines:
- 100: Life-threatening emergencies (health < 5, imminent death)
- 90-95: Urgent safety (low health, night threats, no shelter)
- 70-85: Important progression (tools, resources, shelter)
- 50-65: Medium-term goals (exploration, mining)
- 0-40: Low priority (aesthetic, optional tasks)

Key Principles:
1. **Safety first**: Always consider health, hunger, time of day, and nearby threats
2. **Prerequisites**: Don't plan actions that require unavailable resources
3. **Progression**: Early game focuses on wood → crafting table → tools → mining
4. **Context awareness**: Consider biome, inventory, and environmental factors
5. **Immediacy**: The immediate_action should be executable right now with current resources

Common Action Names:
- Resource: collect_wood, collect_cobblestone, mine_iron, mine_diamonds, hunt_food
- Crafting: craft_table, craft_planks, craft_wooden_pickaxe, craft_stone_pickaxe, craft_stone_sword, craft_furnace
- Survival: build_shelter, seek_shelter, emergency_retreat, eat_food, heal
- Exploration: explore, scout_area, idle_scan

Remember: Respond with ONLY the JSON object, nothing else.
"""


ACTION_PLAN_SCHEMA = {
    "reasoning": "Brief explanation of the plan (2-3 sentences)",
    "subgoals": [
        {
            "name": "action_name",
            "params": {"key": "value"},
            "priority": "0-100 integer"
        }
    ],
    "immediate_action": "what to do right now",
    "context_notes": ["observation1", "observation2"]
}


@mcp.tool()
def plan_actions(game_state: dict) -> str:
    """
    Generate a structured action plan from current Minecraft game state.
    
    Takes the current game state and returns a complete prompt that should be
    sent to Gemini (or any LLM) to generate an action plan.
    
    Args:
        game_state: Current game state with inventory, health, position, etc.
        
    Returns:
        Complete prompt string with system instructions and few-shot examples
    """
    guidelines = get_planning_guidelines()

    allowed_actions = [
        a for category in guidelines["subgoals"].values() for a in category
    ]
    prompt = SYSTEM_PROMPT + "\n\n"

    prompt += "=== ALLOWED ACTIONS (you MUST use only these) ===\n"

    prompt += json.dumps(allowed_actions, indent=2) + "\n\n"
    prompt += "Do NOT create or invent actions outside this list.\n\n"

    prompt += "=== EXAMPLES ===\n\n"
    
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        prompt += f"Example {i}:\n"
        prompt += f"INPUT STATE:\n{json.dumps(example['state'], indent=2)}\n\n"
        prompt += f"OUTPUT PLAN:\n{json.dumps(example['plan'], indent=2)}\n\n"
        prompt += "---\n\n"
    
    prompt += "=== YOUR TASK ===\n\n"
    prompt += f"Current Game State:\n{json.dumps(game_state, indent=2)}\n\n"
    prompt += "Generate an action plan following the format above. Respond with ONLY the JSON object, no markdown, no code blocks.\n"
    
    return prompt


@mcp.tool()
def get_planning_guidelines() -> dict:
    """
    Get the planning guidelines and action name conventions.
    Useful for understanding what actions are available and how priorities work.
    
    Returns:
        Dictionary with priority levels, common actions, and progression tips
    """
    return {
        "priority_levels": {
            "100": "Life-threatening emergencies",
            "90-95": "Urgent safety issues",
            "70-85": "Important progression",
            "50-65": "Medium-term goals",
            "0-40": "Low priority tasks"
        },
        "subgoals": {
            "resource_gathering": [
                "collect_wood",
                "collect_cobblestone",
                "mine_iron",
                "mine_diamonds",
                "hunt_food"
            ],
            "crafting": [
                "craft_planks",
                "craft_table",
                "craft_wooden_pickaxe",
                "craft_stone_pickaxe",
                "craft_stone_sword",
                "craft_furnace",
                "craft_iron_pickaxe"
            ],
            "survival": [
                "emergency_retreat",
                "seek_shelter",
                "build_shelter",
                "eat_food",
                "heal"
            ],
            "exploration": [
                "explore",
                "scout_area",
                "idle_scan"
            ]
        },
        "progression_order": [
            "1. Collect wood (logs)",
            "2. Craft planks from logs",
            "3. Craft crafting table",
            "4. Craft wooden pickaxe",
            "5. Mine cobblestone",
            "6. Craft stone tools",
            "7. Build shelter before night",
            "8. Mine iron ore",
            "9. Craft furnace and smelt iron",
            "10. Craft iron tools"
        ],
        "survival_tips": [
            "Always monitor health and hunger",
            "Seek shelter before night (time_of_day=dusk)",
            "Avoid combat with low health",
            "Prioritize food when hunger is low",
            "Keep torches for night-time safety"
        ]
    }


@mcp.tool()
def validate_action_plan(plan_json: str) -> dict:
    """
    Validate that an action plan follows the correct structure.
    
    Args:
        plan_json: JSON string of the action plan to validate
        
    Returns:
        Validation result with errors if any
    """
    try:
        plan = json.loads(plan_json)
        
        errors = []
        warnings = []
        
        if "reasoning" not in plan:
            errors.append("Missing 'reasoning' field")
        if "subgoals" not in plan:
            errors.append("Missing 'subgoals' field")
        if "immediate_action" not in plan:
            errors.append("Missing 'immediate_action' field")
            
        if "subgoals" in plan:
            if not isinstance(plan["subgoals"], list):
                errors.append("'subgoals' must be a list")
            else:
                for i, sg in enumerate(plan["subgoals"]):
                    if "name" not in sg:
                        errors.append(f"Subgoal {i} missing 'name'")
                    if "priority" not in sg:
                        warnings.append(f"Subgoal {i} missing 'priority'")
                    elif not (0 <= sg["priority"] <= 100):
                        warnings.append(f"Subgoal {i} priority {sg['priority']} outside 0-100 range")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "plan": plan if len(errors) == 0 else None
        }
        
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "errors": [f"Invalid JSON: {str(e)}"],
            "warnings": [],
            "plan": None
        }


@mcp.resource("minecraft://examples")
def get_examples() -> str:
    """
    Get all few-shot examples as a formatted resource.
    Useful for debugging or understanding the expected format.
    """
    return json.dumps(FEW_SHOT_EXAMPLES, indent=2)


@mcp.resource("minecraft://schema")
def get_schema() -> str:
    """Get the expected action plan JSON schema"""
    return json.dumps(ACTION_PLAN_SCHEMA, indent=2)


if __name__ == "__main__":
    mcp.run()