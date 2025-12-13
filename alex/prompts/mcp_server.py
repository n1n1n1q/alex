import json
from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP
from few_shot_prompts import FEW_SHOT_EXAMPLES


mcp = FastMCP("minecraft-planner")

MEMORY_BUFFER = []
MAX_MEMORY_LENGTH = 3

SYSTEM_PROMPT = """You are an expert Minecraft strategist and planner. Your job is to analyze the current game state and create a structured action plan.

**CRITICAL: You must respond with ONLY valid JSON.**

Output Format:
{
    "reasoning": "Strategy explanation",
    "subgoals": [
        {"name": "STEVE-1 command", "params": {}, "priority": 0-100}
    ],
    "immediate_action": "STEVE-1 command",
    "context_notes": ["note1"]
}

**ACTION RULES (STEVE-1 FORMAT):**
1. Actions must be **2-3 words** maximum.
2. Structure: **VERB + OBJECT** (e.g., "mine log", "kill cow", "craft table").
3. Use Minecraft IDs for objects (log, dirt, stone, iron_ore).
4. Do NOT use abstract skills like "collect_wood" or "hunt_food".

Examples of valid actions:
- "mine log" (NOT collect_wood)
- "mine stone"
- "kill cow" (NOT hunt_food)
- "kill zombie"
- "craft planks"
- "craft sticks"
- "place dirt"
- "look around"

For dynamic targets (e.g. hunting), construct the string yourself: "kill {mob_name}".
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

    guidelines = get_planning_guidelines()

    allowed_actions = [
        a for category in guidelines["subgoals"].values() for a in category
    ]
    prompt = SYSTEM_PROMPT + "\n\n"

    if MEMORY_BUFFER:
        prompt += "=== PREVIOUS HISTORY (Short-term Memory) ===\n"
        prompt += "Here is what you planned in the previous steps. Use this to maintain continuity.\n\n"
        for i, memory in enumerate(MEMORY_BUFFER, 1):
            turn_num = len(MEMORY_BUFFER) - i + 1
            prompt += f"Turn -{turn_num}:\n"
            prompt += f"  Reasoning: {memory['reasoning']}\n"
            prompt += f"  Action Taken: {memory['immediate_action']}\n"
            # We explicitly list subgoals to remind the agent of its medium-term plan
            prompt += f"  Subgoals Set: {json.dumps(memory['subgoals'])}\n"
            prompt += "---\n"
        prompt += "\n"

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
    prompt += "Generate an action plan following the format above. Generate only ONE or TWO subgoals. Respond with ONLY the JSON object, no markdown, no code blocks.\n"
    
    return prompt


@mcp.tool()
def get_planning_guidelines() -> dict:

    return {
        "subgoals": {
            "gathering": [
                "mine log",
                "mine dirt",
                "mine stone",
                "mine iron_ore",
                "mine coal_ore",
                "mine diamond_ore"
            ],
            "combat": [
                "kill cow",
                "kill sheep",
                "kill pig",
                "kill chicken",
                "kill zombie",
                "kill skeleton",
                "kill creeper"
            ],
            "crafting": [
                "craft planks",
                "craft sticks",
                "craft crafting_table",
                "craft stone_pickaxe",
                "craft furnace",
                "craft torch"
            ],
            "survival": [
                "place dirt",
                "place cobblestone",
                "eat food",
                "look around"
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
            "Craft and place torches for night-time safety"
        ]
    }


@mcp.tool()
def validate_action_plan(plan_json: str) -> dict:

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

    return json.dumps(FEW_SHOT_EXAMPLES, indent=2)


@mcp.resource("minecraft://schema")
def get_schema() -> str:

    return json.dumps(ACTION_PLAN_SCHEMA, indent=2)


if __name__ == "__main__":
    mcp.run()
