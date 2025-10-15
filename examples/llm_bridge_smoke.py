from __future__ import annotations

from alex.llm import Agent

# Minimal raw observation compatible with our state_extractor
RAW_OBS = {
    "inventory": {"logs": 0, "planks": 0},
    "health": 20,
    "hunger": 20,
    "time_of_day": "day",
}


def main():
    agent = Agent()
    result = agent.step(RAW_OBS)
    print("SkillResult:", result)


if __name__ == "__main__":
    main()
