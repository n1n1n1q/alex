from __future__ import annotations

from alex import Agent

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
