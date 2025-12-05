import numpy as np
from alex.extractor import extract_state, state_to_json_file
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    SummonMobsCallback, 
    CommandsCallback, 
)

sim = MinecraftSim(
    seed=42,
    obs_size=(128, 128), 
    action_type="agent", 
    preferred_spawn_biome="savanna",
    num_empty_frames=120,
    callbacks=[
        SummonMobsCallback([{'name': 'cow', 'number': 10, 'range_x': [-5, 5], 'range_z': [-5, 5]}]),
        CommandsCallback(commands=[
            '/give @p minecraft:iron_sword 1',
            '/give @p minecraft:diamond 64',
            '/give @p minecraft:bread 16',
            '/give @p minecraft:apple 16',
            '/give @p minecraft:iron_sword 1',
        ]), 
    ]
)
obs, info = sim.reset()

# Collect and save game state every 10 steps
for i in range(51):
    action = sim.action_space.sample()
    action["mobs"] = [-3, 3, -3, 3, -3, 3]
    action["voxels"] = [-2, 2, -2, 2, -2, 2]
    obs, reward, terminated, truncated, info = sim.step(action)

    if i % 10 == 0:
        state = extract_state(info)
        state_to_json_file(state, f"game_state_step_{i}.json")

    if terminated or truncated:
        break

sim.close()