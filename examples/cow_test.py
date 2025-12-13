import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    SpeedTestCallback, 
    RecordCallback, 
    SummonMobsCallback, 
    MaskActionsCallback, 
    RewardsCallback, 
    CommandsCallback, 
    FastResetCallback
)

sim = MinecraftSim(
    action_type="env",
    callbacks=[
        SpeedTestCallback(50), 
        SummonMobsCallback([{'name': 'cow', 'number': 10, 'range_x': [-5, 5], 'range_z': [-5, 5]}]),
        MaskActionsCallback(inventory=0, camera=np.array([0., 0.])), 
        RecordCallback(record_path="./output", fps=30),
        RewardsCallback([{
            'event': 'kill_entity', 
            'objects': ['cow', 'sheep'], 
            'reward': 1.0, 
            'identity': 'kill sheep or cow', 
            'max_reward_times': 5, 
        }]),
        CommandsCallback(commands=[
            '/give @p minecraft:iron_sword 1',
            '/give @p minecraft:diamond 64',
        ]), 
        FastResetCallback(
            biomes=['mountains'],
            random_tp_range=1000,
        )
    ]
)
obs, info = sim.reset()
print(sim.action_space)
for i in range(2000):
    action = sim.action_space.sample()
    obs, reward, terminated, truncated, info = sim.step(action)
sim.close()