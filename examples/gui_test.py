from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import PlayCallback

sim = MinecraftSim(
    action_type="env",
    callbacks=[
        PlayCallback()
    ]
)

obs, info = sim.reset()
terminated = False

while not terminated:
    action = None
    # obs, reward, terminated, info = sim.step(action)
    result = sim.step(action)
    print(result)

sim.close()
