

def policy_generator():
    from minestudio.models.vpt.body import VPTPolicy #load_vpt_policy
    # model_path = '/alex/models/foundation-2x'
    # weights_path = '/alex/models/foundation-2x/model.safetensors'
    # policy = load_vpt_policy(model_path, None)
    # return policy
    return VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.foundation_model_2x")


def env_generator():
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        SummonMobsCallback, 
        MaskActionsCallback, 
        RewardsCallback, 
        CommandsCallback, 
        JudgeResetCallback,
        FastResetCallback
    )
    sim = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        action_type = "agent",
        timestep_limit=1000,
        callbacks=[
            SummonMobsCallback([{'name': 'cow', 'number': 10, 'range_x': [-5, 5], 'range_z': [-5, 5]}]),
            MaskActionsCallback(inventory=0), 
            RewardsCallback([{
                'event': 'kill_entity', 
                'objects': ['cow'], 
                'reward': 1.0, 
                'identity': 'chop_tree', 
                'max_reward_times': 30, 
            }]),
            CommandsCallback(commands=[
                '/give @p minecraft:iron_sword 1',
                '/give @p minecraft:diamond 64',
                '/effect @p 5 9999 255 true',
            ]),
            FastResetCallback(
                biomes=['plains'],
                random_tp_range=1000,
            ),
            JudgeResetCallback(600),
        ]
    )
    return sim