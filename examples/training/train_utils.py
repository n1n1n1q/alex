

def policy_generator():
    from minestudio.models.vpt.body import VPTPolicy #load_vpt_policy
    # model_path = '/alex/models/foundation-2x'
    # weights_path = '/alex/models/foundation-2x/model.safetensors'
    # policy = load_vpt_policy(model_path, None)
    # return policy
    return VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.foundation_model_3x")


def env_generator():
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        SummonMobsCallback, 
        MaskActionsCallback, 
        RewardsCallback, 
        CommandsCallback, 
        JudgeResetCallback,
        FastResetCallback,
        RecordCallback
    )
    sim = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        action_type = "agent",
        timestep_limit=1000,
        callbacks=[
            SummonMobsCallback([{'name': 'cow', 'number': 20, 'range_x': [-10, 10], 'range_z': [-10, 10]}]),
            MaskActionsCallback(inventory=0), 
            RewardsCallback([
            {
                'event': 'kill_entity', 
                'objects': ['cow'], 
                'reward': 5.0,
                'identity': 'kill_cow', 
                'max_reward_times': 50, 
            },
            {
                'event': 'damage_dealt',
                'objects': ['cow'],
                'reward': 0.1,
                'identity': 'damage_cow',
                'max_reward_times': 500,
            }
        ])
            CommandsCallback(commands=[
                '/give @p minecraft:netherite_sword 1',
                # '/give @p minecraft:diamond 64',
                '/effect @p 5 9999 255 true',
                # '/effect clear @p',
            ]),
            FastResetCallback(
                biomes=['plains'],
                random_tp_range=1000,
            ),
            JudgeResetCallback(600),
            RecordCallback('./records/kill_cow'),
        ]
    )
    return sim