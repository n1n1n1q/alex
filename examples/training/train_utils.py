

def policy_generator():
    from minestudio.models.vpt.body import VPTPolicy #load_vpt_policy
    # model_path = '/alex/models/foundation-2x'
    # weights_path = '/alex/models/foundation-2x/model.safetensors'
    # policy = load_vpt_policy(model_path, None)
    # return policy
    # return VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_from_early_game_2x")
    return VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.foundation_model_1x")



def env_generator():
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        SummonMobsCallback, 
        MaskActionsCallback, 
        RewardsCallback, 
        CommandsCallback, 
        JudgeResetCallback,
        FastResetCallback,
        RecordCallback,
        HardResetCallback,
        # BarrierBoxCallback,
    )
    sim = MinecraftSim(
        obs_size=(128, 128), 
        preferred_spawn_biome="plains", 
        action_type = "agent",
        timestep_limit=1000,
        callbacks=[
            # HardResetCallback(spawn_positions=[{"seed": 67, "position": [0, 70, 0]}]),
            # HardResetCallback(spawn_positions=[{"seed": 935877912, "position": [0, 70, 0]}]),
            MaskActionsCallback(inventory=0), 
            RewardsCallback([
            {
                'event': 'kill_entity', 
                'objects': ['cow'], 
                'reward': 10.0,
                'identity': 'kill_cow', 
                'max_reward_times': 50, 
            },
            {
                'event': 'damage_dealt',
                'objects': ['cow'],
                'reward': 1.0,
                'identity': 'damage_cow',
                'max_reward_times': 500,
            },
            {
                'event': 'pickup',
                'objects': ['leather', 'beef'],
                'reward': 1.0,
                'identity': 'pickup_cow_drops',
                'max_reward_times': 100,
            },
            {
                'event': 'mobs',
                'objects': ['cow'], 
                'reward': -0.1,
                'identity': 'see_cow', 
                'max_reward_times': 500, 
            },
            {   
                'event': 'mine_block',
                'objects': ['*'],
                'reward': -1,
                'identity': 'breaking_blocks',
                'max_reward_times': 200,
            },
            ]),
            CommandsCallback(commands=[
                # '/fill -12 64 -12 12 75 12 minecraft:air',
                
                # # Build 25x25 bedrock box (walls are OUTSIDE the playable area)
                # # North wall (at z=-13, OUTSIDE the playable area)
                # '/fill -13 64 -13 13 75 -13 minecraft:bedrock',
                # # South wall (at z=13, OUTSIDE the playable area)
                # '/fill -13 64 13 13 75 13 minecraft:bedrock',
                # # West wall (at x=-13, OUTSIDE the playable area)
                # '/fill -13 64 -13 -13 75 13 minecraft:bedrock',
                # # East wall (at x=13, OUTSIDE the playable area)
                # '/fill 13 64 -13 13 75 13 minecraft:bedrock',
                
                # # Optional: Add a safe grass/dirt floor
                # '/fill -12 63 -12 12 63 12 minecraft:grass_block',
                
                '/give @p minecraft:iron_sword 1',
                '/effect @p 5 9999 255 true',
                # '/effect clear @p',
            ]),
            SummonMobsCallback([{'name': 'cow', 'number': 20, 'range_x': [-10, 10], 'range_z': [-10, 10]}]),
            FastResetCallback(
                biomes=['plains'],
                random_tp_range=0,
            ),
            JudgeResetCallback(600),
            # BarrierBoxCallback(size=25, height=10, block_type='bedrock'),
            RecordCallback('./records/kill_cow_x1_penalized_balanced'),
        ]
    )
    return sim