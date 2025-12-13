def policy_generator():
    from minestudio.models.vpt.body import VPTPolicy

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
    )

    sim = MinecraftSim(
        obs_size=(128, 128),
        preferred_spawn_biome="plains",
        action_type="agent",
        timestep_limit=1000,
        callbacks=[
            MaskActionsCallback(inventory=0),
            RewardsCallback(
                [
                    {
                        "event": "kill_entity",
                        "objects": ["cow"],
                        "reward": 15.0,
                        "identity": "kill_cow",
                        "max_reward_times": 50,
                    },
                    {
                        "event": "damage_dealt",
                        "objects": ["cow"],
                        "reward": 1.0,
                        "identity": "damage_cow",
                        "max_reward_times": 500,
                    },
                    {
                        "event": "pickup",
                        "objects": ["leather", "beef"],
                        "reward": 1.0,
                        "identity": "pickup_cow_drops",
                        "max_reward_times": 100,
                    },
                    {
                        "event": "mine_block",
                        "objects": ["*"],
                        "reward": -2,
                        "identity": "breaking_blocks",
                        "max_reward_times": 200,
                    },
                ]
            ),
            CommandsCallback(
                commands=[
                    "/give @p minecraft:iron_sword 1",
                    "/effect give @p minecraft:strength 9999 5 true",
                ]
            ),
            SummonMobsCallback(
                [
                    {
                        "name": "cow",
                        "number": 20,
                        "range_x": [-10, 10],
                        "range_z": [-10, 10],
                    }
                ]
            ),
            FastResetCallback(
                biomes=["plains"],
                random_tp_range=0,
            ),
            JudgeResetCallback(600),
            RecordCallback("./records/kill_cow_x1_basic"),
        ],
    )
    return sim
