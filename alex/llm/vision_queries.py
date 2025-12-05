"""
Predefined scene queries for MineCLIP-based visual analysis.
These queries are used to classify various aspects of Minecraft environments.
"""

SCENE_QUERIES = {
    "biome": {
        "forest": "forest trees oak birch spruce",
        "plains": "flat grass plains open field",
        "desert": "desert sand cactus dry",
        "mountain": "mountain hills stone high elevation",
        "jungle": "jungle vines tropical dense trees",
        "snow": "snow ice cold frozen winter",
        "swamp": "swamp water lily pads murky",
        "ocean": "ocean water sea waves beach",
        "cave": "cave underground dark stone",
    },
    "time_of_day": {
        "day": "bright daylight sunny clear sky",
        "night": "dark night stars moon",
        "sunset": "sunset orange sky evening dusk",
        "sunrise": "sunrise morning dawn",
    },
    "weather": {
        "clear": "clear sky sunny",
        "rain": "rain raining wet weather",
        "storm": "thunderstorm lightning dark clouds",
    },
    "hostile_mobs": {
        "zombie": "zombie undead green monster",
        "skeleton": "skeleton archer bones bow",
        "creeper": "creeper green explosive monster",
        "spider": "spider black legs eight",
        "enderman": "enderman tall black purple eyes",
    },
    "passive_mobs": {
        "cow": "cow cattle brown animal",
        "sheep": "sheep wool white animal",
        "pig": "pig pink animal",
        "chicken": "chicken bird feathers",
        "horse": "horse mount riding",
    },
    "resources": {
        "wood": "trees logs wood oak birch",
        "stone": "stone rocks cobblestone",
        "ore": "ore mining iron coal diamond",
        "water": "water river lake pond",
        "crops": "wheat carrots potatoes farming",
    },
    "structures": {
        "village": "village houses buildings npcs",
        "house": "building house structure shelter",
        "cave_entrance": "cave entrance hole opening",
        "chest": "chest loot container",
    },
    "safety": {
        "safe": "peaceful calm safe area",
        "dangerous": "danger hostile threatening monsters",
        "enclosed": "enclosed protected walls ceiling",
        "exposed": "open exposed outside vulnerable",
    },
}
