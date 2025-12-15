# <img src="assets/alex.png" alt="ALEX Icon" width="28" style="vertical-align: middle;"/> ALEX 1.0: Minecraft RL-micro and LLM-macro management AI agent

ALEX is a Minecraft AI agent that combines reinforcement learning (RL) for micro-level tasks and large language models (LLMs) for macro-level decision-making. ALEX utilizes the strengths of both approaches to navigate and interact with the complex Minecraft environment effectively.

<img src="assets/diagram.png" alt="ALEX Diagram" width="600"/>

## Architecture & Pipeline

ALEX implements a hierarchical agent architecture that combines vision-language models with reinforcement learning:

### Planning & Decision Layer  
The overall architecture consists of several experts (2 LLM ones, and a heuristic-based one):
- Meta-planner coordinates as a heuristic-based completition evaluator
- Planner uses LLMs (via HF Inference API) to generate high-level action plans
- Reflex manager handles immediate threats and opportunities (low health, hostile mobs, valuable items)

### Vision Processing Layer  
Pipeline has a vision perception layer which utilizes MineCLIP encoders for visual scene understanding and spatial attention
- Multi-modal vision queries for object detection, inventory analysis, and environment perception
- Scene analyzer that extracts structured game state from raw observations
- Spatial attention concept which splits the image into patches, and inferences MineCLIP on them to understand the image better


### Knowledge base
The knowledge system is integrated within the pipeline to not hallucinate on minecraft prompts and commands
- RAG system retrieves relevant Minecraft wiki information for decision-making
- Vector-based retrieval using wiki dataset for crafting recipes, mob behaviors, and game mechanics
- Prompt engineering with few-shot examples for consistent LLM outputs

### Execution Layer
The RL-based executor which utilizes STEVE-1 model
- STEVE-1 policy executor translates natural language commands into low-level actions
- VPT-based vision-to-action model for fine-grained control
- Action sequence generation with temporal consistency

The system operates on two timescales: fast reactive reflexes (every step) and complex planning (every 50-100 steps), enabling both tactical responsiveness and strategic goals planning.

## Prerequisites and installation

Clone the repository:
```bash
git clone --recursive https://github.com/n1n1n1q/alex.git
cd alex
``` 

Download MineCLIP weights ([avg.pth](https://drive.google.com/file/d/1mFe09JsVS5FpZ82yuV7fYNFYnkz9jDqr/view) or [attn.pth](https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view)) and place them in `models/` directory.

Download the MineDojo Wiki dataset:
```bash
mkdir -p data
wget -c --show-progress "https://zenodo.org/record/6640448/files/wiki_samples.zip" -O data/wiki_samples.zip
unzip data/wiki_samples.zip -d data && rm data/wiki_samples.zip
```

Put WANDB_API_KEY into `.env` file.

## Environment setup

### Docker setup

Build docker with   
```bash
chmod +x docker/build.sh
./docker/build.sh
```

Run the environment with
  
```bash
chmod +x docker/run.sh
./docker/run.sh
```

If no CUDA GPU available, use the no GPU version:

```bash
chmod +x docker/run_no_gpu.sh
./docker/run_no_gpu.sh
```

### Conda setup

```bash
conda create -n minestudio python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate minestudio
conda install --channel=conda-forge openjdk=8 -y
```

### Setup for MacOS with Conda

```bash
chmod +x install_macos.sh
```

After setting up your appropriate environment, install submodules:

```bash
cd alex
pip install -e submodules/MineStudio 
pip install -e submodules/MineCLIP
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## References

- [MineStudio](https://github.com/CraftJarvis/MineStudio)
- [MineCLIP](https://github.com/MineDojo/MineCLIP)
- [MineDojo](https://github.com/MineDojo/MineDojo)
- [STEVE-1](https://github.com/Shalev-Lifshitz/STEVE-1)
- [VPT](https://github.com/openai/Video-Pre-Training)
- [MineRL](https://github.com/minerllabs/minerl)
- [Malmo](https://github.com/microsoft/malmo)

## Contributors

- [Oleh Basystyi](github.com/n1n1n1q)
- [Anna Stasyshyn](github.com/annastasyshyn)
- [Maksym Zhuk](github.com/Zhukowych)
- [Zakhar Tepliakov](github.com/xpxzpvck)
- [Danyil Ikonnikov](github.com/Danylo-Ik)
- [Bozhena Sai](https://github.com/BozhenaSai)
