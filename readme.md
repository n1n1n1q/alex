# <img src="assets/alex.png" alt="ALEX Icon" width="28" style="vertical-align: middle;"/> ALEX 1.0: Minecraft RL-micro and LLM-macro management AI agent

ALEX is a Minecraft AI agent that combines reinforcement learning (RL) for micro-level tasks and large language models (LLMs) for macro-level decision-making. ALEX utilizes the strengths of both approaches to navigate and interact with the complex Minecraft environment effectively.

<img src="assets/diagram.png" alt="ALEX Diagram" width="600"/>

## Features

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