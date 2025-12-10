#! /bin/bash
conda create -n minestudio python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate minestudio
conda install --channel=conda-forge openjdk=8 -y
git clone https://github.com/annastasyshyn/MineStudio
cd MineStudio 
pip install -e .
echo "Download the ,jar from here: https://drive.usercontent.google.com/download?id=1aS2eNTcxPq8E6UcK4iQaU1IIVtXkL40z&export=download" 
echo "and put it here: $(find /var/folders -type d -path "*/MineStudio/engine/build/libs" 2>/dev/null | head -n 1)"
