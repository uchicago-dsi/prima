# prima
Polygenic Risk and Imaging Multimodal Assessment

# chimec list
/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv

Micromamba install:
```
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -r ~/micromamba 
source ~/.bashrc
micromamba config append channels conda-forge
```

Installation recipe:
```bash
micromamba config append channels conda-forge
micromamba create -y --name prima python=3.11
micromamba activate prima
git clone git@github.com:uchicago-dsi/prima.git
cd prima
micromamba install -y pytorch torchvision pytorch-cuda=12.1 selenium firefox geckodriver -c pytorch -c nvidia
pip install -e . # if developing
# pip install . # if not developing
pip install -r requirements.txt
```
