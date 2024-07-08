# CSHL
Python material for the Cold Spring Harbor Laboratory Ion Channel and Neural Circuits Summer School. 
## Installation
### Clone git repo to your computer
- You can use the desktop git : https://desktop.github.com/
- Or use the command line : 
```
cd <path where you want to store the CSHL folder>
```
```
git clone https://github.com/afmontarras/CSHL.git
```
### Copy the folder (create your playground)
During the course, the reference notebooks will be updated daily following your progress. You will want to create/modify them yourself day by day. So copy the cloned folder so that your own changes will not be affected by updating (pull) the git.
### Environment Manager
You need to install Miniforge:
- Miniforge/mamba: https://github.com/conda-forge/miniforge (https://mamba.readthedocs.io/en/latest/index.html)

### Create your CSHL environment
 Write in your terminal:
```
cd <path to CSHL directory>
```
```
conda update -n base -c conda-forge conda
```
```
conda env create --name CSHL --file CSHL.yml
```
This will create a CSHL python environment based on the CSHL.yml file present in the CSHL folder. It contains every package you need to start working with th notebooks. You need to activate the environment to work with it: `conda activate CSHL`.

### Launch jupyter lab
```
conda activate CSHL
jupyter lab
```
## Update actions : 
### - CSHL git repo
```
cd <path to CSHL directory>
git pull
```

### - CSHL environment
You need to be on the base environment (easiest way is to open a new terminal): 
```
conda env update --name CSHL --file CSHL.yml
```