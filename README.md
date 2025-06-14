# Ion Channels Course - CSHL
Python material for the Cold Spring Harbor Laboratory (CSHL) course:  
*Ion Channels in Synaptic & Neural Circuit Physiology*

## Installation instructions (for instructors and TAs)
### For each of the lab desktop computers, run the following steps:
### 1. Install VSCode 
- Got to https://code.visualstudio.com/
- Download the installer
- Run the installer (default options are fine)
- Pin VSCode to the taskbar at the bottom for easy access

### 2. Install Anaconda on all of the lab desktop computers
 - Got to https://www.anaconda.com/download/success
 - Download the "Distribution" installer (on the left)
 - Run the installer (default options are fine)

 ### 3. Install git on all of the lab desktop computers
 - Got to https://git-scm.com/downloads
 - Download the Windows installer (if you are on Windows desktop)
 - Run the installer (default options are fine) 

### 4. Download (git clone) this code repository to each computer
- Make a new folder on the Desktop called "IonChannelsAnalysisCode" (exact name not important)
- Open VSCode and open the folder
- Open the terminal (View > Terminal, or use the keyboard shortcut ctrl+`) 
- Verify that git is installed correctly by opening a terminal and typing:
```
git --version
```
- If you see a version number, then git is installed correctly.
- If you don't see a version number, then you need to uninstall and reinstall git.

- If git was installed correctly, type the following commands in the terminal:
```
git init
```
```
git clone https://github.com/argalloni/CSHL_IonChannels2025.git
```
- If the command is successful, you should see a new folder called "CSHL_IonChannels" in your VSCode window (on the left)

### 5. Install the required packages
- In the VSCode terminal, type the following commands 

    (wait for each command to finish running before moving on to the next one):
    ```
    cd CSHL_IonChannels
    ```
    ```
    conda update conda
    ```
    ```
    conda env create --name CSHL_IonChannels --file installation_requirements.yml
    ```
    ```
    conda activate CSHL_IonChannels
    ```

### 5. Install the "miniML" python package for miniEPSC analysis
- In the terminal, navigate back up to the main folder (IonChannelsAnalysisCode):
    - if you are in the CSHL_IonChannels subfolder, type:
    ```
    cd ..
    ```
- Now we will follow the installation instructions from the miniML page: (https://delvendahl.github.io/miniML/intro.html) 

- In the terminal, type the following commands:
    ```
    git clone https://github.com/delvendahl/miniML.git
    ```
    You should see a new folder called "miniML" in your VSCode file window (on the left)
- Use the terminal to navigate to the miniML folder, by typing (in the terminal window at thew bottom):
    ```
    cd miniML
    ```
- Install the miniML package (for miniEPSC analysis) by typing:
    ```
    conda create --name miniML python=3.11
    ```
    ```
    conda activate miniML
    ```


- **Note:** miniML is only installed in the local environment called "miniML". This needs to be active in order to run miniML. When running a notebook, make sure the jupyter kernel is set to "miniML".