# Ion Channels Course - CSHL
Python material for the Cold Spring Harbor Laboratory (CSHL) course:  
*Ion Channels in Synaptic & Neural Circuit Physiology*

## Installation instructions (for instructors and TAs)
### For each of the lab desktop computers, run the following steps:
### 1. Install VSCode 
- Got to https://code.visualstudio.com/
- Download the installer
- If on Windows: run the installer (default options are fine)
- If on Mac: just move the app to the Applications folder
- Pin VSCode to the taskbar at the bottom for easy access

### 2. Install Anaconda on all of the lab desktop computers
Anaconda is a program that makes it easy to install Python (and various dependencies/libraries), which we will be using for this course. 

If you already see things related on Anaconda (e.g. the Anaconda-Navigator) on your computer, you can skip this step.
 - Go to: https://www.anaconda.com/download/success
 - Download the "Distribution" installer (on the left)
 - Run the installer (default options are fine)

 ### 3. Install git on all of the lab desktop computers
 Git is a program that is designed to keep track of changes in code, and makes it easy to download code from GitHub.

 - Go to: https://git-scm.com/downloads
 - **If on Windows:** download the Windows installer (if you are on Windows desktop)
    - Run the installer (default options are fine) 
 - **If on Mac**:
    1. Install Homebrew (if you don't already have it). Open the Terminal and paste the following command:
        ```
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        ```
    2. Now we can install git by entering in the terminal:
        ```
        brew install git
        ```
    3. You might also need to install a couple other packages for the images to render properly:
        ```
        conda install -c conda-forge nodejs
        ```
        ```
        conda install jupyter
        ```
    

### 4. Download (git clone) this code repository to each computer
- Make a new folder on the Desktop called "IonChannelsAnalysisCode" (exact name not important)
- Open Visual Studio Code (VSCode)
- From within VSCode, click on the folders tab (first tab on the left) and open the folder "IonChannelsAnalysisCode" (navigate to where you saved it)
- Open the terminal from within VSCode (View > Terminal, or use the keyboard shortcut ctrl+`) 
- Verify that git is installed correctly by typing (in the VSCode terminal):
```
git --version
```
- If you see a version number, then git is installed correctly!

    (If you don't see a version number, then you need to uninstall and reinstall git.)

- Now that git is installed, type the following commands in the terminal:
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
    conda create --name CSHL_IonChannels python=3.11.13
    ```
    ```
    conda activate CSHL_IonChannels
    ```
    ```
    pip install -r requirements.txt
    ```

### 5. Install the "miniML" python package for miniEPSC analysis
For analysis of miniEPSCs, we are using a recent deep-learning based package called miniML.
(You can read more about it here: https://delvendahl.github.io/miniML/intro.html)

- In the terminal, navigate back up to the main folder (IonChannelsAnalysisCode):
    - if you are in the CSHL_IonChannels subfolder, type into the terminal:
    ```
    cd ..
    ```
- Now we will follow the installation instructions from the miniML page:

- In the terminal, type the following commands:
    ```
    git clone https://github.com/delvendahl/miniML.git
    ```
    Once it has finished running, you should see a new folder called "miniML" in your VSCode file window (on the left)

### 5. Running the analysis notebooks

We are done and ready to start running our analysis!

Just put your data (.abf files) into the correct subfolder inside *"/data"* and then open (double click) the notebook you want to run.

- To run code cells, you can either hit the play button to the left of the cell, or you can select the cell and press *"shift-enter"*.

- The first time you run each code notebook, you will get a popup asking to choose which Python kernel/environment (which version of Python) to use. You should select the one we just created, called *"CSHL_IonChannels"*

- You might also get a popup asking to install the *ipykernel*. Click yes/install.

Happy analyzing!

### 5. Bonus: some simple terminal commands

This is not required for the installation, but it is useful to know a few basic commands for navigating the terminal:
- **cd**: change directory
    This is the most important command. It is used to navigate the file system.
    - e.g. `cd CSHL_IonChannels`
    - e.g. `cd ..` (go up one directory)
    - e.g. `cd CSHL_IonChannels/data`
- **ls**: list files in the current directory
    - e.g. `ls`
    - e.g. `ls -l` (list files in long format)
- **pwd**: print working directory
    

