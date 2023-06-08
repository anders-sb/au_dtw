# BSc Thesis - DTW

## Getting started
To get started, it is recommended to set up a virtual environment, for example using a conda installation like [Miniconda](https://docs.conda.io/en/latest/miniconda.html). The following is based on a Miniconda Python 3.10 installation on Windows.

### Visual Studio Code
To set up a virtual environment using Conda from Visual Studio code, follow these steps:
1. Install the **[Python Environment Manager](vscode:extension/donjayamanne.python-environment-manager)** extension in Visual Studio Code.
2. Open the repository in Visual Studio Code.
3. Open [`requirements.txt`](requirements.txt).
4. Click the **Create new environment...** button.
5. Select **Conda** in the menu that appears.
6. Select **Python 3.10**  

A new folder called `.conda` will now be created in the root folder of the repository.  

To complete the setup of the virtual environment, follow these last steps:
1. Open the Python Environment Manager from the left sidebar in Visual Studio Code.  
In the **Conda** section, a new environment called `.conda` should appear. If it does not, click the refresh button. This might take a while.
2. Click the **Set as active workspace interpreter** button of the `.conda` environment (the rightmost button).
3. Click the **Open in Terminal** button to open an Miniconda Prompt.  
If you get an error, make sure you have selected `cmd` as your default terminal profile. To change this, press `CTRL + SHIFT + P` and type `Select Default Profile`, then select `Command Prompt`. Try to click **Open in Terminal** again.
4. In the conda prompt, you should now see `(<path of your virtual enviroment>) <path of your repository>`. If you see `(base) <path of your repository>`, type `conda activate <path of your virtual environment>`. 
5. Type `pip install -r requirements.txt` to install all required packages in the virtual environment.  

You are now able to run Python files as usual. If you get an error when running a Python file, make sure that you have activated the (correct) environment.

### Command line
To set up a virtual environment using Conda from the command line, follow these steps:
1. In a terminal type `conda -V`. If you get an error, open the **Anaconda Prompt**. Otherwise, continue in the same terminal.
2. In the terminal, navigate to the root folder of this repository.
3. Type `conda create --prefix .conda python=3.10` and follow the instructions.
4. In the conda prompt, you should now see `(<path of your virtual enviroment>) <path of your repository>`. If you see `(base) <path of your repository>`, type `conda activate <path of your virtual environment>`. 
5. Type `pip install -r requirements.txt` to install all required packages in the virtual environment.  

With the virtual environment activated, you are now able to run Python files as usual. If you get an error when running a Python file, make sure that you have activated the (correct) environment.