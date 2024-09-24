# Define the package name and requirements file path
$PACKAGE_NAME = Join-Path $PSScriptRoot "/app/streamlit_app.py"  # Python package name
$VENV_DIR = Join-Path $PSScriptRoot "venv"  # Virtual environment path relative to the script
$REQUIREMENTS_FILE = Join-Path $PSScriptRoot "requirements.txt"  # Requirements file path relative to the script

# Function to check if Python is installed and get the correct command
function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        return "python3"
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    } else {
        Write-Host "Python not found. Installing Python via Chocolatey..."
        Install-Chocolatey
        choco install python -y
        return "python"
    }
}

# Function to check if Chocolatey is installed
function Install-Chocolatey {
    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
        Write-Host "Chocolatey not found. Installing Chocolatey..."
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    } else {
        Write-Host "Chocolatey is already installed."
    }
}

# Function to create and activate virtual environment
function Setup-Venv {
    $pythonCmd = Get-PythonCommand
    if (-not (Test-Path $VENV_DIR)) {
        Write-Host "Creating virtual environment..."
        & $pythonCmd -m venv $VENV_DIR
    } else {
        Write-Host "Virtual environment already exists."
    }

    # Activate the virtual environment
    $venvActivate = Join-Path $VENV_DIR "Scripts\Activate.ps1"
    if (Test-Path $venvActivate) {
        Write-Host "Activating virtual environment..."
        & $venvActivate
    } else {
        Write-Host "Virtual environment activation script not found. Please delete venv folder and try again."
        exit 1
    }
}

# Function to install required Python packages
function Install-Requirements {
    $pythonCmd = Get-PythonCommand
    if (Test-Path $REQUIREMENTS_FILE) {
        Write-Host "requirements.txt found. Installing required Python packages in virtual environment..."
        & $pythonCmd -m pip install --upgrade pip
        & $pythonCmd -m pip install -r $REQUIREMENTS_FILE
    } else {
        Write-Host "Error: requirements.txt not found. The project cannot run without the required dependencies."
        exit 1  # Exit the script with an error
    }
}

# Function to run the Python package as a module
function Run-Script {
    Write-Host "Running Python package: $PACKAGE_NAME"
    streamlit run $PACKAGE_NAME
}

# Main logic
Get-PythonCommand
Setup-Venv
Install-Requirements
Run-Script
