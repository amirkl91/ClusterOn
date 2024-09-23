#!/bin/bash

# Define the package name and requirements file
PACKAGE_NAME="app"
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"

# Function to check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Python is not installed. Installing Python..."
        install_python
    else
        echo "Python is already installed."
    fi
}

# Function to install Python (for macOS and Linux)
install_python() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS installation using Homebrew
        if ! command -v brew &> /dev/null; then
            echo "Homebrew is not installed. Installing Homebrew first..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux installation using apt (Debian-based)
        sudo apt update
        sudo apt install -y python3 python3-pip
    else
        echo "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Function to create and activate a virtual environment
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    else
        echo "Virtual environment already exists."
    fi

    # Activate the virtual environment
    source "$VENV_DIR/bin/activate"
}

# Function to install required Python packages
install_requirements() {
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "requirements.txt found. Installing required Python packages in virtual environment..."
        python3 -m pip install --upgrade pip
        python3 -m pip install -r "$REQUIREMENTS_FILE"
    else
        echo "Error: requirements.txt not found. The project cannot run without the required dependencies."
        exit 1  # Exit the script with an error
    fi
}

# Function to run the Python package as a module
run_script() {
    echo "Running Python package: $PACKAGE_NAME"
    python3 -m "$PACKAGE_NAME"
}

# Main logic
check_python
setup_venv
install_requirements
run_script
