#!/bin/bash
# Setup script for Parking Optimization Project

echo "Setting up Parking Optimization Project..."
echo "========================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Create virtual environment
echo -e "\nCreating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\nUpgrading pip..."
pip install --upgrade pip

# Install requirements
echo -e "\nInstalling requirements..."
pip install -r requirements.txt

echo -e "\n✅ Setup complete!"
echo ""
echo "To run the project:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run demo: python main.py --mode demo"
echo "3. Or run tests: python test_system.py"
echo ""
echo "For more information, see QUICK_START.md"
