## Integrated Code Instructions

To run the simulation, you must first set up a Conda environment:

### 1. Create and Activate the Environment

```bash
# Create a new environment with Python 3.10
conda create -n carboncast310 python=3.10

# Activate the environment
conda activate carboncast310
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. (Optional) VSCode Setup for Auto-Activation

To auto-activate the environment in VSCode terminal, open `.vscode/settings.json` and add:

```json
{
  "python.defaultInterpreterPath": "C:\\Users\\YourName\\anaconda3\\envs\\carboncast310\\python.exe"
}
```

> Replace `"YourName"` with your actual Windows username.

### 4. Run the Simulation

```bash
python Forecast_simulation.py
```

