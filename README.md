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
> **Note:**  
> This integrated simulation is a continuation of the [CarbonCast](https://github.com/carbonfirst/CarbonCast.git) project and already includes the trained models required to make carbon intensity predictions.  
> 
> If you wish to view the raw weather and emissions data, please visit the CarbonCast GitHub repository linked above.
>
> This repository only contains the predicted forecast data for **Australia** and **The Netherlands**.  
> 
> To generate forecast data for other regions:
> 1. First, download the corresponding raw data from the [CarbonCast repository](https://github.com/carbonfirst/CarbonCast.git).
> 2. Then, run the `secondTierForecasts.py` script using the following syntax:

```bash
python3 secondTierForecasts.py <configFileName> <-l/-d> <-s>
```

**Parameters:**
- `configFileName`: e.g., `secondTierConfig.json` — specify your region(s) inside.
- **Regions supported**:  
  `CISO`, `PJM`, `ERCO`, `ISNE`, `NYISO`, `FPL`, `BPAT`, `SE`, `DE`, `ES`, `NL`, `PL`, `AUS_QLD`
- `-l` or `-d`: Choose whether to use **Lifecycle** (`-l`) or **Direct** (`-d`) emissions models.
- `-s`: Optional flag to **use saved models** instead of retraining.

## Citation

If you use **CarbonCast** in your research or project, please consider citing the original paper:

```
@inproceedings{maji2022carboncast,
  title={CarbonCast: multi-day forecasting of grid carbon intensity},
  author={Maji, Diptyaroop and Shenoy, Prashant and Sitaraman, Ramesh K},
  booktitle={Proceedings of the 9th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
  pages={198--207},
  year={2022}
}
```

You can also find the CarbonCast project here: [https://github.com/carbonfirst/CarbonCast](https://github.com/carbonfirst/CarbonCast)


> Example:
> ```bash
> python3 secondTierForecasts.py secondTierConfig.json -l -s
> ```


