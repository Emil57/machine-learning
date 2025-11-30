# ML Playground — Poetry + DVC + CML

This repository is a minimal playground to experiment with Poetry, DVC and CML.

Quick overview:
- `src/ml_project/prepare_data.py` — generate a synthetic dataset and save `data/train.csv`.
- `src/ml_project/train.py` — trains two models and writes `models/*.joblib`.
- `src/ml_project/evaluate.py` — evaluates models and writes `metrics/metrics.json`.
- `dvc.yaml` — contains a 3-stage pipeline: `prepare`, `train`, `evaluate`.
- GitHub Actions workflow in `.github/workflows/cml.yaml` shows a CML-enabled CI example.

Local setup (Windows PowerShell):

1) Install Poetry: `pip install poetry` (or follow Poetry docs).

2) Install dependencies with Poetry:

```powershell
poetry install
```

3) Initialize DVC (one-time):

```powershell
dvc init
# optionally add a remote, e.g. local remote for testing
dvc remote add -d local_remote dvc-storage
dvc remote modify local_remote type local
dvc remote modify local_remote path ./dvc-storage
```

4) Run the pipeline locally:

```powershell
poetry run python src/ml_project/prepare_data.py
poetry run python src/ml_project/train.py
poetry run python src/ml_project/evaluate.py

# or use DVC to run the full pipeline
dvc repro
```

5) Use Git + DVC to add large files to remote storage as needed:

```powershell
git add dvc.yaml .gitignore
git add data/train.csv models metrics -f
git commit -m "Add initial ML playground and DVC pipeline"
dvc add data/train.csv
dvc push
```

Notes:
- This is intentionally minimal to serve as a playground. Replace synthetic data with real datasets and configure a cloud DVC remote (S3, GCS, Azure) for collaborative experiments.
- The GitHub workflow demonstrates how CML can be integrated; customize credentials and remotes before using in production.

Automated setup scripts
 - `scripts/setup.ps1` — PowerShell helper that installs Poetry (if missing), runs `poetry install`, installs DVC (if missing), and initializes DVC. Run:

```powershell
.\scripts\setup.ps1
# or skip Poetry and create .venv instead
.\scripts\setup.ps1 -UsePoetry:$false
```

 - `scripts/setup.sh` — POSIX shell equivalent for Linux/macOS. Run:

```bash
./scripts/setup.sh
./scripts/setup.sh --no-poetry
```

These scripts are idempotent and intended to make the environment reproducible for contributors.

## Poetry Commands Reference

Below are the main Poetry commands used to set up and manage this project's virtual environment:

### Initial Setup
```powershell
# 1. Configure Poetry to create venv in project directory (one-time)
poetry config virtualenvs.in-project true

# 2. Install all project dependencies (including dev)
poetry install

# 3. Activate the virtual environment (optional; use `poetry run` instead if preferred)
poetry shell

# 4. Verify Poetry configuration is correct
poetry check
```

### Adding/Removing Dependencies
```powershell
# Add a regular dependency
poetry add numpy scikit-learn

# Add a dev dependency (testing, linting, etc.)
poetry add --group dev pytest black flake8

# Add a specific version
poetry add "pandas>=2.0,<3.0"

# Remove a dependency
poetry remove pytest
```

### Running Commands in the Environment
```powershell
# Run a script using Poetry (without activating shell)
poetry run python src/ml_project/train.py

# Run DVC pipeline
poetry run dvc repro

# Activate shell and run commands directly
poetry shell
dvc repro
exit
```

### Dependency Management
```powershell
# Update the lock file (regenerate poetry.lock from pyproject.toml)
poetry lock

# Update lock file without changing versions
poetry lock --no-update

# Show installed dependencies
poetry show

# Show outdated dependencies
poetry show --outdated
```

### Virtual Environment Management
```powershell
# Show info about the virtual environment
poetry env info

# List all Poetry-managed environments
poetry env list

# Remove virtual environment
poetry env remove python3.13
```
