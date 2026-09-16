# Lab 2 — Model training and experiment tracking: answers

**Project:** Food-11 / `mlops-lab-1`  
**Working directory:** `/Users/joseazzi/mlops/mlops-lab-1`  
**MLflow tracking server used:** `http://127.0.0.1:5001`

## Question 1: Look at `pyproject.toml` and `uv.lock`. What changed?

`pyproject.toml` now declares the training dependencies: MLflow, PyTorch (`torch`), `torchvision`, and scikit-learn (alongside the existing DVC dependencies). `uv.lock` was regenerated to record resolved versions and their transitive dependencies so the project can reproduce the same Python package environment. I also changed the project's Python requirement to `>=3.12,<3.13` and `.python-version` to `3.12` to match the working Conda environment; those changes are separate from adding the training libraries. On this Mac, the libraries were installed in the `mlops-lab-2` Conda environment, so we did not need the optional CUDA/CPU-only `uv` index configuration shown in the handout.

## Question 2: What are `--backend-store-uri` and `--default-artifact-root` used for? How do metadata and artifacts differ?

`--backend-store-uri sqlite:///mlflow.db` tells MLflow to store experiment and run **metadata** in the local SQLite database. Metadata includes run IDs, parameters, metric values, timestamps, and artifact locations. `--default-artifact-root ./mlruns` specifies where newly created experiments store **artifacts**, such as saved model files. In short, the database records *what happened and where outputs are*; the artifact directory contains the output files themselves. The paths are relative to the repository directory where the server was started.

## Question 3: Why shouldn't `mlflow.db` and `mlruns/` be tracked by Git or DVC?

They are generated locally by training runs and change whenever a run is logged. Committing the SQLite database to Git would add a frequently changing binary file and risk merge conflicts; committing models would make Git history unnecessarily large. DVC is used here for the versioned Food-11 datasets. DVC tracking of MLflow's entire active tracking directory would duplicate MLflow's artifact management and generate a new data version for each local run. We added `mlflow.db`, its SQLite side files, and `mlruns/` to `.gitignore`.

## Question 4: What happens the first time you call `mlflow.set_experiment("food11")`?

MLflow creates an experiment called `food11` if it does not exist yet, and makes it the active experiment for subsequent runs. The first training run printed a message that `food11` did not exist and was being created; it then appeared in the MLflow UI. Later runs were logged to that same experiment.

## Question 5: How do `mlflow.log_param` and `mlflow.log_metric` differ? Why does only the metric take `step`?

`mlflow.log_param` records a configuration value fixed for the run, such as `lr=0.01` or `batch_size=32`. `mlflow.log_metric` records a numerical measurement that may change during training, such as `val_accuracy` or `train_loss`. The `step` identifies the epoch of each measurement so MLflow can plot its progression. A parameter is set once for the run, so it does not need an epoch step.

## Question 6: Where are the run's params, charts, and model artifact? Where does the model live on disk?

In the `food11` experiment, opening a run shows its parameters and metric history; the metric charts plot values logged across the five epochs. The run also links to its logged PyTorch model (in this version of MLflow, the Runs table has a **Models** column). The model's actual files are stored by the local MLflow server beneath `/Users/joseazzi/mlops/mlops-lab-1/mlruns/`, because we started that server from the repository root with `--default-artifact-root ./mlruns`. MLflow assigns the nested experiment/model directory; `mlflow.db` contains metadata, not the model weights. To locate the exact model directory on this machine, run `find mlruns -name MLmodel -print` from the repository root.

## Question 7: Which learning rate gave the best `val_accuracy`? Is higher always better?

The best observed learning rate was **0.01**, with `val_accuracy ≈ 0.679` at batch size 32. Higher is **not always** better: learning rates beyond the tested range might overshoot during optimization. The results only show that 0.01 was best among these four runs.

| Learning rate | Batch size | Final validation accuracy | Test accuracy |
| ---: | ---: | ---: | ---: |
| 0.0001 | 32 | 0.390 | 0.390 |
| 0.001 | 32 | 0.670 | 0.719 |
| **0.01** | **32** | **0.679** | **0.725** |
| 0.001 | 64 | 0.665 | 0.703 |

## Question 8: What pattern does the parallel coordinates plot show for `lr`, `batch_size`, and `val_accuracy`?

With batch size held at 32, increasing `lr` from 0.0001 to 0.001 greatly improved validation accuracy (about 0.390 to 0.670); increasing it to 0.01 improved it slightly further (about 0.679). With `lr=0.001`, increasing batch size from 32 to 64 gave a slightly lower validation accuracy (about 0.670 to 0.665). These are observations from one run per setting, so they do not establish a general rule about learning rate or batch size.

## Question 9: Which run ranked first by `val_accuracy`, and what is its run ID?

The top run was **`debonair-hare-775`** (`lr=0.01`, `batch_size=32`), with final `val_accuracy ≈ 0.679` and `test_accuracy = 0.725`.

**Run ID for the next lab:** `6b7cc0c1f81542fdbbe56bf5843cc158`
