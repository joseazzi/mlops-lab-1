# Lab 1 - Git/DVC and Data Preparation

**Name:** Jose Azzi  
**Matricule:** 233524

## Question 1

After running `uv init`, the project contains:

- **`pyproject.toml`**: contains the project configuration, such as the project name, Python version, dependencies, and build settings.
- **`README.md`**: contains documentation and information about the project.
- **`src/mlops_lab_1/`**: contains the Python source code of the project.
- **`__init__.py`**: tells Python that `mlops_lab_1` is a Python package and can contain package initialization code.

## Question 2

`dvc init` creates the `.dvc` directory and `.dvcignore`. The `.dvc/config` file stores the DVC project configuration, `.dvc/.gitignore` prevents DVC internal files and cache from being tracked by Git, and `.dvcignore` specifies files that DVC should ignore. The DVC configuration files and `.dvcignore` should be pushed to Git, while DVC cache files, temporary files, and credentials should not be pushed.

## Question 3

Because `--global` is used, the DagsHub credentials are stored in the user's global DVC configuration, outside the Git repository. Other options are `--local`, which stores settings only for the current local repository, `--system`, which applies system-wide, and using no scope flag, which stores settings in the repository's DVC configuration. Credentials should **never be pushed to GitHub** because they are sensitive information such as usernames, passwords, or access tokens.

## Question 4

After running `dvc add data`, DVC added the `data` folder to `.gitignore`. This prevents Git from tracking the actual dataset files. The dataset is managed by DVC instead, while Git tracks only the DVC metadata/pointer file.

## Question 5

Yes, the `data.dvc` file was created. It is a DVC metadata/pointer file that describes the tracked `data` folder. It contains information such as the hash, size, number of files, and path of the dataset. Git tracks this small file instead of the actual dataset.

## Question 6

On GitHub, the code and the `data.dvc` pointer file are present, but the actual dataset is not stored there because the `data` folder is ignored by Git. The `data.dvc` file points to the tracked version of the dataset. On DagsHub, the actual dataset is stored in the DVC remote after running `dvc push`.

## Question 7

After cloning the GitHub repository in a new folder, the actual `data` folder is not present because Git only tracks the `data.dvc` pointer file, not the dataset itself. The command needed to retrieve the dataset from the DVC remote is `dvc pull`.

## Question 8

No. After checking out the older Git commit and running `dvc checkout`, the `food11_processed` and `food11_processed_mini` folders should no longer be present. The older `data.dvc` file points to the previous version of the dataset, which only contained `food11_raw`.
