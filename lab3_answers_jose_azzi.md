# Lab 3 Answers — Jose Azzi

## Question 1

My registered model was assigned **version 1**.

A logged model artifact belongs to one specific MLflow training run. It stores the model produced by that run and preserves its relationship with the run's parameters and metrics. A registered model gives the model a shared name, version number, aliases, and a lifecycle that can be managed separately from the original run.

## Question 2

MLflow replaced the deprecated built-in stages such as `Staging` and `Production` with user-defined **aliases**, commonly names such as `champion` and `challenger`. Tags can also store additional information.

Model versions preserve the exact history of the models produced by different runs. An alias is more flexible because it is a mutable pointer. For example, `champion` can be moved from version 1 to version 2 without changing the serving code or deleting either version.

## Question 3

Using `models:/food11@champion` lets the application obtain the model through the MLflow Model Registry instead of depending on a specific local `.pth` file path. MLflow resolves the registered model version and understands its model flavor and required files.

To serve a newer model, I only need to move the `champion` alias to the new registered version. The serving code can continue using the same model URI.

## Question 4

Copying `pyproject.toml` and `uv.lock` before the source code allows Docker to cache the dependency-installation layer. Dependencies change less often than application code.

If I only modify a line in `serve.py`, Docker reuses the cached Python environment and dependencies. It only rebuilds the small layer that copies `src`, which makes rebuilding much faster.

## Question 5

I compared the builder stage, which represents a naive image that retains the build tools, with the final multi-stage runtime image.

- Builder/naive-style image content size: **444 MB**
- Multi-stage runtime image content size: **413 MB**
- Difference: **31 MB**, approximately **7% smaller**
- Local disk usage was approximately **2.12 GB** for the builder image and **1.98 GB** for the runtime image.

The `docker history` output showed that the largest layer was the copied `.venv`, with an uncompressed size of approximately **1.41 GB**. This layer is large because it contains PyTorch, MLflow, and the other Python dependencies.

## Question 6

Without `.dockerignore`, Docker must send unnecessary files as part of the build context. This makes builds slower, uses more storage, reduces cache efficiency, and may accidentally include private or local files in the image.

Large folders such as `data`, `mlruns`, `.git`, and `.venv` would make the build context much larger. Sending them alone does not necessarily break the build, but copying the host `.venv` into the image could break the application because it may contain macOS or architecture-specific binaries. Local MLflow files could also introduce machine-specific paths and unnecessary state.

## Question 7

Inside a container, `127.0.0.1` refers to the container itself, not to the Mac host. Therefore, the container cannot use `127.0.0.1:5001` to reach the MLflow server running on the Mac.

On Docker Desktop, `host.docker.internal` resolves to the host machine's network gateway. It allows the container to connect to services running on the Mac.

## Question 8

I stopped the container and started a new container from the same image without rebuilding it. The API and model loaded successfully.

This shows that the serving code, Python environment, and dependencies are baked into the Docker image. The model itself is resolved at runtime through the MLflow Model Registry and loaded from the mounted `mlruns` artifact directory. Therefore, the container still needs access to the MLflow server and model artifacts.

## Question 9

The Dockerfile is stored in Git, but the locally built Docker image is not yet available to other machines.

The missing step is to publish the image to a container registry such as Docker Hub or GitHub Container Registry. It should use an immutable version tag or image digest instead of relying only on `latest`. A CI pipeline could build, test, tag, and push the image reproducibly.

Because this application loads the model at runtime, another machine would also need access to a shared MLflow server and remote artifact storage instead of depending on my Mac's local `mlruns` directory.
