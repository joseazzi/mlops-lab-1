# Lab 4 – Docker Compose Answers

**Student:** Jose Azzi

## Question 1

Without a mounted volume, everything written to `/mlflow-data` is stored inside the container’s writable filesystem. Stopping the container does not immediately delete the data, but removing the container does. A new container created from the same image displays a fresh MLflow UI without the previous runs, models, or artifacts.

## Question 2

A named volume is managed by Docker, survives container recreation, and does not depend on a specific directory on the host machine. It also avoids host-path and permission problems.

A bind mount would work, but it would connect MLflow storage to a specific folder in the repository or host filesystem, making the configuration less portable.

## Question 3

Docker Compose creates a private network for the services and provides DNS resolution using service names. Therefore, `mlflow` automatically resolves to the IP address of the MLflow container.

In Lab 3, the containers were not connected through this Compose network, so `mlflow` was not a known hostname. We needed `host.docker.internal` to access MLflow running on the host.

## Question 4

Reading `INFERENCE_URL` from an environment variable makes the frontend configurable.

Inside Compose, it uses:

`http://inference:8000`

If the frontend is run independently outside Compose, it can use another address such as:

`http://127.0.0.1:8000`

This can be changed without editing or rebuilding the frontend image.

## Question 5

The inference service does not publish a port because it only needs to be accessed by the frontend, not directly by the host machine.

The frontend reaches inference through the private Compose network using the service name:

`http://inference:8000`

MLflow and the frontend publish ports because they need to be opened in the host browser.

## Question 6

`depends_on` only controls container startup order. It does not guarantee that the MLflow server is ready to accept requests.

If `serve.py` tries to load the model before MLflow is ready, model loading fails and the inference process exits. This can be improved using health checks, startup retries, or a container restart policy.

## Question 7

The `docker compose ps` output showed:

- `mlflow` publishes container port 5000 as host port 5002.
- `frontend` publishes port 8501.
- `inference` has internal port 8000 but does not publish it to the host.

This matches the Compose configuration. Host port 5002 was used for MLflow because port 5000 was already occupied by a macOS service.

## Question 8

The model is loaded once when the inference service starts. Therefore, changing the `champion` alias does not immediately change the model used by an already-running inference container.

After moving `champion` from version 1 to version 2, I ran:

```bash
docker compose restart inference
```

The restarted service resolved `models:/food11@champion` again and loaded version 2.

The same bread image was still predicted as **Bread with 96.3% confidence** because versions 1 and 2 referenced the same trained model.

The lab guide uses the `Staging` stage, but the installed MLflow version uses the `champion` alias for the same purpose.

## Question 9

A rebuild is unnecessary because the Docker image contains the inference application and its dependencies, not one fixed registered-model version.

The model is fetched from MLflow when the container starts. Restarting the inference container reruns `serve.py`, resolves the current `champion` alias, and loads the model version assigned to it.

## Question 10

After running:

```bash
docker compose down
docker compose up -d
```

the registered `food11` model was still available. Versions 1 and 2 remained registered, and the `champion` alias still pointed to version 2.

This happened because `docker compose down` removed the containers and network but preserved the named volume.

Using:

```bash
docker compose down -v
```

would also delete the named volume. The next `docker compose up` would create an empty volume, so the MLflow database, registered models, aliases, runs, and artifacts would be missing. The inference service would then fail to load `food11@champion` until a model was registered again.

## Question 11

Docker Compose is mainly intended to run services on one machine. To run three inference replicas behind a load balancer and survive machine failures, an orchestrator such as Kubernetes or Docker Swarm would be needed.

The deployment would also need:

- Replica management
- Service discovery
- Load balancing
- Health checks
- Automatic recovery
- Shared persistent storage
- A network database instead of local SQLite
- Shared object storage for MLflow artifacts

These features allow services and data to remain available even if one machine fails.
