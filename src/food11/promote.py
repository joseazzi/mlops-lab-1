
import argparse
import math
import os

from mlflow import MlflowClient

MODEL_NAME = "food11"
TEST_SIZE = 1096


def promote(version: str, n: int) -> None:
    client = MlflowClient(
        tracking_uri=os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5002",
        )
    )

    candidate = client.get_model_version(MODEL_NAME, version)
    champion = client.get_model_version_by_alias(
        MODEL_NAME, "champion"
    )

    if candidate.version == champion.version:
        raise ValueError("Candidate is already the champion")

    candidate_run = client.get_run(candidate.run_id)
    champion_run = client.get_run(champion.run_id)

    if candidate.run_id == champion.run_id:
        raise ValueError(
            "Candidate and champion use the same training run"
        )

    candidate_acc = candidate_run.data.metrics["test_accuracy"]
    champion_acc = champion_run.data.metrics["test_accuracy"]

    se = math.sqrt(
        champion_acc * (1 - champion_acc) / n
    )
    threshold = champion_acc + 2 * se

    print(f"Champion accuracy: {champion_acc:.4f}")
    print(f"Candidate accuracy: {candidate_acc:.4f}")
    print(f"Required accuracy: {threshold:.4f}")

    if candidate_acc >= threshold:
        decision = "promoted"
        client.set_registered_model_alias(
            MODEL_NAME, "champion", version
        )
        print("PROMOTED: Candidate is the new champion")
    else:
        decision = "challenger"
        client.set_registered_model_alias(
            MODEL_NAME, "challenger", version
        )
        print("REJECTED: Candidate remains challenger")

    for key, value in {
        "decision": decision,
        "candidate_accuracy": str(candidate_acc),
        "champion_accuracy": str(champion_acc),
        "promotion_threshold": str(threshold),
    }.items():
        client.set_model_version_tag(
            MODEL_NAME, version, key, value
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--n", type=int, default=TEST_SIZE)

    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")

    promote(args.version, args.n)
