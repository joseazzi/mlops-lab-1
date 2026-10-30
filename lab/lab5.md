# Lab 5 - CI/CD with GitHub Actions

## Question 1

I wouldn't use the real MLflow model in unit tests because then the tests would depend on things outside the code I'm trying to check. For example, MLflow might be down, the model files might not be accessible, or loading the model might take too long. In those cases the test could fail even if the API code itself is correct. Using a fake model avoids that problem.

## Question 2

The test with FakeModel checks that /predict accepts an image and returns a response with the expected JSON fields, including a category and a confidence value between 0 and 1. It also checks the way the API handles the model's output. It doesn't tell us if the real ResNet18 model classifies food images correctly.

## Question 3

The check that caught the duplicate was the one comparing image hashes between the training, validation and evaluation folders.

When I copied a training image into the evaluation folder, the validator failed and returned exit code 1. I removed the extra image afterward.

One thing I noticed is that the validation was **already failing before I added the image**. It reported 43 duplicate pairs in the mini dataset, then 44 after the extra copy. So the duplicate check works, but there are still duplicates in the dataset that need to be fixed.

This matters because a model can get a better test score if it sees the same image during training and evaluation. A damaged image is easier to notice because it might cause the program to crash, while a duplicate can go unnoticed and give a misleading result.

## Question 4

I would run validate_data.py in two places:

1. On my laptop, after preparing the images and before starting training.
2. In the training pipeline, after downloading the dataset from DVC and before running train.py.

That way, the data is checked where it is actually available, even if the GitHub runner doesn't have access to it.

## Question 5

needs: test means the smoke job only runs if test succeeds. If a unit test fails, GitHub skips the smoke test. Without needs, both jobs could start independently, so Docker might be built even though the unit tests have already failed.

I saw this when I intentionally broke a test in my pull request: test failed and smoke was skipped.

## Question 6

One of the earlier successful CI runs took **2 minutes 37 seconds**. For the cache experiment, I made two empty commits on the lab5-cache-experiment branch, so the code stayed the same.

| Run | Run ID | Total time | Result |
| --- | --- | --- | --- |
| First | 41119234219 | 78 seconds | Passed |
| Second | 41119478605 | 84 seconds | Passed |

The second run was actually **6 seconds slower**. I expected caching to help, but the total workflow time depends on more than downloading dependencies. Runner startup, scheduling and the Docker build can also change the time.

I didn't record the duration of each individual uv sync step, so I can't say for sure which step was faster. I can only conclude that the overall second run wasn't faster.

## Question 7

Using the commit SHA as a Docker tag makes it easier to know exactly which code is inside an image. For example, I pulled this image from GHCR:

ghcr.io/joseazzi/mlops-lab-1:cbcdeca9bf009897eb8e6b8cc08006f3d61c5afc

With latest, two people could pull the image at different times, before and after a new version is published, and end up running different code. The SHA avoids that confusion because it refers to a specific commit.

## Question 8

The push job needs permission to read the repository and upload an image to GHCR. That's why we used contents: read and packages: write.

I think it's safer to give a workflow only the permissions it needs. If one of its steps were compromised and had more permissions, it could potentially change repository files or publish or delete packages.

## Question 9

I created a branch and intentionally changed a test so that 0_12.jpg was expected to be Egg instead of Bread. The test failed, the smoke job was skipped, and GitHub didn't let me merge the pull request into main.

After I fixed the test and an import-order problem, both checks passed and I merged PR #1.

This is useful when several people work on the same project because everybody has to pass the same checks before merging. We don't have to depend on each person remembering to run the tests locally.

## Question 10

I tested the promotion script with two new training runs. My current champion was version 2, with a test accuracy of about **71.90%**. The mini evaluation set contained **1,096 images**.

The lab uses this rule:

\[
SE = \sqrt{\frac{p(1-p)}{n}}, \qquad
\text{minimum accuracy} = p + 2SE
\]

Using p = 0.718978 and n = 1096, I got a minimum required accuracy of about **74.61%**. So a candidate needed to improve by about **2.72 percentage points** to become champion.

| Version | Test accuracy | Result |
| --- | --- | --- |
| 2 | 71.90% | Stayed champion |
| 3 | 14.69% | Not promoted |
| 4 | 71.90% | Not promoted |

Version 3 was trained with a very small learning rate (0.00001), and its accuracy was very low. Version 4 used 0.001 and reached the same accuracy as the champion, but not enough to pass the quality gate.

After running the script, the aliases were:

- champion: version 2
- challenger: version 4

The script also saved the decision and accuracy values as tags on both candidate versions. One limitation is that the mini dataset contains duplicate images across splits, so the accuracies should be checked again after cleaning the data.

## Question 11

Two ways to run promote.py automatically after training would be:

1. Use a self-hosted runner on a computer that can access my local MLflow server. It could run the promotion script after training and registration.
2. Use a hosted MLflow server, for example the one provided by DagsHub, so a CI job can connect to it and run the script.

Right now I still start the training and promotion commands manually.

## Question 12

Here is a simple overview of the pipeline we used in the lab:

```text
git push / pull request
        |
        v
GitHub Actions
  1. Ruff + pytest
  2. Build Docker image
  3. Smoke test (/health)
  4. Push image to GHCR (only on main)
        |
        v
Docker image tagged with the commit SHA
        |
        v
Pull and run the image on a machine (manual)
        |
        v
API serves the model marked as champion in MLflow
```

The steps inside GitHub Actions run automatically. If a test fails, the next job doesn't run. For a pull request, the checks need to pass before I can merge it. I still pull and run the Docker image myself.

The model side is separate: I train a model, register it in MLflow, then run promote.py manually. The script compares its test accuracy with the current champion. If the improvement is large enough, it changes the champion alias; otherwise, it assigns challenger to the candidate.

The Docker image's **commit SHA** tells me which version of the code I'm using. The **MLflow alias** tells the API which model version to load. So updating the code doesn't automatically mean changing the model.
