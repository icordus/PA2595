# CI Workflow and Raspberry Pi Runner Instructions

## Project

**Course:** PA2595 - Machine Learning Engineering  
**Project:** A Reproducible Decision Tree Pipeline for Student Performance Risk Prediction  
**Repository:** https://github.com/icordus/PA2595  
**Workflow file:** `.github/workflows/ci.yml`

## Purpose of the CI workflow

The `ci.yml` file defines a GitHub Actions workflow used to run the machine learning pipeline on a self-hosted Raspberry Pi runner. The purpose of the workflow is to verify that the project can be executed in a reproducible way outside the developer's local machine.

The workflow is configured as a manual workflow using `workflow_dispatch`. This means it does not run automatically on every commit. Instead, it can be started manually from the GitHub Actions page when the project needs to be tested.

## Runner used

The workflow uses a self-hosted GitHub Actions runner with the label:

```yaml
runs-on: raspberrypi2
```

This means that GitHub sends the job to the Raspberry Pi machine where the runner was previously installed and registered. 
The Raspberry Pi acts as the build and test environment for the project.

## What the CI workflow does

The workflow performs the following main steps.

1. It checks out the repository content using the GitHub Actions checkout step. This downloads the project source code onto the Raspberry Pi runner.

2. It sets up Python 3.11 for the project. The workflow uses Python 3.11 because the project depends on modern Python libraries such as pandas, scikit-learn, matplotlib, joblib, Streamlit and pytest.

3. It creates a Python virtual environment. This keeps the project dependencies isolated from the system Python installation on the Raspberry Pi.

4. It installs the required dependencies from `requirements.txt`. These dependencies are needed for data processing, model training, evaluation, prediction, Streamlit execution and software testing.

5. It downloads the UCI Student Performance dataset during the CI run. The dataset is downloaded as a zip file, extracted, and the `student-mat.csv` file is copied into the project data folder. This makes the workflow reproducible because the dataset does not need to be stored manually on the runner before each run.

6. It runs the machine learning pipeline. The pipeline trains the Decision Tree model, saves the trained pipeline artifact, generates evaluation results, and runs a prediction example.

7. It uploads the generated result files from the `results/` folder as GitHub Actions artifacts. This makes files such as metrics and plots available for download from the GitHub Actions run page.

8. It runs the automated tests. The tests verify important parts of the pipeline, such as target creation, removal of the `G3` column to prevent data leakage, preprocessing, training and prediction behavior.

## Expected project outputs

After a successful CI run, the workflow should generate or verify the following outputs:

```text
models/student_performance_decision_tree.joblib
results/metrics.txt
results/confusion_matrix.png
results/decision_tree.png
```

The model artifact stores the full trained pipeline, including preprocessing and the Decision Tree model. The result files are used to support the project report and presentation.

## Raspberry Pi setup summary

The Raspberry Pi was prepared as a self-hosted GitHub Actions runner for this project. The main setup steps were:

1. Installed or verified a Linux operating system on the Raspberry Pi.
2. Installed required system tools such as Git, Python, pip, curl and unzip.
3. Created or selected a working directory for the GitHub Actions runner.
4. Downloaded the GitHub Actions self-hosted runner package for the Raspberry Pi architecture.
5. Registered the runner with the GitHub repository using the registration token from GitHub.
6. Added the runner label `raspberrypi2` so that the workflow can target this machine.
7. Started the runner service so that the Raspberry Pi can receive jobs from GitHub Actions.
8. Verified that the runner appeared as online in the repository settings.
9. Triggered the workflow manually from the GitHub Actions page.
10. Checked the job logs and uploaded artifacts after the run completed.

## Useful Raspberry Pi commands

The following commands are examples of what was needed or useful on the Raspberry Pi runner:

```bash
sudo apt update
sudo apt install -y git curl unzip python3 python3-pip python3-venv
python3 --version
git --version
```

If the GitHub runner is installed as a service, it can normally be checked with commands similar to:

```bash
user@raspberrypi2:~$ sudo systemctl status actions.runner.icordus-PA2595.raspberrypi4.service
```

To enable the runner service to start automatically on boot and to start it immediately, use:

```bash
user@raspberrypi2:~$ sudo systemctl enable actions.runner.icordus-PA2595.raspberrypi4.service

user@raspberrypi2:~$ sudo systemctl start actions.runner.icordus-PA2595.raspberrypi4.service
```

The exact service name may differ depending on how the runner was installed.

## Runner Capabilities and Tools

The Raspberry Pi runner is equipped with additional container orchestration and deployment tools: Kubernetes, Docker, and Helm.

```bash
user@raspberrypi2:~$ helm version
version.BuildInfo{Version:"v3.20.2", GitCommit:"8fb76d6ab555577e98e23b7500009537a471feee", GitTreeState:"clean", GoVersion:"go1.25.9"}

user@raspberrypi2:~$ kubectl version
Client Version: v1.28.5
Kustomize Version: v5.0.4-0.20230601165947-6ce0bf390ce3
Server Version: v1.28.15

user@raspberrypi2:~$ docker version
Client: Docker Engine - Community
 Version:           24.0.6

Server: Docker Engine - Community
 Engine:
  Version:          24.0.6
```

These tools enable the runner to support workflows that involve container deployment, Kubernetes orchestration, and Helm chart management in addition to the standard machine learning pipeline execution.

## Notes about reproducibility

The CI workflow improves reproducibility because the project can be executed from a clean checkout using documented steps. The workflow installs dependencies, downloads the dataset, trains the model, runs prediction and executes tests in a controlled environment.

This supports the Machine Learning Engineering goal of making the project more repeatable, testable and easier to validate.

## Important note

The CI workflow matches the actual project file names in the current project structure. The main executable modules used are:

```bash
python -m src.preprocess
python -m src.train --data data/raw/student-mat.csv
python -m src.predict --data data/raw/student-mat.csv
python -m pytest -q
```

All these modules exist in the project and are called with the correct command-line arguments as specified in the workflow.
