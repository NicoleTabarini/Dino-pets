# Dino-pets: DINOv2 + LoRA on Oxford-IIIT Pet

Use Docker Compose for training and tests. PyTorch is installed in the image;
no local Python or PyTorch installation is needed.

## Build

Install Docker Desktop, start it, and build the CPU image:

```bash
docker compose build
```

The project directory is mounted into the container. MLflow run data and
training artifacts are kept in it. Hugging Face downloads are cached in a
named Docker volume between runs.

## Train

```bash
docker compose run --rm petlora \
  python -m train.train --dataset-name timm/oxford-iiit-pet \
  --mode lora --epochs 3 --seed 42 --val-frac 0.15
```

Training loads `timm/oxford-iiit-pet` directly from Hugging Face. The dataset is
cached in a persistent Docker volume, split into stratified training and
validation sets using the seed, and evaluated on the dataset's official test
split after training. Training and validation metrics are logged for each
epoch, and test metrics are logged at the end of every run. 


MLflow uses the SQLite database `mlflow.db` by default, stored in the mounted
project directory (at `/workspace/mlflow.db` in Docker). MLflow artifacts are
also stored in the mounted workspace. To use a remote tracking server, set
`MLFLOW_TRACKING_URI` in the host environment to a URI reachable from the
container.

## Run tests

```bash
docker compose run --rm petlora python -m pytest -q
```

GitHub Actions runs the test suite automatically on every push and pull
request.
