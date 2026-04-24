# Bank Deposit Prediction ML Model Service
ML Model Deployment with FastAPI, Docker, MLflow and additional Streamlit Demo Frontend.

The full training code is available in Google Colab Notebook [here](https://colab.research.google.com/drive/1NXmAicQQx88PZmK4BBmq1l6sqNFkoOHW?usp=sharing).
Check it out if you are interested.

## Setup 
TL;DR Version (Linux / MacOS):
```bash
# 1. Clone the repo
git clone https://github.com/n-n06/ml-practice-6
cd ml-practice-6

# 2. Dir setup (so that Docker does not break)
mkdir -p mlruns
touch mlflow.db
chmod -R u+rwX mlruns mlflow.db

# 3. Create the .env file
cat > .env << 'EOF'
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_PORT=5000
API_PORT=8000
FRONTEND_PORT=8501
EOF

# 4. Build and start all services
docker compose up --build
```

For Windows:
```powershell
# 1. Clone the repo
git clone https://github.com/n-n06/ml-practice-6
cd ml-practice-6

# 2. Create the .env file
@"
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_PORT=5000
API_PORT=8000
FRONTEND_PORT=8501
"@ | Out-File -FilePath .env -Encoding utf8

# 3. Build and start all services
docker compose up --build
```



### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (for local development)

### Running with Docker

First of all, clone the repo.

```bash
git clone https://github.com/n-n06/ml-practice-6
cd ml-practice-6
```

#### API with Frontend 😎 

Builds and starts the FastAPI prediction service, the Streamlit demo app, and the MLflow tracking server.

```bash
docker compose up --build
```

This might take couple of minutes to install all dependencies using `uv`!

| Service   | URL                        |
|-----------|----------------------------|
| API       | http://localhost:8000      |
| Docs      | http://localhost:8000/docs |
| Frontend  | http://localhost:8501      |
| MLflow UI | http://localhost:5000      |



### Running Locally

#### 1. Clone the repo
```bash
git clone https://github.com/n-n06/ml-practice-6
cd ml-practice-6
```
#### 2. Install dependencies

```bash
uv sync
```

#### 3. Train the model

Training logs parameters, metrics, and the model artifact to MLflow. Make sure the MLflow server is running (or set `MLFLOW_TRACKING_URI` to a reachable instance), then:

```bash
uv run python -m src.model.train
```

#### 4. Start the API

```bash
uv run main.py
```

#### 5. Start the frontend (optional)

```bash
uv run demo.py
```



### Environment

The service expects environment variables.
Create a `.env` file in the project root with the following variables:

```env
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_PORT=5000
API_PORT=8000
FRONTEND_PORT=8501
```
When running via Docker Compose, this is set automatically. For local development, export them manually:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_PORT=5000
export API_PORT=8000
export FRONTEND_PORT=8501
```





## Project Structure
```
.
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.frontend
├── main.py
├── pyproject.toml
├── README.md
├── src
│   ├── api
│   │   ├── api.py
│   │   └── schemas.py
│   ├── model
│   │   ├── config.py
│   │   ├── mlflow_utils.py
│   │   ├── predict.py
│   │   ├── preprocess.py
│   │   └── train.py
│   └── ui
│       └── demo.py
└── uv.lock
```
- `docker-compose.yml` - YAML file to run all services (API, frontend, MLflow, trainer)
- `Dockerfile` and `Dockerfile.frontend` - app containerization
- `main.py` - main API service entrypoint
- `pyproject.toml`, `uv.lock` - dependency management using `uv`
- `api/` - API code (FastAPI + Pydantic schemas)
- `model/` - Model training, preprocessing, and inference code
- `model/mlflow_utils.py` - MLflow integration logic (`@mlflow_run` decorator)
- `ui/demo.py` - frontend demo using Streamlit


## Dataset and Model
### Dataset
This is the classic marketing bank dataset uploaded originally in the UCI Machine Learning Repository. 
This dataset contains banking marketing campaign data and we can use it to optimize marketing campaigns to attract more customers to **term deposit subscription.**

> **A term deposit** is a type of deposit account held at a financial institution where money is locked up for some set period of time.
Term deposits are usually short-term deposits with maturities ranging from one month to a few years.
Typically, term deposits offer higher interest rates than traditional liquid savings accounts, whereby customers can withdraw their money at any time.

#### Dataset columns:
- Age:	Age of customer
- Job:	Job of customer. Categorical variable with values: 'admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed','unknown'
- Marital:	Marital status of customer. Categorical variable with values: 'divorced', 'married', 'single',  (note: 'divorced' means divorced or widowed)
- Education:	Customer education level. Categorical variable with values: 'secondary', 'tertiary', 'primary' and 'unknown'
- Default:	Has credit in default?
- Housing:	If costumer has housing loan
- Loan:	Has Personal Loan
- Balance:	Customer's individual balance
- Contact:	Communication type
- Month:	Last contact month of year
- Day:	Last contact day of the week
- Duration:	Last contact duration, in seconds
- Campaign:	Number of contacts performed during this campaign and for this client
- Pdays: Number of days that passed by after the client was last contacted from a previous campaign
- Previous:	Number of contacts performed before this campaign and for this client
- Poutcome:	outcome of the previous marketing campaign
- **Deposit**:	has the client subscribed a term deposit (Target value)

### Model
The model used is XGBoost that showed the best results during inital training compared to other models.
The objective of the model is to accurately predict if a customer will subscribe to a term deposit with a final goal of
optimizing bank marketing campaigns to increase the revenue of the bank.

Model artifacts, parameters, and metrics are tracked and versioned using **MLflow**. The trained model is registered
in the MLflow Model Registry under the name `bank-deposit-model` and promoted to the `production` alias, from where
the API loads it at runtime.

To train the model manually, run:
```
uv run python -m src.model.train
```

## MLflow Integration

The project uses [MLflow](https://mlflow.org/) to track experiments, log artifacts, and manage model versions.

### Architecture
- A dedicated **`mlflow`** service runs the tracking server on port `5000` with a SQLite backend store and local artifact storage.
- A **`trainer`** service runs training once on startup. It logs parameters, metrics, and the trained pipeline to MLflow,
  then registers the model and assigns it the `production` alias.
- The **`api`** service loads the model at startup from the MLflow Model Registry via `models:/bank-deposit-model@production`.

You can inspect all runs, metrics, and registered model versions at [http://localhost:5000](http://localhost:5000).

### The `@mlflow_run` Decorator

MLflow logging logic is encapsulated in `src/model/mlflow_utils.py` as a Python decorator. This keeps the training
code clean and focused on the ML logic, while all MLflow-specific concerns (starting runs, logging params/metrics,
registering models, managing aliases) live in one place.

Usage is as simple as:

```python
from src.model.mlflow_utils import mlflow_run

@mlflow_run
def train_model():
    # ... training logic ...
    return {
        "model": best_model,
        "params": best_params,
        "metrics": {"accuracy": acc, "f1": f1},
    }
```

The decorator expects the wrapped function to return a dict with `model`, `params`, and `metrics` keys. It then
handles starting a run, logging everything to MLflow, registering the model in the registry, and promoting the new
version to the `production` alias — so any future call to the API automatically picks up the latest trained model.

## API Service
The API service has 2 main endpoints
1. `GET /` to check the health of the API
2. `POST /predict` to get the prediction for the provided JSON input

To view the Swagger UI documentation, go to `/docs`.

### Prediction Endpoint
#### Input
The prediction endpoint receives the following payload
```json
{
  "age": 0,
  "job": "management",
  "marital": "married",
  "education": "primary",
  "default": "yes",
  "balance": 0,
  "housing": "yes",
  "loan": "yes",
  "contact": "telephone",
  "day": 0,
  "month": "string",
  "duration": 0,
  "campaign": 0,
  "pdays": 0,
  "previous": 0,
  "poutcome": "success"
}
```
The input is validated using Pydantic to prevent invalid payload.

#### Output
The following output that includes the prediction and deposit probability is expected:
```json
{
  "deposit_prediction": false,
  "deposit_probability": 0.16522127389907837
}
```

## Demo Frontend UI
To simplify the API service testing, I created a simple frontend using Streamlit.
To test it, go to `localhost:8501`. Make sure to follow the setup instructions for **API with Frontend** from the Setup section of this document.
Simply input all required fields and press predict to get the prediction from the ML API Service.

<img width="1091" height="859" alt="image" src="https://github.com/user-attachments/assets/b05d3805-034e-4732-8157-4dd0827b1a7d" />



## MLFlow Integration Results
The Model's runs
<img width="1910" height="752" alt="image" src="https://github.com/user-attachments/assets/48ef1d87-9862-4961-9785-6b970139b759" />

Registered model with inferred schema:
<img width="1910" height="752" alt="image" src="https://github.com/user-attachments/assets/8dd18a34-cfd7-4f67-afc1-902bfb0bc288" />

