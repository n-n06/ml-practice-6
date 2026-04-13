# Bank Deposit Prediction ML Model Service
ML Model Deployment with FastAPI, Docker and additional Streamlit Demo Frontend

## Setup 

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (for local development)

### Running with Docker

#### API with Frontend 😎 

Builds and starts both the FastAPI prediction service and the Streamlit demo app.

```bash
docker compose up --build
```

| Service   | URL                        |
|-----------|----------------------------|
| API       | http://localhost:8000      |
| Docs      | http://localhost:8000/docs |
| Frontend  | http://localhost:8501      |

#### API only

If you **don't need the frontend**, start only the API service:

```bash
docker compose up api
```

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

If `model.joblib` does not exist yet, run training first:

```bash
uv run python src/model/train.py
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

No environment variables are required for local development. The model is loaded from `src/model/model.joblib` by default. If you move the model file, update `MODEL_PATH` in `src/model/config.py` accordingly.

## Project Structure
```
.
├── demo.py      
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.frontend
├── main.py
├── pyproject.toml
├── README.md
├── src
│   ├── api
│   │   ├── api.py
│   │   └── models.py
│   ├── model
│   │   ├── config.py
│   │   ├── model.joblib
│   │   ├── predict.py
│   │   ├── preprocess.py
│   └── └── train.py
└── uv.lock
```
- `demo.py` - frontend demo using Streamlit
- `docker-copmose.yml` - YAML file to run all services
- `Dockerfile` and `Dockerfile.frontend` - app containerization
- `main.py` - main API service entrypoint
- `pyproject.toml`, `uv.lock` - dependency management using `uv`
- `api/` - API code
- `model/` - Model training code
- `model/config.py` - Config variables
- `model/model.joblib` - ML model artifact
- `model/train.py` - Python module to train the model


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

The model has already been trained and the model artifact was saved as `model.joblib`.
To train the model manually and save it to `model.joblib`, run
```
uv run src/model/train.py
```

Additionally, the service automatically reruns the training logic if the model artifact is missing, 
so running training procedure manually is redundant and not recommended

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
To test it, go to `localhost:8501`. Make sure to follow the setup instructions for **API with Frontend** from the Setup section of this document/
Simply input all required fields and press predict to get the prediction from the ML API Service.


