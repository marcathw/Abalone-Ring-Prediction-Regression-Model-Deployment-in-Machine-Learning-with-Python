# 🐚 Abalone Age Prediction Using Linear Regression and FastAPI

This project predicts the rings of abalone based on physical measurements using a Linear Regression model. The solution is deployed as a RESTful API using FastAPI, allowing users to send input features and receive predictions in real time.

---

## 🔧 Features

- **Linear Regression Model**: Predicts the number of rings (a proxy for age).
- **Model Evaluation**:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² Score
- **FastAPI Deployment**:
  - `/predict` endpoint for real-time ring prediction.
  - JSON-based input/output format.
  - Lightweight and scalable backend service.

---

## 🧠 Concepts Used

- Regression modeling with `LinearRegression` from `scikit-learn`
- Model evaluation using MAE, MSE, RMSE, and R²
- REST API development with `FastAPI`
- Data serialization with Pydantic
- Python ecosystem: `pandas`, `scikit-learn`, `uvicorn`, `fastapi`
