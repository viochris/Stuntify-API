# 👶 Stuntify API: Intelligent Inference Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Deployed%20on-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)
![Status](https://img.shields.io/badge/Status-Operational-success)

## 📌 Overview
**Stuntify API** is the high-performance backend engine designed to power the Stuntify ecosystem.

Built with **FastAPI**, this service provides real-time stunting risk assessment with minimal latency. Unlike traditional scripts, this API implements a rigorous **MLOps Inference Pipeline** that strictly validates data types via Pydantic and orchestrates multiple serialized artifacts to ensure production-grade reliability.

## ✨ Key Features

### 🧠 Multi-Artifact Orchestration
The system acts as a centralized inference unit. It reconstructs the exact mathematical environment used during training by synchronizing 4 key artifacts:
* **Gender Encoder:** Translates categories (`Laki-laki`) into machine vectors.
* **Standard Scaler:** Normalizes anthropometric data (`Age`, `Height`, `Weight`) to the model's expected distribution.
* **Classifier Model:** The optimized machine learning model (`.joblib`).
* **Target Decoder:** Translates the mathematical prediction back to human-readable labels (e.g., `Severely Stunted`).

### 🛡️ Defensive Architecture (FastAPI & Pydantic)
* **Strict Schema Validation:** Uses **Pydantic Models** to enforce data integrity. If a user sends a string for `age` instead of an integer, the API rejects it immediately with a clear error message.
* **Smart Error Handling:** Global exception handlers ensure the server never crashes on bad inputs, returning meaningful HTTP 400/500 codes.
* **CORS Enabled:** Fully configured to serve requests from any frontend (React, Vue, Mobile) without cross-origin issues.

### 📖 Automatic Interactive Documentation
Because it is built on FastAPI, Stuntify provides free, interactive documentation (Swagger UI) out of the box. Developers can test endpoints directly in the browser without writing a single line of code.

## 🛠️ Tech Stack
* **Framework:** FastAPI (Asynchronous & High Performance)
* **Validation:** Pydantic
* **Computation:** NumPy, Scikit-Learn, Joblib
* **Deployment:** Hugging Face Spaces (Dockerized)

## 🚀 The Inference Pipeline
1.  **Ingestion:** API receives JSON payload -> `ConditionInput` Pydantic Model.
2.  **Validation:** Data types and required fields are verified.
3.  **Loading:** Global artifacts are loaded into memory (Lazy Loading pattern).
4.  **Preprocessing:**
    * Gender string -> Encoded Integer.
    * Numerics (Age/Height/Weight) -> Scaled Floats.
5.  **Prediction:** The model generates a class index.
6.  **Decoding:** Index -> String Label (e.g., "Normal").

## 🔌 Integration Guide (API Contract)

### Live Base URL
**[https://silvio0-stunting-api.hf.space](https://silvio0-stunting-api.hf.space)**

### Endpoint: Predict Stunting
* **URL:** `/predict-stunting`
* **Method:** `POST`

**1. The Request Body (JSON):**
```json
{
    "jenis_kelamin": "Laki-laki",
    "umur": 19,
    "tinggi": 91.60,
    "berat": 13.30
}

```

**2. The Response (JSON):**

```json
{
    "prediction": "Severely Stunted"
}

```

## 📚 How to Use the Documentation (Swagger UI)

You don't need Postman to test this API. Use the built-in Swagger UI to test directly in your browser:

1. **Access Docs:** [https://silvio0-stunting-api.hf.space/docs](https://silvio0-stunting-api.hf.space/docs)
2. **Select Endpoint:** Click on the **POST /predict-stunting** bar.
3. **Interact:** Click **"Try it out"**.
4. **Input Data:** Edit the Request Body JSON.
5. **Execute:** Click the Blue **"Execute"** button to see the real-time response.

## 📦 Local Installation

1. **Clone the Repository**
```bash
git clone https://github.com/viochris/Stuntify-API.git
cd Stuntify-API
```


2. **Install Dependencies**
```bash
pip install -r requirements.txt
```


3. **Run the Server**
```bash
uvicorn app:app --reload
```


*Output: Uvicorn running on http://127.0.0.1:8000*

---

**Author:** [Silvio Christian, Joe](https://www.linkedin.com/in/silvio-christian-joe)
*"Code that speaks JSON, Logic that saves lives."*
