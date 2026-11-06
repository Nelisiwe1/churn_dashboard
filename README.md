# Customer Churn Dashboard

A Streamlit app that answers: **"Which customers are most likely to leave, and why?"**

It has three tabs:

- **Predict a customer**: enter a customer's details to see their churn risk score, tier, and the features pushing the score up or down
- **At-risk customers**: current customers ranked by risk, filtered by tier and contract, each with their top 3 reasons, and a CSV download
- **Churn drivers**: churn rate by contract, internet type, payment method, tenure, and more

The model is the same one as in the notebook, a balanced logistic regression (ROC-AUC ≈ 0.84). It trains on startup in about a second, so there's no model file to worry about.

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Your browser opens at http://localhost:8501. Keep `telco_churn_clean.csv` in the same folder as `app.py`.

## Deploy it for free (for your portfolio)
1. Put `app.py`, `requirements.txt` and `telco_churn_clean.csv` in a public GitHub repo
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**, choose the repo and `app.py`
3. You get a public link to put on your CV or LinkedIn

## Project files
| File | What it is |
|---|---|
| `churn_prediction.ipynb` | Full analysis: cleaning → EDA → ML → interpretation |
| `app.py` | This dashboard |
| `telco_churn_clean.csv` | Cleaned data (output of notebook Stage 1) |
