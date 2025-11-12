# Customer Churn Prediction App 📊

Predict which customers are likely to churn using machine learning (Random Forest). This interactive app allows you to enter customer details and see real-time churn predictions.

---

## 🔹 Features

- Input customer information (tenure, contract type, internet service, etc.)  
- Predict churn probability using a trained Random Forest model  
- Highlight high-risk customers for proactive intervention  
- Visualize **top 10 feature importances**  
- Streamlit web app

---

## Technologies & Tools

- Python  
- Streamlit  
- Pandas & NumPy  
- Scikit-learn (Random Forest, GridSearchCV)  
- Matplotlib & Seaborn (visualizations)  
- Joblib (model serialization)

---

## 📁 Files in this Repo

- `streamlit_app.py` : Streamlit application for live predictions
- `Customer_Churn_Modeling.ipynb` : notebook, creation of model, and analysis/exploration
- `churn_model.pkl` : Pre-trained Random Forest model  
- `feature_columns.pkl` : Feature list used for input DataFrame  
- `requirements.txt` : Python dependencies  
- `README.md` : Project overview  

---

## Visit site
https://customer-churn-prediction-model-liz.streamlit.app/

## OR

---

## How to Run Locally

- Clone the repository:
```bash
git clone https://github.com/lizbethsarabia/customer-churn-app.git
cd customer-churn-app

pip install -r requirements.txt

streamlit run streamlit_app.py











