# 💻 Laptop Price Predictor

A Machine Learning web application that predicts the price of a laptop based on its configuration.
The app is built using **Python, Scikit-Learn, XGBoost, and Streamlit**.

## 🚀 Demo

This application allows users to select laptop specifications such as brand, RAM, CPU, GPU, storage, and display features to estimate the expected laptop price.

## 🧠 Machine Learning Model

The model is trained on a cleaned laptop dataset and uses a machine learning pipeline with preprocessing and regression.

**Steps involved:**

* Data Cleaning
* Feature Engineering
* Encoding Categorical Features
* Model Training
* Model Pipeline Creation
* Deployment using Streamlit

## 📊 Features Used for Prediction

* Company
* Type of Laptop
* RAM
* Weight
* Touchscreen
* IPS Display
* Screen Resolution
* Screen Size
* CPU Brand
* HDD
* SSD
* GPU Brand
* Operating System

The model also calculates **PPI (Pixels Per Inch)** from resolution and screen size.

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Streamlit
* Pickle

## 📁 Project Structure

```
Laptop_price_predictor
│
├── app.py
├── model_pipeline.pkl
├── clean_data.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/AbdulSamad502/laptop-price-predictor.git
cd laptop-price-predictor
```

Create virtual environment:

```
python -m venv venv
```

Activate environment (Windows):

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

## 📈 Model Pipeline

The trained pipeline includes:

* ColumnTransformer
* OneHotEncoder for categorical features
* Regression model (XGBoost)

## 🎯 Goal of the Project

The purpose of this project is to demonstrate how machine learning models can be deployed as interactive web applications using Streamlit.

## 📌 Future Improvements

* Add model comparison (RandomForest, XGBoost, Linear Regression)
* Add feature importance visualization
* Improve UI design
* Deploy publicly for live predictions

## 👨‍💻 Author

**Abdul Samad**
GitHub: https://github.com/AbdulSamad502

## ⭐ If you like this project

Give it a star on GitHub!
