````markdown
# 🧠 MAMA Tee ML  
**By Samuel Enemona Salifu**

This project demonstrates a simple **Machine Learning workflow** using Python, Pandas, and Scikit-learn.  
We analyze a restaurant dataset (`tips.csv`) to predict **tip amounts** using various regression models — Linear Regression, Decision Tree, and Random Forest — and compare their performance.

---

## 📦 Import Libraries
```python
import pandas as pd
````

---

## 📂 Load the Dataset

```python
df = pd.read_csv('data/tips.csv')
df.head()
```

**Sample Output:**

| total_bill | tip    | gender | smoker | day  | time   | size |
| ---------- | ------ | ------ | ------ | ---- | ------ | ---- |
| 2125.50    | 360.79 | Male   | No     | Thur | Lunch  | 1    |
| 2727.18    | 259.42 | Female | No     | Sun  | Dinner | 5    |
| 1066.02    | 274.68 | Female | Yes    | Thur | Dinner | 4    |
| 3493.45    | 337.90 | Female | No     | Sun  | Dinner | 1    |
| 3470.56    | 567.89 | Male   | Yes    | Sun  | Lunch  | 6    |

---

## 🧾 Data Summary

```python
df.shape
# (744, 7)

df.isnull().sum()
```

| Column     | Missing Values |
| ---------- | -------------- |
| total_bill | 0              |
| tip        | 0              |
| gender     | 0              |
| smoker     | 0              |
| day        | 0              |
| time       | 0              |
| size       | 0              |

---

### 🧠 Data Info

```python
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 744 entries, 0 to 743
Data columns (total 7 columns):
 0   total_bill  float64
 1   tip         float64
 2   gender      object
 3   smoker      object
 4   day         object
 5   time        object
 6   size        int64
```

---

## 📊 Descriptive Statistics

```python
df.describe()
```

| Metric | total_bill | tip     | size  |
| ------ | ---------- | ------- | ----- |
| count  | 744.0      | 744.0   | 744.0 |
| mean   | 2165.00    | 325.95  | 3.18  |
| std    | 954.25     | 148.78  | 1.53  |
| min    | 44.69      | 0.00    | 1.00  |
| 25%    | 1499.02    | 218.00  | 2.00  |
| 50%    | 2102.61    | 320.46  | 3.00  |
| 75%    | 2743.80    | 415.56  | 4.00  |
| max    | 5538.29    | 1090.00 | 6.00  |

---

## ⚙️ Data Preparation

### Split Features and Target

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

y = df['tip']
X = df.drop('tip', axis=1)
```

### One-Hot Encode Categorical Features

```python
X = pd.get_dummies(X, drop_first=True)
```

**Resulting Columns:**

```
['total_bill', 'size', 'gender_Male', 'smoker_Yes', 'day_Mon', 
 'day_Sat', 'day_Sun', 'day_Thur', 'day_Tues', 'day_Wed', 'time_Lunch']
```

### Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

---

## 🧮 Model Training and Evaluation

### 1️⃣ Linear Regression

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

**MAE:**

```python
from sklearn.metrics import mean_absolute_error
print("MAE", mean_absolute_error(y_test, y_pred))
# MAE 118.89702992837498
```

---

### 2️⃣ Decision Tree

```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("MAE", mean_absolute_error(y_test, y_pred_dt))
# MAE 149.52464285714288
```

---

### 3️⃣ Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("MAE", mean_absolute_error(y_test, y_pred_rf))
# MAE 120.50495000000001
```

---

## 📈 Model Comparison

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "MAE": [
        mean_absolute_error(y_test, y_pred),
        mean_absolute_error(y_test, y_pred_dt),
        mean_absolute_error(y_test, y_pred_rf)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred)),
        np.sqrt(mean_squared_error(y_test, y_pred_dt)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf))
    ],
    "R2 Score": [
        r2_score(y_test, y_pred),
        r2_score(y_test, y_pred_dt),
        r2_score(y_test, y_pred_rf)
    ]
})

print("\n===== Model Comparison Table =====")
print(results)
```

**Output:**

| Model             | MAE        | RMSE       | R2 Score  |
| ----------------- | ---------- | ---------- | --------- |
| Linear Regression | 118.897030 | 151.220741 | 0.015844  |
| Decision Tree     | 149.524643 | 199.897517 | -0.719713 |
| Random Forest     | 120.504950 | 154.852847 | -0.031999 |

---

## 🧭 Insights

* **Linear Regression** performed best among the tested models, achieving the **lowest MAE (≈119)**.
* **Decision Tree** tended to overfit, resulting in a high RMSE and negative R² score.
* **Random Forest** provided more stable results but was slightly less accurate than Linear Regression for this dataset.

---

## 🚀 Next Steps

* Feature scaling and hyperparameter tuning
* Model interpretability (using SHAP or feature importance)
* Deploy model with **Streamlit** or **Flask**

---

## 👨🏽‍💻 Author

**Samuel Enemona Salifu**
📧 [ssalifu292@gmail.com](mailto:ssalifu292@gmail.com)
🌐 [GitHub](https://github.com/samvokes88) | [LinkedIn]([(https://www.linkedin.com/in/samuel-enemona-salifu/)].
