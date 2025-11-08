# Mama-Tee-restaurant-Model
## By Samuel Enemona Salifu
Building a regression task to predict the amount of tips
💡 Tip Prediction using Machine Learning

This project demonstrates a regression analysis using the Tips Dataset, predicting restaurant tips based on total bill, gender, smoker status, day, time, and group size.

🧩 Dataset Overview
<details> <summary>🔍 Click to expand</summary>

File: data/tips.csv

total_bill	tip	gender	smoker	day	time	size
2125.50	360.79	Male	No	Thur	Lunch	1
2727.18	259.42	Female	No	Sun	Dinner	5
1066.02	274.68	Female	Yes	Thur	Dinner	4
3493.45	337.90	Female	No	Sun	Dinner	1
3470.56	567.89	Male	Yes	Sun	Lunch	6

Shape: (744, 7)
Missing Values: None
Data Types: 4 categorical, 3 numerical

</details>
⚙️ Data Preprocessing
<details> <summary>⚗️ Click to view preprocessing steps</summary>
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/tips.csv")

# Define target and features
y = df["tip"]
X = df.drop("tip", axis=1)

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

</details>
🧠 Model Training
<details> <summary>🤖 Linear Regression</summary>
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

</details> <details> <summary>🌳 Decision Tree Regressor</summary>
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

</details> <details> <summary>🌲 Random Forest Regressor</summary>
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

</details>
📊 Model Evaluation
<details> <summary>📈 Click to show evaluation metrics</summary>
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

print(results)

</details>
🧾 Results Summary
Model	MAE (↓)	RMSE (↓)	R² Score (↑)
Linear Regression	118.90	151.22	0.016
Decision Tree	149.52	199.90	-0.720
Random Forest	120.50	154.85	-0.032
💬 Key Insights

Linear Regression produced the lowest MAE and best R², performing slightly better than ensemble methods.

Decision Tree showed signs of overfitting, reflected in its negative R² score.

Random Forest offered stable predictions but didn’t outperform the simpler model.

Further feature engineering and hyperparameter tuning could improve performance.

🧰 Requirements
pip install pandas numpy scikit-learn

🧑‍💻 Author

Samuel Enemona
