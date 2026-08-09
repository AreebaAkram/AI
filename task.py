import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Streamlit UI
st.title("Student Performance Prediction using Linear Regression")

st.write("""
    This app predicts the salary based on years of experience using Linear Regression.
""")
exp = np.array([[1], [3], [5], [7], [9]])  
sal = np.array([70000, 75000, 80000, 85000, 90000])

model = LinearRegression().fit(exp, sal)


prediction_10_years = model.predict([[10]])


st.subheader("Prediction for 10 Years of Experience")
st.write(f"Predicted Salary for 10 Years of Experience: ${prediction_10_years[0]:,.2f}")

st.subheader("Model Coefficients")
st.write(f"Coefficient (a1): {model.coef_[0]:.2f}")
st.write(f"Intercept (a0): {model.intercept_:.2f}")

years = st.slider("Select Years of Experience", min_value=1, max_value=20, value=5)

user_prediction = model.predict([[years]])
st.subheader(f"Predicted Salary for {years} Years of Experience")
st.write(f"Predicted Salary: ${user_prediction[0]:,.2f}")

st.subheader("Linear Regression Line with Data Points")

plt.figure(figsize=(8, 6))
plt.scatter(exp, sal, color='blue', label="Data points")
plt.plot(exp, model.predict(exp), color='red', label="Regression Line")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Salary vs. Years of Experience')
plt.legend()
st.pyplot(plt)
