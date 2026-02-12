# 🏠 California Housing Price Predictor

A simple and interactive **Machine Learning web app** built with **Streamlit** to predict California housing prices based on user input features. The app uses a trained regression model and shows both the predicted house value and the model's accuracy.

---

## 📌 Features

- Interactive **sliders** for inputting features like Median Income, House Age, Average Rooms, Bedrooms, Population, Occupancy, Latitude, and Longitude.
- **Predict button** to generate predictions on demand.
- Displays **Predicted House Value** along with **Model Accuracy (R² Score)**.
- **Clean & colorful UI** with responsive layout and columns.
- Expandable section to show **user-entered features** for verification.

---

## 🛠️ Requirements

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Your custom modules:
  - `src/ingestion.py`
  - `src/validation.py`
  - `src/transformation.py`
  - `src/model.py`

---

