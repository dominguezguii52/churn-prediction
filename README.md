# Customer Churn Prediction & Machine Learning Model

A machine learning project that uses Python and `scikit-learn` to predict customer churn based on historical account data, pricing structures, and support interactions. This model helps operations and customer success teams identify at-risk clients proactively.

## Features
- **Predictive Modeling:** Trains a Logistic Regression classifier to evaluate customer retention risk.
- **Data Analysis:** Evaluates key variables including tenure months, monthly charges, and support tickets.
- **Correlation Mapping:** Generates a statistical heatmap using `seaborn` and `matplotlib` to visualize feature relationships influencing churn.

## Project Structure
- `main.py`: Core machine learning script for data splitting, model training, and heatmap generation.
- `customer_data.csv`: Structured dataset containing customer behavior features and churn labels.
- `churn_correlation.png`: Exported heatmap showing correlation metrics.

## Technologies Used
- Python
- Pandas
- Scikit-Learn
- Seaborn / Matplotlib
