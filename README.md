# 🧍⚖️ Simple Linear Regression: Height vs. Weight Prediction

This repository contains a simple implementation of the **Linear Regression** model using Python and `scikit-learn`. The project analyzes the linear relationship between a person's **Weight** (independent variable) and their **Height** (dependent variable) using a simple, publicly available dataset.

This work serves as a foundational exercise in linear modeling, which is a crucial step in my current **deep research project**.

## 🎯 Project Overview & Steps

The `implementation.ipynb` notebook follows a standard machine learning workflow:

1.  **Data Loading and Exploration:** Loading the `height-weight.csv` dataset, viewing the first few rows, and visualizing the data using a scatter plot and `seaborn`'s pairplot to confirm the linear relationship.
2.  **Correlation Check:** Calculating the correlation coefficient between Height and Weight (found to be $\approx 0.931$).
3.  **Data Preparation:** Separating the features (X: Weight) and target (y: Height), followed by a Train-Test split (`test_size=0.25`, `random_state=42`).
4.  **Feature Scaling:** Standardizing the training and test features using `StandardScaler` to ensure optimal performance for the model.
5.  **Model Training:** Applying `sklearn.linear_model.LinearRegression` to the scaled training data.
6.  **Prediction and Evaluation:** Making predictions on the test set and evaluating model performance using various metrics.

## 🛠️ Technology and Libraries

* **Language:** Python
* **Notebook:** Jupyter Notebook (`implementation.ipynb`)
* **Core Libraries:**
    * `pandas` (Data manipulation)
    * `numpy` (Numerical operations)
    * `matplotlib` / `seaborn` (Data visualization)
    * `scikit-learn` (Model training, scaling, and evaluation)
    * `statsmodels` (For OLS summary and detailed statistical review)

## 📊 Key Results

The model exhibits a strong linear relationship between the variables.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Coefficient ($\beta_1$ / Slope)** | $17.2982$ | For every unit increase in the **scaled** Weight, Height is predicted to increase by approximately 17.3 units. |
| **Intercept ($\beta_0$)** | $156.47$ | The predicted Height when the **scaled** Weight is 0 (i.e., at the mean weight of the training data). |
| **R-squared ($R^2$ Score)** | $0.736$ | Approximately **73.6%** of the variance in Height can be explained by the Weight feature. |
| **Mean Squared Error (MSE)** | $114.84$ | The average squared difference between the observed Height and the predicted Height. |
| **Root Mean Squared Error (RMSE)** | $10.72$ | The standard deviation of the residuals (prediction errors). |
| **Adjusted $R^2$** | $0.670$ | (The R-squared adjusted for the number of predictors). |

## 🏃 Getting Started

### Prerequisites

You need **Python 3.x** installed. We highly recommend setting up a virtual environment (`venv` or `conda`).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_GITHUB_REPO_URL]
    cd [YOUR_REPO_NAME]
    ```
2.  **Install the dependencies:**
    *(The required packages are listed in `requirements.txt`)*
    ```bash
    pip install -r requirements.txt
    ```
3.  **Acquire Data:**
    Ensure you have the `height-weight.csv` file in the same directory as the notebook, as it is read directly by the code.
4.  **Run the analysis:**
    Launch the Jupyter Notebook environment and open the analysis file:
    ```bash
    jupyter notebook implementation.ipynb
    ```

## 📜 requirements.txt

The packages used in your notebook are `pandas`, `matplotlib`, `numpy`, `seaborn`, `scikit-learn`, and `statsmodels`. You should create a file named `requirements.txt` with the following content: