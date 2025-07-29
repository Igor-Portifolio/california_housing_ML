# Housing Prices 


## Introduction

This project aims to cover all the key steps of a Machine Learning workflow  

The dataset used was the California Housing dataset, with the goal of training a model that could 
best predict house prices based on various features.


The project was organized in such a way that the data folder contains the raw dataset (housing.csv), 
while the images folder stores the visualizations and plots generated during the exploratory data analysis phase. 
The models folder holds the trained model artifacts, such as the `best_model.pkl` file. All source code is placed
within the `src` directory, which is structured into specific modules: `data_io.py` for data loading, `exploration.py`
for data analysis, `features.py` and `preprocessing.py` for feature engineering and data transformation, `models.py` for 
model training and evaluation, and `main.py` serves as the entry point of the project. Additionally, a README.md file 
is included at the root level to document the project.

The README will be structured as follows: first, it will outline the technologies and tools used,
followed by an explanation of each module within the `src` folder.
---
## Tools and Technologies 

## Tools and Technologies  

- Python 3.11  
- Pandas (data manipulation and analysis)  
- NumPy (numerical computing)  
- Scikit-learn (machine learning models and preprocessing)  
- Matplotlib (data visualization)  
- Joblib (model persistence and serialization)  
- Git & GitHub (version control and project management)  
- Virtual Environments (dependency isolation and management)  

### Keywords  
- Data Analysis  
- Data Transformation in Pipelines  
- Machine Learning (ML)  
- Regression Models  
- Object-Oriented Programming (OOP)  
- Model Evaluation & Validation  
- Feature Engineering  
- End-to-End ML Workflow  

---
## `src` file 

In this project, the following modules were organized:
`data_io.py`, `exploration.py`, `features.py`, `preprocessing.py`
`models.py` and `main.py`.

---
### `data_io.py`

####  Features

- **Load housing dataset** directly from the `data/` directory.  
- **Save processed data** into structured subfolders for reproducibility.  
- **Save trained models** in the `models/` directory.  
- **Load models** for inference or retraining.  
- **List all saved models** stored in the project.  


####  Functions Overview  

- **`load_housing_data(filepath: str = None) -> pd.DataFrame`**  
  Loads the housing dataset as a pandas DataFrame.  
  - Default: `data/housing.csv`  
  - Validates file existence and prints dataset shape.  

- **`save_data(df: pd.DataFrame, filename: str, subfolder: str = "processed")`**  
  Saves a DataFrame into a specified subfolder inside `data/`.  

- **`save_model(model, filename: str, subfolder: str = "trained")`**  
  Saves a trained model (`.pkl` format) into the `models/` directory.  

- **`load_model(filename: str, subfolder: str = "trained")`**  
  Loads a trained model from disk for evaluation or prediction.  

- **`list_saved_models(subfolder: str = "trained") -> list`**  
  Lists all `.pkl` models stored in the `models/` directory.  

####  Example Usage  

```python
from data_io import load_housing_data, save_data, save_model, load_model, list_saved_models
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = load_housing_data()

# Save processed data
save_data(df, "housing_processed.csv")

# Train and save model
model = RandomForestRegressor()
model.fit(df.drop("median_house_value", axis=1), df["median_house_value"])
save_model(model, "random_forest.pkl")

# Load model
loaded_model = load_model("random_forest.pkl")

# List saved models
print(list_saved_models())
```
---
## `exploration.py`

This module is responsible for **exploratory data analysis (EDA)** on the California Housing dataset.  
It provides functions to inspect, visualize, and understand the main characteristics of the data before moving into preprocessing and modeling.  

####  Features

- **Load and inspect dataset**: view schema, categorical distributions, and descriptive statistics.  
- **Visualize histograms** of all features to understand their distribution.  
- **Geographical data visualization**: explore relationships between location, population, and house values.  
- **Correlation analysis**: investigate relationships between features and target variable.  

#### Main Functions

- **`load_and_basic_info()`**  
  Loads the dataset and prints:
  - Data structure (`.info()`)  
  - Categorical values (`ocean_proximity`)  
  - Descriptive statistics (`.describe()`)  

- **`plot_histograms(housing)`**  
  Displays histograms of all features.  
  - The histogram shows that **most median incomes are between 1.5 and 6**, a very important attribute for predicting house prices.  
  - Strata can be created with `pd.cut` to ensure proportional representation in the training set.  

- **`plot_geographical_data(housing)`**  
  Creates a scatter plot of **longitude vs latitude** with population size and median house value as visual cues.  
  - **House prices are strongly related to location** (e.g., close to the ocean) and **population density**.  

- **`analyze_correlations(housing)`**  
  Computes correlations between features and `median_house_value`.  
  - The **most promising attribute to predict house prices is the median income**.  
  - Also displays a **scatter matrix** for selected features.  

- **`main_exploration()`**  
  Runs the entire exploration pipeline:  
  1. Loads and prints basic dataset info  
  2. Plots histograms  
  3. Visualizes geographical distribution  
  4. Analyzes correlations  

#### Example Usage

```python
from exploration import main_exploration

# Run full exploration workflow
housing, correlations = main_exploration()

```
---
## `features.py` 


This module is responsible for **feature engineering** in the California Housing dataset.  
It defines a custom transformer that extends Scikit-learn’s pipeline functionality to add new attributes that improve model performance.  

#### Features

- Implements a **Scikit-learn-compatible transformer** (`CombinedAttributesAdder`)  
- Dynamically adds new engineered features:
  - `rooms_per_household`
  - `population_per_household`
  - `bedrooms_per_room`  
- Ensures safe division using `np.divide` to avoid division-by-zero errors  
- Returns updated feature names via `get_feature_names_out()`  

#### Main Components

##### `CombinedAttributesAdder`
A custom transformer inheriting from `BaseEstimator` and `TransformerMixin`, making it easy to plug into **Scikit-learn pipelines**.  

#### Parameters
- `add_bedrooms_per_room=True` → adds ratio of bedrooms to rooms  
- `add_rooms_per_household=True` → adds ratio of rooms to households  
- `add_population_per_household=True` → adds ratio of population to households  

#### Methods
- **`fit(X, y=None)`**  
  Returns `self`, as no training is needed.  

- **`transform(X)`**  
  Adds new features to the dataset:  
  - `rooms_per_household = total_rooms / households`  
  - `population_per_household = population / households`  
  - `bedrooms_per_room = total_bedrooms / total_rooms` (safe division)  

- **`get_feature_names_out(input_features=None)`**  
  Returns updated feature names list, including engineered features.  

---

## `preprocessing.py`


This module handles the **data preprocessing pipeline** for the California Housing dataset.  
It combines feature engineering, handling of missing values, categorical encoding, scaling, and a **stratified train-test split** to ensure representative data sampling.  



#### Key Features

- **Income Categories Creation**:  
  Uses `pd.cut` to create income categories for stratified sampling.  
  Ensures that the training and test sets are representative of the entire population distribution.  

- **Stratified Train-Test Split**:  
  Splits the dataset proportionally across income categories to reduce sampling bias.  

- **Preprocessing Pipeline**:  
  - **Numeric pipeline**:
    - Handles missing values with `SimpleImputer(strategy="median")`  
    - Adds engineered features via `CombinedAttributesAdder`  
    - Standardizes values using `StandardScaler`  
  - **Categorical pipeline**:
    - Encodes categories with `OneHotEncoder`  

- **Feature Name Retrieval**:  
  Retrieves updated feature names after transformation, including engineered and one-hot-encoded features.  

- **Full Preprocessing Workflow**:  
  Provides a single function to run the entire preprocessing, from splitting to transformation.  



####  Main Components

#####  `create_income_categories(df)`
Creates an `"income_cat"` column to stratify the dataset into relevant income bins:  
- (0, 1.5] → Category 1  
- (1.5, 3.0] → Category 2  
- (3.0, 4.5] → Category 3  
- (4.5, 6.0] → Category 4  
- (6.0, ∞) → Category 5  

#####  `stratified_train_test_split(df)`
Splits data into training and testing sets with stratification based on `"income_cat"`.  

#####  `create_preprocessing_pipeline()`
Builds the **ColumnTransformer** with numerical and categorical pipelines.  

#####  `preprocess_data(X_train, X_test=None)`
Applies preprocessing to training (and optionally test) data.  

####  `get_feature_names(preprocessor)`
Returns the final list of processed feature names.  

####  `full_preprocessing_pipeline(df)`
Runs the **entire preprocessing workflow**, returning processed arrays, raw splits, preprocessor, and feature names.  



####  Example Usage

```python
from data_io import load_housing_data

print("Loading data...")
housing_data = load_housing_data()

print("Running preprocessing...")
processed_data = preprocess_housing_data(housing_data)

print("\n=== SUMMARY ===")
print(f"Training data: {processed_data['X_train'].shape}")
print(f"Test data: {processed_data['X_test'].shape}")
print(f"Available features: {len(processed_data['feature_names'])}")
print("\nFeatures:", processed_data['feature_names'])
```
---

## `models.py`

This module implements and evaluates different **machine learning models** for predicting house prices in the California Housing dataset.  
It includes training routines, cross-validation evaluation, hyperparameter tuning with Grid Search, and automatic best-model selection.  



#### Key Features

- **Model Training Functions**:  
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  

- **Cross-Validation Evaluation**:  
  Calculates model performance using RMSE (Root Mean Squared Error).  

- **Grid Search**:  
  Performs hyperparameter tuning using `GridSearchCV`.  

- **Automatic Best Model Selection**:  
  Compares tuned models (Decision Tree, Linear Regression, Random Forest) and returns the one with the lowest RMSE.  



#### Main Components

#####  `train_linear_regression(X, y)`
Trains a **Linear Regression** model.  

#####  `train_decision_tree(X, y, max_depth=None)`
Trains a **Decision Tree Regressor** with optional `max_depth`.  

#####  `train_random_forest(X, y, n_estimators=100, max_depth=None)`
Trains a **Random Forest Regressor** with configurable number of estimators and depth.  

#####  `evaluate_model_cv(model, X, y, scoring='neg_mean_squared_error', cv=10)`
Performs cross-validation on a model and returns:  
- Mean RMSE  
- Standard deviation of RMSE  

#####  `grid_search_model(model, param_grid, X, y, cv=5, scoring='neg_mean_squared_error')`
Runs a **Grid Search** over hyperparameters to find the best estimator.  

##### `best_model(X, y, cv=10)`
- Tunes all three models using grid search  
- Evaluates them with cross-validation  
- Selects the model with the lowest RMSE  
- Returns the best estimator  


#### Example Usage

```python
from data_io import load_housing_data
from preprocessing import preprocess_housing_data
from models import best_model, evaluate_model_cv, train_random_forest

# Load data
housing_data = load_housing_data()
processed_data = preprocess_housing_data(housing_data)

X_train = processed_data['X_train']
y_train = processed_data['y_train']

# Train a Random Forest
rf = train_random_forest(X_train, y_train, n_estimators=100)
results = evaluate_model_cv(rf, X_train, y_train)
print("Random Forest RMSE:", results['mean_rmse'])

# Automatically find the best model
best = best_model(X_train, y_train)
print("Selected model:", best)
```

---

## `main.py`

This is the **main execution script** of the project.  
It orchestrates the full workflow of the **California Housing Price Prediction Pipeline**, from loading the dataset, preprocessing, training models, evaluating performance, and saving the best model.  



####  Workflow

1. **Load Data**  
   Uses `data_io.load_housing_data()` to fetch the dataset.

2. **Preprocess Data**  
   Calls `preprocess_housing_data()` to apply transformations:  
   - Missing value imputation  
   - Feature engineering  
   - Standardization  
   - One-hot encoding  
   - Stratified train-test split  

3. **Train & Select Best Model**  
   Runs `best_model()` from the `models.py` module, which:  
   - Performs grid search on Decision Tree, Linear Regression, and Random Forest  
   - Selects the model with the lowest RMSE via cross-validation  

4. **Evaluate Final Model**  
   Tests the selected model on the **test dataset** using cross-validation RMSE.

5. **Save Model**  
   Saves the trained model into a `.pkl` file for later use with `save_model()`.  



#### Functions

##### `main()`
- Runs the entire ML pipeline step by step.  
- Prints progress and results to the console.  
- Saves the best model as `"best_model.pkl"`.  



#### Usage

Run the script directly:

```bash
python main.py
```

---

## Images from data Exploration 

### Income categories histogram
![Histograma de categorias de renda](images/Histogram%20of%20income%20categories.png)

### Geographical Distribution of Data
![Dados geográficos](images/Geografical%20Data.png)

### Correlation Between Median Income and House Value
![Renda vs Preço Mediano](images/Median%20income%20versus%20median%20house%20value.png)

### Scatter Matrix
![Matriz de dispersão](images/This%20scatter%20matrix%20plots%20every%20numerical%20attribute%20against%20every%20other.png)

