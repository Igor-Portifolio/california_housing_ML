from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV


# Models
def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_decision_tree(X, y, max_depth=None):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X, y)
    return model


def train_random_forest(X, y, n_estimators=100, max_depth=None):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X, y)
    return model


# Cross validation
def evaluate_model_cv(model, X, y, scoring='neg_mean_squared_error', cv=10):
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    rmse_scores = np.sqrt(-scores)
    return {
        'mean_rmse': rmse_scores.mean(),
        'std_rmse': rmse_scores.std(),
    }


def grid_search_model(model, param_grid, X, y, scoring='neg_mean_squared_error', cv=5, n_jobs=-1):
    model = model()
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, return_train_score=True, n_jobs=n_jobs)
    grid_search.fit(X, y)
    return grid_search


def best_model(X, y, cv=10, n_jobs=-1):
    # Param grids
    param_grid_tree = {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    param_grid_linear = {
        'fit_intercept': [True, False]
    }

    param_grid_forest = {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
        'max_features': ['sqrt', 'log2']
    }

    # Grid search
    bestmodel_1 = grid_search_model(DecisionTreeRegressor, param_grid_tree, X, y, cv=cv, n_jobs=n_jobs).best_estimator_
    bestmodel_2 = grid_search_model(LinearRegression, param_grid_linear, X, y, cv=cv, n_jobs=n_jobs).best_estimator_
    bestmodel_3 = grid_search_model(RandomForestRegressor, param_grid_forest, X, y, cv=cv,
                                    n_jobs=n_jobs).best_estimator_

    # Cross-validation
    error1 = evaluate_model_cv(bestmodel_1, X, y, cv=cv)['mean_rmse']
    error2 = evaluate_model_cv(bestmodel_2, X, y, cv=cv)['mean_rmse']
    error3 = evaluate_model_cv(bestmodel_3, X, y, cv=cv)['mean_rmse']

    errors = {
        'decision_tree': error1,
        'linear_regression': error2,
        'random_forest': error3
    }

    models = {
        'decision_tree': bestmodel_1,
        'linear_regression': bestmodel_2,
        'random_forest': bestmodel_3
    }

    best_name = min(errors, key=errors.get)
    print(f"Best model: {best_name} with RMSE = {errors[best_name]:.4f}")
    return models[best_name]


# For direct execution
if __name__ == "__main__":
    from data_io import load_housing_data
    from preprocessing import preprocess_housing_data

    print("Loading data...")
    housing_data = load_housing_data()

    print("Running preprocessing...")
    processed_data = preprocess_housing_data(housing_data)

    X_train = processed_data['X_train']
    y_train = processed_data['y_train']

    linear = train_linear_regression(X_train, y_train)
    results = evaluate_model_cv(linear, X_train, y_train)

    print(results['mean_rmse'])

    param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                  {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}, ]

    best_model = best_model(X_train, y_train)
