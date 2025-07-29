import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from features import CombinedAttributesAdder
from sklearn.preprocessing import StandardScaler


# Create categories for income
def create_income_categories(df):
    df_copy = df.copy()
    df_copy["income_cat"] = pd.cut(
        df_copy["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )
    return df_copy


# Train test split | Plus stratified train test
def stratified_train_test_split(df, target_column='median_house_value', test_size=0.2, random_state=42):
    df_with_cats = create_income_categories(df)

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_index, test_index in split.split(df_with_cats, df_with_cats["income_cat"]):
        strat_train_set = df_with_cats.loc[train_index]
        strat_test_set = df_with_cats.loc[test_index]

    # Remove auxiliary column
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Separate features and target
    X_train = strat_train_set.drop(target_column, axis=1)
    y_train = strat_train_set[target_column].copy()

    X_test = strat_test_set.drop(target_column, axis=1)
    y_test = strat_test_set[target_column].copy()

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# Include mediam in the missing values
def create_preprocessing_pipeline():
    numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                                 ('attribs_adder', CombinedAttributesAdder()),
                                 ('std_scaler', StandardScaler()), ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop=None, sparse_output=False))
    ])

    preprocessing_pipeline = ColumnTransformer([
        ('num', numeric_pipeline, ['longitude', 'latitude', 'housing_median_age',
                                   'total_rooms', 'total_bedrooms', 'population',
                                   'households', 'median_income']),
        ('cat', categorical_pipeline, ['ocean_proximity'])
    ])

    return preprocessing_pipeline


# Option to process only the train data
def preprocess_data(X_train, X_test=None, fit_preprocessor=True):
    preprocessor = create_preprocessing_pipeline()

    if fit_preprocessor:
        X_train_processed = preprocessor.fit_transform(X_train)
        print("Preprocessor fitted and applied to training data")
    else:
        X_train_processed = preprocessor.transform(X_train)
        print("Preprocessor applied to training data")

    if X_test is not None:
        X_test_processed = preprocessor.transform(X_test)
        print("Preprocessor applied to test data")
        return X_train_processed, X_test_processed, preprocessor

    return X_train_processed, preprocessor


# Names of the process data
def get_feature_names(preprocessor):
    # Numerical data
    num_pipeline = preprocessor.named_transformers_['num']
    adder = num_pipeline.named_steps['attribs_adder']
    numeric_feature_names = adder.get_feature_names_out()

    # categorical data
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out(['ocean_proximity'])

    return list(numeric_feature_names) + list(cat_feature_names)


def full_preprocessing_pipeline(df, target_column='median_house_value', test_size=0.2, random_state=42):
    print("=== STARTING COMPLETE PREPROCESSING ===")

    print("\n1. Performing stratified split...")
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df, target_column, test_size, random_state
    )

    print("\n2. Applying preprocessing...")
    X_train_processed, X_test_processed, preprocessor = preprocess_data(X_train, X_test)

    # Get feature names
    feature_names = get_feature_names(preprocessor)

    print(f"\n3. Final shape:")
    print(f"   X_train: {X_train_processed.shape}")
    print(f"   X_test: {X_test_processed.shape}")
    print(f"   Features: {len(feature_names)}")

    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'X_train_raw': X_train,
        'X_test_raw': X_test
    }


# Function for use in other modules
def preprocess_housing_data(df):
    return full_preprocessing_pipeline(df)


# For direct execution
if __name__ == "__main__":
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
