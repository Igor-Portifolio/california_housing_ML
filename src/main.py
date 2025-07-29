from data_io import load_housing_data, save_model
from preprocessing import preprocess_housing_data
from models import best_model, evaluate_model_cv


def main():
    print("Loading data...")
    housing = load_housing_data()

    print("Running preprocessing...")
    processed_data = preprocess_housing_data(housing)

    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']

    print("Training models and selecting the best one...")
    model = best_model(X_train, y_train)

    print("Training complete. Evaluating on test set...")
    final_rmse = evaluate_model_cv(model, X_test, y_test)
    print(f"Final RMSE on test set: {final_rmse['mean_rmse']}.")

    print("Saving model...")
    save_model(model, "best_model.pkl")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
