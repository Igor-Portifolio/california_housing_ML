from data_io import load_housing_data
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

matplotlib.use('TkAgg')


def load_and_basic_info():
    housing = load_housing_data()

    print("=== Basic Info ===")
    housing.info()
    print("\n=== Categorical values===")
    print(housing['ocean_proximity'].value_counts())

    pd.set_option('display.max_columns', None)
    print("\n=== Descriptive statistics ===")
    print(housing.describe())

    return housing


def plot_histograms(housing):
    housing.hist(bins=50, figsize=(20, 15))
    plt.suptitle("Distribuição das Variáveis")
    plt.show()


def plot_geographical_data(housing):
    print("\n=== Geographical analysis ===")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="population", figsize=(10, 7),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
    plt.title("Geographical analysis")
    plt.legend()
    plt.show()


def analyze_correlations(housing):
    print("\n=== Correlation analysis ===")
    corr = housing.select_dtypes(include=[float, int]).corr()
    target_corr = corr['median_house_value'].sort_values(ascending=False)
    print("Correlations with median_house_value:")
    print(target_corr)

    # Scatter matrix das variáveis mais importantes
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.suptitle("Scatter Matrix")
    plt.show()

    return target_corr


def main_exploration():
    housing = load_and_basic_info()
    plot_histograms(housing)
    plot_geographical_data(housing)
    correlations = analyze_correlations(housing)

    return housing, correlations


if __name__ == "__main__":
    housing, correlations = main_exploration()
