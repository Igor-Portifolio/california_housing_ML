import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

LONGITUDE_IX = 0
LATITUDE_IX = 1
HOUSING_MEDIAN_AGE_IX = 2
TOTAL_ROOMS_IX = 3
TOTAL_BEDROOMS_IX = 4
POPULATION_IX = 5
HOUSEHOLDS_IX = 6
MEDIAN_INCOME_IX = 7


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True, add_rooms_per_household=True,
                 add_population_per_household=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.add_rooms_per_household = add_rooms_per_household
        self.add_population_per_household = add_population_per_household

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_features = []

        if self.add_rooms_per_household:
            rooms_per_household = X[:, TOTAL_ROOMS_IX] / X[:, HOUSEHOLDS_IX]
            new_features.append(rooms_per_household.reshape(-1, 1))

        if self.add_population_per_household:
            population_per_household = X[:, POPULATION_IX] / X[:, HOUSEHOLDS_IX]
            new_features.append(population_per_household.reshape(-1, 1))

        if self.add_bedrooms_per_room:
            rooms = X[:, TOTAL_ROOMS_IX]
            bedrooms = X[:, TOTAL_BEDROOMS_IX]
            bedrooms_per_room = np.divide(bedrooms, rooms,
                                          out=np.zeros_like(bedrooms),
                                          where=rooms != 0)
            new_features.append(bedrooms_per_room.reshape(-1, 1))

        if new_features:
            return np.c_[X, *new_features]
        else:
            return X

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            input_features = ['longitude', 'latitude', 'housing_median_age',
                              'total_rooms', 'total_bedrooms', 'population',
                              'households', 'median_income']

        output_features = list(input_features)

        if self.add_rooms_per_household:
            output_features.append('rooms_per_household')

        if self.add_population_per_household:
            output_features.append('population_per_household')

        if self.add_bedrooms_per_room:
            output_features.append('bedrooms_per_room')

        return output_features
