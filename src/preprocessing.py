"""
Preprocessing and Feature Engineering Module for EV Charging Demand Prediction.

This module provides data cleaning, leakage prevention audits, domain-specific
feature transformations, and Scikit-learn Pipeline components for estimating
EV charging session energy requirements (in kWh).

Author: Academic Machine Learning Mentorship & Research Team
Project: EV Charging Demand Prediction Using Ensemble Machine Learning
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# =====================================================================
# DATA LEAKAGE AUDIT & FEATURE CLASSIFICATION
# =====================================================================
"""
DATA LEAKAGE PREVENTION RATIONALE:
----------------------------------
In real-world deployment, when an EV plugs into a charging station, the operator
must estimate the total expected energy consumption BEFORE the session concludes.

The raw dataset contains several post-session measurements that constitute
TARGET LEAKAGE if included during training:

1. 'Energy Consumed (kWh)' -> The primary TARGET variable (y).
2. 'Charging End Time'     -> Only known when the vehicle disconnects.
3. 'State of Charge (End %)' -> Directly implies energy delivered:
                              (SOC_end - SOC_start) * Capacity / 100 ~ Target.
4. 'Charging Duration (hours)' -> Duration is determined by the total energy delivered
                              divided by the charging rate.
5. 'Charging Rate (kW)'   -> Actual delivered rate is monitored during the active session.
6. 'Charging Cost (USD)'  -> Computed directly as a function of Energy Consumed * Tariff.
7. 'User ID' & 'Charging Station ID' -> High-cardinality nominal identifiers that lack
                              generalizability and induce artificial memorization.

EXCLUDED COLUMNS:
['User ID', 'Charging Station ID', 'Charging End Time', 'State of Charge (End %)',
 'Charging Duration (hours)', 'Charging Rate (kW)', 'Charging Cost (USD)']

INCLUDED (PRE-SESSION / ARRIVAL) FEATURES:
- Vehicle & Battery Attributes: Battery Capacity (kWh), Vehicle Model, Vehicle Age (years)
- Arrival State: State of Charge (Start %), Distance Driven (since last charge) (km)
- Operational / Environmental Context: Temperature (°C), Charger Type, User Type,
  Charging Station Location, Time of Day, Day of Week
"""

EXCLUDED_LEAKAGE_COLUMNS = [
    'User ID',
    'Charging Station ID',
    'Charging End Time',
    'State of Charge (End %)',
    'Charging Duration (hours)',
    'Charging Rate (kW)',
    'Charging Cost (USD)'
]

RAW_RENAME_MAP = {
    'Battery Capacity (kWh)': 'battery_capacity',
    'State of Charge (Start %)': 'soc_start',
    'Distance Driven (since last charge) (km)': 'distance_driven',
    'Vehicle Age (years)': 'vehicle_age',
    'Charger Type': 'charger_type',
    'User Type': 'user_type',
    'Time of Day': 'time_of_day',
    'Day of Week': 'day_of_week',
    'Vehicle Model': 'vehicle_model',
    'Charging Station Location': 'location',
    'Energy Consumed (kWh)': 'energy_consumed'
}

NUMERIC_FEATURES = [
    'battery_capacity',
    'soc_start',
    'distance_driven',
    'temperature',
    'vehicle_age',
    'is_weekend',
    'estimated_energy_to_80pct',
    'temp_deviation_from_21c',
    'distance_per_capacity'
]

CATEGORICAL_FEATURES = [
    'charger_type',
    'user_type',
    'time_of_day',
    'day_of_week',
    'vehicle_model',
    'location'
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-learn transformer that computes domain-relevant features
    at the time of vehicle arrival without inducing target leakage.
    
    Derived Features:
    1. 'is_weekend': Binary indicator (1 if Saturday or Sunday, else 0).
    2. 'estimated_energy_to_80pct': Theoretical kWh required to reach 80% SOC
       (standard fast-charging taper threshold): max(0, (80 - SOC_start) * Capacity / 100).
    3. 'temp_deviation_from_21c': Absolute difference from optimal lithium-ion
       operating temperature (21°C), which correlates with battery thermal management loads.
    4. 'distance_per_capacity': Distance driven per unit capacity (km/kWh), representing
       prior battery consumption intensity.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. Weekend indicator
        if 'day_of_week' in X.columns and 'is_weekend' not in X.columns:
            weekend_days = {'Saturday', 'Sunday'}
            X['is_weekend'] = X['day_of_week'].isin(weekend_days).astype(int)

        # 2. Estimated energy required to reach 80% charge threshold
        if 'battery_capacity' in X.columns and 'soc_start' in X.columns:
            soc_diff_to_80 = np.maximum(0.0, 80.0 - X['soc_start'])
            X['estimated_energy_to_80pct'] = (soc_diff_to_80 / 100.0) * X['battery_capacity']

        # 3. Thermal deviation from optimal ambient temperature (21°C)
        if 'temperature' in X.columns:
            X['temp_deviation_from_21c'] = np.abs(X['temperature'] - 21.0)

        # 4. Driving intensity relative to battery capacity
        if 'distance_driven' in X.columns and 'battery_capacity' in X.columns:
            X['distance_per_capacity'] = X['distance_driven'] / (X['battery_capacity'] + 1e-5)

        return X


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize raw column names to clean, uniform snake_case identifiers.
    Handles potential unicode issues in temperature column.
    """
    df = df.copy()
    rename_dict = {}
    for col in df.columns:
        if 'Temp' in col:
            rename_dict[col] = 'temperature'
        elif col in RAW_RENAME_MAP:
            rename_dict[col] = RAW_RENAME_MAP[col]
        else:
            clean_name = (
                col.strip()
                .lower()
                .replace(' ', '_')
                .replace('(', '')
                .replace(')', '')
                .replace('%', 'pct')
            )
            rename_dict[col] = clean_name
    df = df.rename(columns=rename_dict)
    return df


def load_and_clean_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load raw CSV data, perform leakage filtering, drop unlabelled target rows,
    standardize column names, and separate features (X) from target (y).
    """
    df = pd.read_csv(csv_path, encoding='utf-8')
    df = standardize_dataframe(df)

    # In supervised regression, rows with missing target cannot be trained/validated
    df = df.dropna(subset=['energy_consumed']).copy()

    # Drop leakage columns if present
    cols_to_drop = [
        'user_id', 'charging_station_id', 'charging_end_time',
        'state_of_charge_end_pct', 'charging_duration_hours',
        'charging_rate_kw', 'charging_cost_usd'
    ]
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)

    # Start time feature extraction if present
    if 'charging_start_time' in df.columns:
        # Time of day and Day of week are already explicitly in the dataset,
        # but we drop the raw datetime timestamp string to prevent overfitting.
        df = df.drop(columns=['charging_start_time'])

    target_col = 'energy_consumed'
    y = df[target_col]
    X = df.drop(columns=[target_col])

    return X, y


def build_preprocessor(numeric_cols: List[str] = None, categorical_cols: List[str] = None) -> ColumnTransformer:
    """
    Build a leak-free Scikit-learn ColumnTransformer.
    - Numerical: SimpleImputer (median) followed by StandardScaler.
    - Categorical: OneHotEncoder with handle_unknown='ignore' and drop='first'.
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_FEATURES
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def build_full_pipeline(regressor_model: Any) -> Pipeline:
    """
    Construct a complete end-to-end Pipeline combining:
    1. FeatureEngineer (domain feature engineering on arrival data)
    2. ColumnTransformer (imputation, scaling, one-hot encoding)
    3. Regressor Model (e.g., StackingRegressor, RandomForestRegressor, etc.)
    """
    return Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', build_preprocessor()),
        ('regressor', regressor_model)
    ])
