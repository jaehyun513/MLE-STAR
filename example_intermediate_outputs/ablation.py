
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Define features (X) and target (y)
X = train_df.drop('Transported', axis=1)
y = train_df['Transported'].astype(int) # Convert boolean to int

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Baseline Model ---
# Preprocessing
categorical_features = X.select_dtypes(include=['object']).columns.drop('Name')
numerical_features = X.select_dtypes(include=['number']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

lgbm_classifier = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgbm_classifier.fit(X_train_processed, y_train)

y_pred = lgbm_classifier.predict(X_val_processed)
accuracy = accuracy_score(y_val, y_pred)
baseline_accuracy = accuracy
print(f'Baseline Validation Performance: {baseline_accuracy}')

# --- Ablation 1: No StandardScaler ---
numeric_transformer_no_scaling = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))])

preprocessor_no_scaling = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_no_scaling, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

X_train_processed_no_scaling = preprocessor_no_scaling.fit_transform(X_train)
X_val_processed_no_scaling = preprocessor_no_scaling.transform(X_val)

lgbm_classifier.fit(X_train_processed_no_scaling, y_train)
y_pred_no_scaling = lgbm_classifier.predict(X_val_processed_no_scaling)
accuracy_no_scaling = accuracy_score(y_val, y_pred_no_scaling)
print(f'Ablation 1 (No StandardScaler) Validation Performance: {accuracy_no_scaling}')

# --- Ablation 2: No OneHotEncoder ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

preprocessor_no_onehot = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features)],
    remainder='passthrough')

X_train_subset = X_train[numerical_features]
X_val_subset = X_val[numerical_features]

X_train_processed_no_onehot = preprocessor_no_onehot.fit_transform(X_train_subset)
X_val_processed_no_onehot = preprocessor_no_onehot.transform(X_val_subset)

lgbm_classifier.fit(X_train_processed_no_onehot, y_train)
y_pred_no_onehot = lgbm_classifier.predict(X_val_processed_no_onehot)
accuracy_no_onehot = accuracy_score(y_val, y_pred_no_onehot)
print(f'Ablation 2 (No OneHotEncoder) Validation Performance: {accuracy_no_onehot}')

# --- Ablation 3: No Imputation ---
numeric_transformer_no_imputation = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer_no_imputation = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor_no_imputation = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_no_imputation, numerical_features),
        ('cat', categorical_transformer_no_imputation, categorical_features)])

# Impute numerical features with the mean and categorical features with the mode separately
X_train_num_imputed = X_train[numerical_features].fillna(X_train[numerical_features].mean())
X_train_cat_imputed = X_train[categorical_features].fillna(X_train[categorical_features].mode().iloc[0]) # mode() returns a DataFrame, so use .iloc[0] to get the first row

X_val_num_imputed = X_val[numerical_features].fillna(X_train[numerical_features].mean())
X_val_cat_imputed = X_val[categorical_features].fillna(X_train[categorical_features].mode().iloc[0])

X_train_imputed = pd.concat([X_train_num_imputed, X_train_cat_imputed], axis=1)
X_val_imputed = pd.concat([X_val_num_imputed, X_val_cat_imputed], axis=1)

X_train_processed_no_imputation = preprocessor_no_imputation.fit_transform(X_train_imputed)
X_val_processed_no_imputation = preprocessor_no_imputation.transform(X_val_imputed)

lgbm_classifier.fit(X_train_processed_no_imputation, y_train)
y_pred_no_imputation = lgbm_classifier.predict(X_val_processed_no_imputation)
accuracy_no_imputation = accuracy_score(y_val, y_pred_no_imputation)
print(f'Ablation 3 (No Imputation) Validation Performance: {accuracy_no_imputation}')

if (baseline_accuracy > accuracy_no_scaling and baseline_accuracy > accuracy_no_onehot and baseline_accuracy > accuracy_no_imputation):
    print("Based on the ablation study, StandardScaler, OneHotEncoder and Imputation contribute significantly to the model's performance.")
elif (accuracy_no_scaling > baseline_accuracy):
    print("The StandardScaler hurt model performance.")
elif (accuracy_no_onehot > baseline_accuracy):
    print("The OneHotEncoder hurt model performance.")
elif (accuracy_no_imputation > baseline_accuracy):
    print("The Imputation hurt model performance.")

final_validation_score = accuracy_score(y_val, y_pred)
print(f'Final Validation Performance: {final_validation_score}')
