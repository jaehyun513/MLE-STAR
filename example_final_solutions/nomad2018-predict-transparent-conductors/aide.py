import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import os

# Load the training and test data
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

# Prepare the data
numerical_features = [
    "spacegroup",
    "number_of_total_atoms",
    "percent_atom_al",
    "percent_atom_ga",
    "percent_atom_in",
    "lattice_vector_1_ang",
    "lattice_vector_2_ang",
    "lattice_vector_3_ang",
    "lattice_angle_alpha_degree",
    "lattice_angle_beta_degree",
    "lattice_angle_gamma_degree",
]

X = train_df[numerical_features]
y_formation_energy = train_df["formation_energy_ev_natom"]
y_bandgap_energy = train_df["bandgap_energy_ev"]

X_test = test_df[numerical_features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X = pd.DataFrame(X_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Define the parameter grid for GridSearchCV
param_grid = {
    "kernel": ["rbf", "linear", "poly", "sigmoid"],
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "gamma": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
}


# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmsle_formation_list = []
rmsle_bandgap_list = []

# Lists to store out-of-fold predictions
oof_formation = np.zeros(len(train_df))
oof_bandgap = np.zeros(len(train_df))


for fold, (train_index, val_index) in enumerate(kf.split(X, y_formation_energy)):
    print(f"Fold {fold + 1}")

    # Split the data into training and validation sets
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_formation_train, y_formation_val = (
        y_formation_energy.iloc[train_index],
        y_formation_energy.iloc[val_index],
    )
    y_bandgap_train, y_bandgap_val = (
        y_bandgap_energy.iloc[train_index],
        y_bandgap_energy.iloc[val_index],
    )

    # Define the models
    krr_formation = KernelRidge()
    krr_bandgap = KernelRidge()

    # Set up GridSearchCV for formation energy
    grid_search_formation = GridSearchCV(
        krr_formation, param_grid, scoring="neg_mean_squared_log_error", cv=5, n_jobs=-1
    )

    # Train the formation energy model using GridSearchCV
    grid_search_formation.fit(X_train, y_formation_train)
    best_krr_formation = grid_search_formation.best_estimator_

    y_formation_pred = best_krr_formation.predict(X_val)
    y_formation_pred[y_formation_pred < 0] = 0  # Ensure non-negative predictions

    # Calculate RMSLE for formation energy
    rmsle_formation = np.sqrt(mean_squared_log_error(y_formation_val, y_formation_pred))
    rmsle_formation_list.append(rmsle_formation)
    oof_formation[val_index] = y_formation_pred

    # Set up GridSearchCV for bandgap energy
    grid_search_bandgap = GridSearchCV(
        krr_bandgap, param_grid, scoring="neg_mean_squared_log_error", cv=5, n_jobs=-1
    )

    # Train the bandgap energy model
    grid_search_bandgap.fit(X_train, y_bandgap_train)
    best_krr_bandgap = grid_search_bandgap.best_estimator_

    y_bandgap_pred = best_krr_bandgap.predict(X_val)
    y_bandgap_pred[y_bandgap_pred < 0] = 0  # Ensure non-negative predictions

    # Calculate RMSLE for bandgap energy
    rmsle_bandgap = np.sqrt(mean_squared_log_error(y_bandgap_val, y_bandgap_pred))
    rmsle_bandgap_list.append(rmsle_bandgap)
    oof_bandgap[val_index] = y_bandgap_pred

# Calculate the mean RMSLE across all folds
mean_rmsle_formation = np.mean(rmsle_formation_list)
mean_rmsle_bandgap = np.mean(rmsle_bandgap_list)
overall_mean_rmsle = np.mean([mean_rmsle_formation, mean_rmsle_bandgap])

print(f"Mean RMSLE for Formation Energy: {mean_rmsle_formation}")
print(f"Mean RMSLE for Bandgap Energy: {mean_rmsle_bandgap}")
print(f"Overall Mean RMSLE: {overall_mean_rmsle}")

# Train the models on the entire training dataset using the best hyperparameters
best_krr_formation.fit(X, y_formation_energy)
best_krr_bandgap.fit(X, y_bandgap_energy)

# Make predictions on the test set
formation_energy_predictions = best_krr_formation.predict(X_test)
bandgap_energy_predictions = best_krr_bandgap.predict(X_test)

# Ensure non-negative predictions
formation_energy_predictions[formation_energy_predictions < 0] = 0
bandgap_energy_predictions[bandgap_energy_predictions < 0] = 0

# Create the submission DataFrame
submission_df = pd.DataFrame(
    {
        "id": test_df["id"],
        "formation_energy_ev_natom": formation_energy_predictions,
        "bandgap_energy_ev": bandgap_energy_predictions,
    }
)

# Save the submission file
os.makedirs("submission", exist_ok=True)
submission_df.to_csv("submission/submission.csv", index=False)
