
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import numpy as np
import os
from catboost import CatBoostRegressor, Pool

# Function to calculate RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Load data
train_file = os.path.join('input', 'train.csv')
test_file = os.path.join('input', 'test.csv')
data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Feature Engineering from XYZ files - LightGBM
import os
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, ConvexHull

def extract_geometry_features(df, data_dir):
    all_features = []
    for idx in df['id']:
        filename = os.path.join(data_dir, str(idx), 'geometry.xyz')
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()

                # Extract lattice vectors (not used in this version but kept for consistency)
                lattice_vectors = []
                for i in range(2, 5):  # Lattice vectors are on lines 2, 3, and 4
                    if i < len(lines):
                        parts = lines[i].split()
                        if len(parts) > 3:
                            lattice_vectors.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        else:
                            lattice_vectors.append([0.0, 0.0, 0.0])  # Pad if data is missing
                    else:
                        lattice_vectors.append([0.0, 0.0, 0.0]) # Pad if line is missing


                # Extract atom coordinates and symbols
                atom_coords = []
                atom_symbols = []
                for i in range(5, len(lines)):  # Atoms start from line 5
                    parts = lines[i].split()
                    if len(parts) > 3:
                        atom_symbols.append(parts[-1])
                        atom_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    
                num_atoms = len(atom_symbols)

                # Calculate Voronoi volumes
                if num_atoms > 3: # Need at least 4 points for Voronoi
                    try:
                        vor = Voronoi(atom_coords)
                        volumes = []
                        for region_index in vor.point_region:
                            if region_index == -1:
                                volumes.append(0.0)  # Volume undefined
                                continue
                            region = vor.regions[region_index]
                            if not all(i >= 0 for i in region):
                                volumes.append(0.0)  # Volume undefined (unbounded region)
                                continue

                            try:
                                vertices = vor.vertices[region]
                                hull = ConvexHull(vertices)
                                volumes.append(hull.volume)
                            except:
                                volumes.append(0.0)  # Handle cases where volume calculation fails

                        voronoi_mean = np.mean(volumes) if volumes else 0.0
                        voronoi_std = np.std(volumes) if volumes else 0.0
                        voronoi_min = np.min(volumes) if volumes else 0.0
                        voronoi_max = np.max(volumes) if volumes else 0.0
                    except:
                        voronoi_mean = 0.0
                        voronoi_std = 0.0
                        voronoi_min = 0.0
                        voronoi_max = 0.0


                else:
                    voronoi_mean = 0.0
                    voronoi_std = 0.0
                    voronoi_min = 0.0
                    voronoi_max = 0.0
                
                # Basic stats of coordinates
                x_coords = [coord[0] for coord in atom_coords]
                y_coords = [coord[1] for coord in atom_coords]
                z_coords = [coord[2] for coord in atom_coords]
                
                x_mean = np.mean(x_coords) if x_coords else 0.0
                y_mean = np.mean(y_coords) if y_coords else 0.0
                z_mean = np.mean(z_coords) if z_coords else 0.0
                
                x_std = np.std(x_coords) if x_coords else 0.0
                y_std = np.std(y_coords) if y_coords else 0.0
                z_std = np.std(z_coords) if z_coords else 0.0
                
                # Combine features
                features = [num_atoms, x_mean, y_mean, z_mean, x_std, y_std, z_std, voronoi_mean, voronoi_std, voronoi_min, voronoi_max]

                all_features.append(features)
        except FileNotFoundError:
            print(f"File not found: {filename}. Filling with default values.")
            features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            all_features.append(features)


    feature_names = ['num_atoms_lgbm', 'x_mean_lgbm', 'y_mean_lgbm', 'z_mean_lgbm', 'x_std_lgbm', 'y_std_lgbm', 'z_std_lgbm', 'voronoi_mean_lgbm', 'voronoi_std_lgbm', 'voronoi_min_lgbm', 'voronoi_max_lgbm']  # Example names
    features_df = pd.DataFrame(all_features, columns=feature_names)
    return features_df


# Apply feature extraction
train_geometry_dir = 'input/train'
test_geometry_dir = 'input/test'

train_geometry_features = extract_geometry_features(data, train_geometry_dir)
test_geometry_features = extract_geometry_features(test_data, test_geometry_dir)

# Add geometry features to the main dataframes, using index to join (assuming the order is preserved)
data = pd.concat([data, train_geometry_features], axis=1)
test_data = pd.concat([test_data, test_geometry_features], axis=1)

X = data.drop(['id','formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
y_formation = data['formation_energy_ev_natom']
y_bandgap = data['bandgap_energy_ev']

# Use all training data for training the final model
X_train = X
y_formation_train = y_formation
y_bandgap_train = y_bandgap

# Function to train LightGBM model
def train_model(X_train, y_train):
    model = lgb.LGBMRegressor(objective='regression',
                           n_estimators=100,
                           learning_rate=0.1,
                           max_depth=5,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           random_state=42,
                           n_jobs=-1,
                           reg_alpha=0.1,
                           reg_lambda=0.1)
    model.fit(X_train, y_train)
    return model

# Train models on full training data
model_formation_lgbm = train_model(X_train, y_formation_train)
model_bandgap_lgbm = train_model(X_train, y_bandgap_train)

# Prepare submission file
test_X = test_data.drop('id', axis=1)

# Predict formation energy
formation_predictions_lgbm = model_formation_lgbm.predict(test_X)
formation_predictions_lgbm[formation_predictions_lgbm < 0] = 0  # Ensure non-negative

# Predict bandgap energy
bandgap_predictions_lgbm = model_bandgap_lgbm.predict(test_X)
bandgap_predictions_lgbm[bandgap_predictions_lgbm < 0] = 0  # Ensure non-negative


#----------------------------------------------------------------------------------------------------------------------
# Catboost Solution
import numpy as np

def extract_atomic_features(geometry_file, include_percent_atoms=True):
    """Extracts atomic features and distance-based features from the geometry file.
    Args:
        geometry_file (str): Path to the geometry file.
        include_percent_atoms (bool): Whether to include the percent atom features.
    Returns:
        dict: A dictionary of atomic features.
    """
    try:
        with open(geometry_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {geometry_file}")
        return {}

    # Skip header lines
    atom_lines = lines[2:]
    
    elements = []
    x_coords = []
    y_coords = []
    z_coords = []
    atomic_numbers = []

    for line in atom_lines:
        parts = line.split()
        if len(parts) == 5 and parts[0] == 'atom':
            elements.append(parts[4])
            x_coords.append(float(parts[1]))
            y_coords.append(float(parts[2]))
            z_coords.append(float(parts[3]))
            atomic_numbers.append(get_atomic_number(parts[4]))  # Convert element to atomic number

    elements_counts = {}
    for element in elements:
        elements_counts[element] = elements_counts.get(element, 0) + 1

    coords = np.array([x_coords, y_coords, z_coords]).T
    
    # Calculate Coulomb matrix
    coulomb_matrix = compute_coulomb_matrix(atomic_numbers, coords)

    # Calculate eigenvalues and sort them
    eigenvalues = np.linalg.eigvalsh(coulomb_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
    
    # Pad eigenvalues with zeros if the number of atoms is less than the expected number
    num_atoms = len(atomic_numbers)
    num_eigenvalues = len(eigenvalues)
    
    # Define number of eigenvalues to use as features
    n_eigenvalues = 30 #Empirically chosen
    
    if num_eigenvalues < n_eigenvalues:
        padding_size = n_eigenvalues - num_eigenvalues
        eigenvalues = np.pad(eigenvalues, (0, padding_size), 'constant')
    
    eigenvalue_features = {f'eigenvalue_{i}': eigenvalues[i] for i in range(n_eigenvalues)}

    # Calculate interatomic distances
    distances = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            distances.append(dist)

    distance_features = {}
    if distances:
        distance_features['min_distance'] = np.min(distances)
        distance_features['max_distance'] = np.max(distances)
        distance_features['mean_distance'] = np.mean(distances)
        distance_features['std_distance'] = np.std(distances)
    else:
        distance_features['min_distance'] = 0
        distance_features['max_distance'] = 0
        distance_features['mean_distance'] = 0
        distance_features['std_distance'] = 0
        
    all_features = {
        'num_atoms_catboost': num_atoms,
        'avg_x_catboost': np.mean(x_coords) if x_coords else 0,
        'avg_y_catboost': np.mean(y_coords) if y_coords else 0,
        'avg_z_catboost': np.mean(z_coords) if y_coords else 0,
        'std_x_catboost': np.std(x_coords) if x_coords else 0,
        'std_y_catboost': np.std(y_coords) if y_coords else 0,
        'std_z_catboost': np.std(z_coords) if y_coords else 0,
        **eigenvalue_features,
        **distance_features
    }
    
    if include_percent_atoms:
        total_atoms = sum(elements_counts.values())
        percent_atoms = {f'percent_{element}': count / total_atoms for element, count in elements_counts.items()}
        all_features.update(elements_counts)
        
    else:
        all_features.update(elements_counts)

    return all_features



def get_atomic_number(element):
    """Maps element symbol to atomic number."""
    atomic_number_map = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9
    }
    return atomic_number_map.get(element, 0)  # Returns 0 if element is not found

def compute_coulomb_matrix(atomic_numbers, coords):
    """Computes the Coulomb matrix."""
    n_atoms = len(atomic_numbers)
    coulomb_matrix = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                coulomb_matrix[i, j] = 0.5 * atomic_numbers[i] ** 2.4
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                coulomb_matrix[i, j] = (atomic_numbers[i] * atomic_numbers[j]) / dist
    return coulomb_matrix


# Load the training data

# Load the test data


# Features
features = ['spacegroup', 'number_of_total_atoms', 'percent_atom_al', 'percent_atom_ga',
            'percent_atom_in', 'lattice_vector_1_ang', 'lattice_vector_2_ang',
            'lattice_vector_3_ang', 'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',
            'lattice_angle_gamma_degree']
target_formation_energy = 'formation_energy_ev_natom'
target_bandgap_energy = 'bandgap_energy_ev'

# Extract atomic features from training geometry files
geometry_features_train = []
for id in data['id']:
    file_path = f'./input/train/{id}/geometry.xyz'
    atomic_features = extract_atomic_features(file_path)
    geometry_features_train.append(atomic_features)

geometry_df_train = pd.DataFrame(geometry_features_train)
data = pd.concat([data, geometry_df_train], axis=1)

# Extract atomic features from testing geometry files
geometry_features_test = []
for id in test_data['id']:
    file_path = f'./input/test/{id}/geometry.xyz'
    atomic_features = extract_atomic_features(file_path)
    geometry_features_test.append(atomic_features)

geometry_df_test = pd.DataFrame(geometry_features_test)
test_data = pd.concat([test_data, geometry_df_test], axis=1)

# Update features list
features.extend(geometry_df_train.columns)

# Prepare data for training
X_train_catboost = data[features]
y_formation_train_catboost = data[target_formation_energy]
y_bandgap_train_catboost = data[target_bandgap_energy]

# Prepare test data
X_test_catboost = test_data[features]

# --- Formation Energy Model ---
train_pool_formation = Pool(X_train_catboost, np.log1p(y_formation_train_catboost))

model_formation_catboost = CatBoostRegressor(objective='RMSE',
                                    eval_metric='MSLE',
                                    loss_function = 'RMSE',
                                    iterations=1000,
                                    learning_rate=0.01,
                                    depth=6,
                                    l2_leaf_reg=3,
                                    random_seed=42,
                                    verbose=0)

model_formation_catboost.fit(train_pool_formation)

# Predict formation energy on test data
formation_pred_test_catboost = np.expm1(model_formation_catboost.predict(X_test_catboost))

# --- Bandgap Energy Model ---
train_pool_bandgap = Pool(X_train_catboost, np.log1p(y_bandgap_train_catboost))

model_bandgap_catboost = CatBoostRegressor(objective='RMSE',
                                  eval_metric='MSLE',
                                  loss_function = 'RMSE',
                                  iterations=1000,
                                  learning_rate=0.01,
                                  depth=6,
                                  l2_leaf_reg=3,
                                  random_seed=42,
                                  verbose=0)

model_bandgap_catboost.fit(train_pool_bandgap)

# Predict bandgap energy on test data
bandgap_pred_test_catboost = np.expm1(model_bandgap_catboost.predict(X_test_catboost))


# Ensemble with target-specific weights
formation_weight_lgbm = 0.4
formation_weight_catboost = 0.6
bandgap_weight_lgbm = 0.4
bandgap_weight_catboost = 0.6

formation_predictions = (formation_weight_lgbm * formation_predictions_lgbm) + (formation_weight_catboost * formation_pred_test_catboost)
bandgap_predictions = (bandgap_weight_lgbm * bandgap_predictions_lgbm) + (bandgap_weight_catboost * bandgap_pred_test_catboost)

# Clipping
formation_predictions[formation_predictions < 0] = 0
bandgap_predictions[bandgap_predictions < 0] = 0

# Optional thresholding
formation_predictions[formation_predictions < 0.01] = 0
bandgap_predictions[bandgap_predictions < 0.01] = 0

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_data['id'],
    'formation_energy_ev_natom': formation_predictions,
    'bandgap_energy_ev': bandgap_predictions
})

# Save submission to CSV in the final directory
output_dir = './final'
os.makedirs(output_dir, exist_ok=True)
submission_file = os.path.join(output_dir, 'submission.csv')
submission.to_csv(submission_file, index=False)

# Validation
X_train, X_val, y_formation_train, y_formation_val, y_bandgap_train, y_bandgap_val = train_test_split(
    data.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1),
    data['formation_energy_ev_natom'],
    data['bandgap_energy_ev'],
    test_size=0.2,
    random_state=42
)

# Use the same features as the training data
X_val_lgbm = X_val[X.columns]
X_val_catboost = data[features].iloc[X_val.index]

formation_predictions_val_lgbm = model_formation_lgbm.predict(X_val_lgbm)
formation_predictions_val_lgbm[formation_predictions_val_lgbm < 0] = 0

bandgap_predictions_val_lgbm = model_bandgap_lgbm.predict(X_val_lgbm)
bandgap_predictions_val_lgbm[bandgap_predictions_val_lgbm < 0] = 0

formation_pred_val_catboost = np.expm1(model_formation_catboost.predict(X_val_catboost))
bandgap_pred_val_catboost = np.expm1(model_bandgap_catboost.predict(X_val_catboost))

formation_predictions_val = (formation_weight_lgbm * formation_predictions_val_lgbm) + (formation_weight_catboost * formation_pred_val_catboost)
bandgap_predictions_val = (bandgap_weight_lgbm * bandgap_predictions_val_lgbm) + (bandgap_weight_catboost * bandgap_pred_val_catboost)

formation_predictions_val[formation_predictions_val < 0] = 0
bandgap_predictions_val[bandgap_predictions_val < 0] = 0
formation_predictions_val[formation_predictions_val < 0.01] = 0
bandgap_predictions_val[bandgap_predictions_val < 0.01] = 0

rmsle_formation = rmsle(y_formation_val, formation_predictions_val)
rmsle_bandgap = rmsle(y_bandgap_val, bandgap_predictions_val)

final_validation_score = (rmsle_formation + rmsle_bandgap) / 2

print(f'Final Validation Performance: {final_validation_score}')
