#  /$$$$$$$$ /$$$$$$ /$$$$$$$  /$$$$$$ /$$   /$$  /$$$$$$         /$$$$$$   /$$$$$$  /$$$$$$$$                                            
# | $$_____/|_  $$_/| $$__  $$|_  $$_/| $$  /$$/ /$$__  $$       /$$__  $$ /$$__  $$| $$_____/                                            
# | $$        | $$  | $$  \ $$  | $$  | $$ /$$/ | $$  \ $$      | $$  \__/| $$  \ $$| $$                                                  
# | $$$$$     | $$  | $$  | $$  | $$  | $$$$$/  | $$$$$$$$      |  $$$$$$ | $$$$$$$$| $$$$$                                               
# | $$__/     | $$  | $$  | $$  | $$  | $$  $$  | $$__  $$       \____  $$| $$__  $$| $$__/                                               
# | $$        | $$  | $$  | $$  | $$  | $$\  $$ | $$  | $$       /$$  \ $$| $$  | $$| $$                                                  
# | $$$$$$$$ /$$$$$$| $$$$$$$/ /$$$$$$| $$ \  $$| $$  | $$      |  $$$$$$/| $$  | $$| $$$$$$$$                                            
# |________/|______/|_______/ |______/|__/  \__/|__/  |__/       \______/ |__/  |__/|________/                                            
                                                                                                                                        
                                                                                                                                        
                                                                                                                                        
#  /$$   /$$ /$$      /$$        /$$$$$$   /$$$$$$   /$$$$$$  /$$$$$$$$ /$$$$$$$                                                          
# | $$$ | $$| $$$    /$$$       /$$__  $$ /$$$_  $$ /$$__  $$|_____ $$/| $$____/                                                          
# | $$$$| $$| $$$$  /$$$$      |__/  \ $$| $$$$\ $$|__/  \ $$     /$$/ | $$                                                               
# | $$ $$ $$| $$ $$/$$ $$        /$$$$$$/| $$ $$ $$  /$$$$$$/    /$$/  | $$$$$$$                                                          
# | $$  $$$$| $$  $$$| $$       /$$____/ | $$\ $$$$ /$$____/    /$$/   |_____  $$                                                         
# | $$\  $$$| $$\  $ | $$      | $$      | $$ \ $$$| $$        /$$/     /$$  \ $$                                                         
# | $$ \  $$| $$ \/  | $$      | $$$$$$$$|  $$$$$$/| $$$$$$$$ /$$/     |  $$$$$$/                                                         
# |__/  \__/|__/     |__/      |________/ \______/ |________/|__/       \______/                                                          
                                                                                                                                        
                                                                                                                                        
                                                                                                                                        
#  /$$$$$$$   /$$$$$$   /$$$$$$  /$$   /$$ /$$       /$$$$$$  /$$$$$$         /$$$$$$  /$$$$$$$  /$$     /$$ /$$$$$$$   /$$$$$$   /$$$$$$ 
# | $$__  $$ /$$__  $$ /$$__  $$| $$  /$$/| $$      |_  $$_/ /$$__  $$       /$$__  $$| $$__  $$|  $$   /$$/| $$__  $$ /$$__  $$ /$$__  $$
# | $$  \ $$| $$  \ $$| $$  \__/| $$ /$$/ | $$        | $$  | $$  \__/      | $$  \__/| $$  \ $$ \  $$ /$$/ | $$  \ $$| $$  \ $$| $$  \__/
# | $$  | $$| $$$$$$$$| $$ /$$$$| $$$$$/  | $$        | $$  |  $$$$$$       |  $$$$$$ | $$$$$$$/  \  $$$$/  | $$$$$$$/| $$  | $$|  $$$$$$ 
# | $$  | $$| $$__  $$| $$|_  $$| $$  $$  | $$        | $$   \____  $$       \____  $$| $$____/    \  $$/   | $$__  $$| $$  | $$ \____  $$
# | $$  | $$| $$  | $$| $$  \ $$| $$\  $$ | $$        | $$   /$$  \ $$       /$$  \ $$| $$          | $$    | $$  \ $$| $$  | $$ /$$  \ $$
# | $$$$$$$/| $$  | $$|  $$$$$$/| $$ \  $$| $$$$$$$$ /$$$$$$|  $$$$$$/      |  $$$$$$/| $$          | $$    | $$  | $$|  $$$$$$/|  $$$$$$/
# |_______/ |__/  |__/ \______/ |__/  \__/|________/|______/ \______/        \______/ |__/          |__/    |__/  |__/ \______/  \______/ 
                                                                                                                                        
                                                                                                                                 


# This code is part of a comprehensive project focused on developing a "Digital Twin" for an internal combustion engine using deep learning techniques. The script is structured to perform data loading, cleaning, feature engineering, and model training with hyperparameter optimization using Optuna. Additionally, it integrates Weights & Biases (W&B) for experiment tracking and visualization.
# Key Features:
# 1. Reproducibility: A fixed random seed is set to ensure that results
#   are consistent across runs, which is crucial for scientific research.
# 2. Data Cleaning: The script implements a rigorous data cleaning process, including physical boundary checks and statistical outlier removal using the 3-sigma rule.
# 3. Feature Engineering: Depending on the selected mode, it can perform no additional feature engineering, add polynomial features, or use a genetic programming approach to create new features.
# 4. Hyperparameter Optimization: Optuna is used to find the best architecture for the deep neural network, including the number of layers, units, dropout rates, and learning rate.
# 5. Model Training: The final model is trained on the training set and evaluated on a test set that contains data from a specific RPM (1400 RPM) that was not seen during training, testing the model's generalization capabilities.
# 6. Visualization: The script generates heatmaps to visualize feature correlations, loss curves for training and validation, and scatter plots to compare actual vs. predicted values. All visualizations are saved locally and also logged to W&B for easy access and comparison.
# 7. W&B Integration: The script uses W&B to track hyperparameter optimization with Optuna and to log training metrics and visualizations, providing a powerful interface for analyzing the results of different experiments.
# 8. Modularity: The code is organized into functions for each major step, making it easier to read, maintain, and extend in the future.
# 9. Scalability: The use of DataLoaders and batch processing allows the code to handle larger datasets efficiently, and the integration with W&B makes it easier to manage multiple experiments and compare results.
#10. Final Evaluation: The script evaluates the final model on the test set using metrics like Mean Absolute Error (MAE), R-squared, and MSE, providing insights into the model's performance in real physical units.                                                                                                                                       
# Note: Ensure that the required libraries (e.g., torch, optuna, wandb, gplearn) are installed in your Python environment before running this script. Additionally, adjust file paths and W&B configurations as needed for your specific setup.

# Author: Dagklis Spyros -30/4/2026
# Student ID nm20275
# email spd37@hotmail.com
# Disclaimer: This code is provided for educational purposes as part of an academic project. It is not intended for commercial use or deployment in production environments. The author is not responsible for any misuse or unintended consequences resulting from the use of this code. Always ensure that you have the appropriate permissions and ethical considerations in place when working with data and machine learning models.
# You are encouraged to review and understand the code thoroughly before using it, and to modify it as necessary to fit your specific use case. The code is provided "as is" without any warranties or guarantees of performance, accuracy, or suitability for any particular purpose. Use at your own risk.




import os
import csv
import optuna
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import shap   # SHAP (SHapley Additive exPlanations) για ερμηνευσιμότητα μοντέλου (Part 2, Step 6)
import wandb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures

from gplearn.genetic import SymbolicTransformer
from optuna.integration.wandb import WeightsAndBiasesCallback


# FEATURE ENGINEERING MODE
# ==========================================
# none -> Χωρίς επιπλέον χαρακτηριστικά, μόνο τα αρχικά (raw features)
# poly -> Προσθήκη πολυωνυμικών χαρακτηριστικών (degree=2) ΔΕΝ ΖΗΤΗΘΗΚΕ ΑΛΛΑ ΕΙΝΑΙ ΕΝΔΙΑΦΕΡΟΥΣΑ ΠΡΟΣΘΗΚΗ 
# gplearn -> Χρήση γενετικού αλγορίθμου για δημιουργία νέων χαρακτηριστικών

FEATURE_ENG_MODE = os.environ.get('FEATURE_ENG_MODE', 'none')  # None, 'poly', or 'gplearn' (env-overridable)
N_TRIALS = 20              # Sufficient trials for deep architecture search

# RUN_ABLATION -> Αν True, εκτελείται η ελεγχόμενη μελέτη ablation (σάρωση νευρώνων/επιπέδων)
# για τους δύο βασικούς στόχους (NOx, Fuel_Consumption). Δίνει τον πίνακα/γράφημα
# "κάθε αλλαγή -> test loss" που ζητά το Part 2, Step 5. Θέστε False για γρήγορες δοκιμές.
RUN_ABLATION = True

# ==========================================
# ==========================================
# 1. SETUP & REPRODUCIBILITY
# ==========================================
# Καθορισμός σταθερού seed για την εξασφάλιση αναπαραγωγιμότητας των αποτελεσμάτων.
# Είναι κρίσιμο στην έρευνα ώστε οι αλλαγές στην απόδοση να οφείλονται στο μοντέλο και όχι στην τύχη.
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
# Επιλογή συσκευής: CUDA για επιτάχυνση μέσω GPU ή CPU ως εναλλακτική.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATA LOADING & CLEANING
# ==========================================
def load_and_clean_hippo_data(filepath):
    print(f"\n" + "="*45)
    print(f"ΑΝΑΦΟΡΑ ΚΑΘΑΡΙΣΜΟΥ")
    print("="*45)
    
    # 1. Φόρτωση δεδομένων και αφαίρεση στηλών/γραμμών που είναι εξ ολοκλήρου κενές.
    df = pd.read_csv(filepath, sep='\t', skiprows=[1])
    df = df.dropna(axis=1, how='all')
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    initial_count = len(df)

    # 2. Φυσικά Όρια (Physical Boundaries)
    # Διασφάλιση μηχανολογικής ορθότητας: Μεταβλητές όπως στροφές, ροπή και ρύποι δεν νοούνται ως αρνητικές.
    for col in df.columns:
        if col != 'Time': 
            df = df[df[col] >= 0]
    
    # Η πίεση στην πολλαπλή εισαγωγής (Intake Pressure) πρέπει πάντα να είναι θετική τιμή.
    if 'Intake_Pressure' in df.columns:
        df = df[df['Intake_Pressure'] > 0] 

    # 3. Στατιστικό Φιλτράρισμα (3-Sigma) σε ΟΛΕΣ τις στήλες (Z-score cleaning)
    # Αφαίρεση εξωγενών τιμών (outliers) που πιθανώς οφείλονται σε σφάλματα αισθητήρων.
    # Κρατάμε το 99.7% της κανονικής κατανομής των δεδομένων.
    cols_to_check = [col for col in df.columns if col != 'Time']
    
    for col in cols_to_check:
        mean = df[col].mean()
        std = df[col].std()
        # Κρατάμε μόνο τις τιμές που βρίσκονται μέσα στο εύρος του 99.7% των δεδομένων
        df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    
    final_count = len(df)
    print(f"-> Αρχικές γραμμές: {initial_count}")
    print(f"-> Μετά από Φυσικό & Στατιστικό Φιλτράρισμα (Σε ΟΛΕΣ τις στήλες): {final_count}")
    print(f"-> Διαγράφηκαν συνολικά {initial_count - final_count} προβληματικές γραμμές.\n")
    
    return df

# ==========================================
# 3. PREPARATION & FEATURE ENGINEERING
# ==========================================
def auto_select_features(X_train, y_train, all_features, k=8):
    # Χρήση Mutual Information για την ποσοτικοποίηση της εξάρτησης μεταξύ μεταβλητών.
    # Σε αντίθεση με το correlation, το MI ανιχνεύει και μη-γραμμικές σχέσεις (π.χ. φαινόμενα καύσης).
    scores = mutual_info_regression(X_train, y_train, random_state=42)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_features = [all_features[i] for i in sorted_indices]
    
    best_k = min(k, len(sorted_features))
    print(f"-> Επιλεγμένα Top {best_k} χαρακτηριστικά:")
    for i in range(best_k):
        print(f"   {i+1}. {sorted_features[i]} (Score: {scores[sorted_indices[i]]:.4f})")
    return sorted_features[:best_k]

def prepare_loaders_part_b(df, target_name, batch_size=128):
    targets_list = ['NOx', 'Fuel_Consumption', 'Intake_Pressure', 'lambda']
    original_features = [c for c in df.columns if c not in targets_list and c != 'Time']
    
    # Στρατηγική Leave-One-RPM-Out: Απομονώνουμε μια ολόκληρη περιοχή λειτουργίας (1400 RPM) για το Test.
    # Αυτό ελέγχει την ικανότητα γενίκευσης του "Ψηφιακού Διδύμου" σε άγνωστες συνθήκες.
    df_temp = df.copy()
    df_temp['RPM_Rounded'] = df_temp['Rot._Speed'].round(-1)
    test_rpm_target = 1400.0
    
    test_df = df_temp[df_temp['RPM_Rounded'] == test_rpm_target].copy()
    train_val_df = df_temp[df_temp['RPM_Rounded'] != test_rpm_target].copy()
    
    # Διαχωρισμός σε Training και Validation με ανακάτεμα για σταθερότητα στη σύγκλιση.
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, shuffle=True, random_state=42)
    
    X_train_raw = train_df[original_features]
    X_val_raw = val_df[original_features]
    X_test_raw = test_df[original_features]
    
    print(f"\n[Feature Engineering: {FEATURE_ENG_MODE.upper()}]")
    
    if FEATURE_ENG_MODE == 'none':
        X_train_final = X_train_raw
        X_val_final = X_val_raw
        X_test_final = X_test_raw
        
    elif FEATURE_ENG_MODE == 'poly':
        # Δημιουργία αλληλεπιδράσεων 2ου βαθμού (π.χ. Speed * Torque).
        # Βοηθά το μοντέλο να κατανοήσει συνδυαστικά φαινόμενα.
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        X_train_final = pd.DataFrame(poly.fit_transform(X_train_raw), columns=poly.get_feature_names_out(original_features))
        X_val_final = pd.DataFrame(poly.transform(X_val_raw), columns=poly.get_feature_names_out(original_features))
        X_test_final = pd.DataFrame(poly.transform(X_test_raw), columns=poly.get_feature_names_out(original_features))
        
    elif FEATURE_ENG_MODE == 'gplearn':
        # Συμβολική Παλινδρόμηση: Ανακαλύπτει αυτόματα μαθηματικές σχέσεις που διέπουν τη φυσική του προβλήματος.
        from gplearn.genetic import SymbolicTransformer
        
        gp = SymbolicTransformer(generations=5, population_size=500,
                                 hall_of_fame=20, n_components=5,
                                 function_set=('add', 'sub', 'mul', 'div'),
                                 parsimony_coefficient=0.0005,
                                 max_samples=0.9, verbose=0,
                                 random_state=42)
        
        gp.fit(X_train_raw.values, train_df[target_name].values)
        
        gp_features_train = gp.transform(X_train_raw.values)
        gp_features_val = gp.transform(X_val_raw.values)
        gp_features_test = gp.transform(X_test_raw.values)
        
        new_cols = [f"GP_{i+1}" for i in range(gp_features_train.shape[1])]
        
        # Συνένωση αρχικών δεδομένων με τα νέα χαρακτηριστικά του Γενετικού Αλγορίθμου.
        X_train_final = pd.DataFrame(np.hstack((X_train_raw.values, gp_features_train)), columns=original_features + new_cols)
        X_val_final = pd.DataFrame(np.hstack((X_val_raw.values, gp_features_val)), columns=original_features + new_cols)
        X_test_final = pd.DataFrame(np.hstack((X_test_raw.values, gp_features_test)), columns=original_features + new_cols)
        
    else:
        raise ValueError("Invalid FEATURE_ENG_MODE. Choose 'none', 'poly', or 'gplearn'.")

    selected = auto_select_features(X_train_final, train_df[target_name].values, X_train_final.columns.tolist(), k=10)
    
    # =====================================================================
    # ΔΗΜΙΟΥΡΓΙΑ HEATMAP ΓΙΑ ΟΠΤΙΚΟΠΟΙΗΣΗ ΣΥΣΧΕΤΙΣΕΩΝ
    # =====================================================================
    print(f"--> Δημιουργία Post-Engineering Heatmap για: {target_name}...")
    
    # Οπτικοποίηση της συσχέτισης των νέων χαρακτηριστικών με τον στόχο.
    df_for_heatmap = X_train_final[selected].copy()
    df_for_heatmap[target_name] = train_df[target_name].reset_index(drop=True)
    
    # Σχεδίαση του Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(df_for_heatmap.corr(), annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=0.5, annot_kws={"size": 9})
    plt.title(f"Post-Feature Engineering Correlation Heatmap (Top 10)\nTarget: {target_name} | Mode: {FEATURE_ENG_MODE.upper()}", 
              fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOTS_DIR = os.path.join(BASE_DIR, 'plots_part_b')
    # Το όνομα αρχείου περιλαμβάνει το FEATURE_ENG_MODE ώστε οι εκτελέσεις none/poly/gplearn
    # να ΜΗΝ αντικαθιστούν η μία την άλλη (κρατάμε ξεχωριστό heatmap ανά mode).
    heatmap_path = os.path.join(PLOTS_DIR, f"02_Post_Eng_Heatmap_{target_name}_{FEATURE_ENG_MODE}.png")
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    plt.close()

    # 4. Κανονικοποίηση (Scaling)
    # Μετατροπή όλων των τιμών στο εύρος [0, 1]. 
    # Απαραίτητο για τα Νευρωνικά Δίκτυα ώστε να μην κυριαρχούν οι μεταβλητές με μεγάλες μονάδες μέτρησης
    # (π.χ. RPM 2000 vs Lambda 1.0).
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    
    # Κανονικοποίηση δεδομένων
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train = scaler_x.fit_transform(X_train_final[selected])
    y_train = scaler_y.fit_transform(train_df[[target_name]])
    X_val = scaler_x.transform(X_val_final[selected])
    y_val = scaler_y.transform(val_df[[target_name]])
    X_test = scaler_x.transform(X_test_final[selected])
    y_test = scaler_y.transform(test_df[[target_name]])
    
    # Μετατροπή σε PyTorch Tensors
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    # Επιστρέφουμε επιπλέον τα κανονικοποιημένα numpy arrays (X_train, X_test) και τα ονόματα
    # των επιλεγμένων χαρακτηριστικών (selected). Χρειάζονται για την ανάλυση SHAP (Part 2, Step 6):
    # το X_train λειτουργεί ως background dataset και το X_test ως δεδομένα προς ερμηνεία.
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(val_ds, batch_size=batch_size), \
           DataLoader(test_ds, batch_size=batch_size), \
           len(selected), scaler_y, \
           X_train.astype(np.float32), X_test.astype(np.float32), selected

# ==========================================
# 4. DEEP NEURAL NETWORK & OPTUNA
# ==========================================
def objective(trial, t_loader, v_loader, in_dim):
    # Το Optuna αναζητά αυτόματα τις βέλτιστες υπερπαραμέτρους (Bayesian Optimization).
    # Δοκιμάζει αριθμό επιπέδων, νευρώνων και dropout για την αποφυγή overfitting.
    n_layers = trial.suggest_int('n_layers', 1, 2)
    layers = []
    current_in = in_dim
    for i in range(n_layers):
        n_units = trial.suggest_int(f'n_units_l{i}', 10, 100)
        layers.append(nn.Linear(current_in, n_units))
        layers.append(nn.ReLU()) # ReLU ενεργοποίηση για εισαγωγή μη-γραμμικότητας.
        # Dropout: Τυχαία απενεργοποίηση νευρώνων για να αναγκάσουμε το δίκτυο να μάθει πιο εύρωστα σχήματα.
        layers.append(nn.Dropout(trial.suggest_float(f'dropout_l{i}', 0.0, 0.2)))
        current_in = n_units
    layers.append(nn.Linear(current_in, 1)) # Output layer για παλινδρόμηση.
    
    model = nn.Sequential(*layers).to(device)
    # To Learning Rate ελέγχει το μέγεθος του βήματος κατά τη διόρθωση των βαρών.
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-4, 1e-3, log=True))
    criterion = nn.MSELoss()
    
    # Σύντομη εκπαίδευση για την αξιολόγηση του συνδυασμού παραμέτρων.
    for epoch in range(20):
        model.train()
        for bx, by in t_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(); criterion(model(bx), by).backward(); optimizer.step()
            
    model.eval()
    val_loss = sum(criterion(model(bx.to(device)), by.to(device)).item() for bx, by in v_loader)
    return val_loss / len(v_loader)

def train_final_model(model, t_loader, v_loader, model_name, lr):
    print(f"Training final model: {model_name}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # Το MSELoss (Mean Squared Error) τιμωρεί αυστηρά τα μεγάλα σφάλματα, ιδανικό για μηχανικές προβλέψεις.
    
    # [WANDB] 1. Ξεκινάμε το tracking για αυτό το συγκεκριμένο μοντέλο
    run = wandb.init(
        entity="dagklis-",                   # Το username σου στο WandB
        project="Advanced_Control_Systems",  # Όνομα project
        name=model_name,                     # π.χ. 'Fuel_Consumption_PartB'
        config={
            "learning_rate": lr,
            "architecture": "Deep_MLP",
            "epochs": 100
        },
        reinit=True # Σημαντικό γιατί τρέχεις 2 μοντέλα (Fuel και NOx)
    )

    t_losses, v_losses = [], []
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        for bx, by in t_loader:
            optimizer.zero_grad()
            preds = model(bx.to(device))
            loss = criterion(preds, by.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        t_losses.append(epoch_loss / len(t_loader))
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for bx, by in v_loader:
                preds = model(bx.to(device))
                loss = criterion(preds, by.to(device))
                v_loss += loss.item()
        v_losses.append(v_loss / len(v_loader))
        
        # [WANDB] 2. Στέλνουμε το Loss της Εποχής στο Dashboard
        wandb.log({
            "epoch": epoch,
            "train_loss": t_losses[-1],
            "val_loss": v_losses[-1]
        })

    torch.save(model.state_dict(), f'best_model_{model_name}.pth')


    return t_losses, v_losses

# ==========================================
# 4b. SHAP EXPLAINABILITY (Part 2, Step 6)
# ==========================================
def run_shap_analysis(model, X_train, X_test, feature_names, exp_name, plots_dir,
                      n_background=100, n_test=100):
    """Υπολογισμός και οπτικοποίηση SHAP values για το εκπαιδευμένο μοντέλο.

    Η SHAP (SHapley Additive exPlanations) αποδίδει σε κάθε χαρακτηριστικό τη συνεισφορά του
    στην πρόβλεψη, επιτρέποντας τον έλεγχο αν το "Ψηφιακό Δίδυμο" μαθαίνει σωστή φυσική.
    Χρησιμοποιούμε τον KernelExplainer (model-agnostic) όπως στο template της εκφώνησης.
    """
    print(f"\n[SHAP] Υπολογισμός SHAP values για: {exp_name} ...")

    # Wrapper ώστε ο SHAP (που δουλεύει με numpy) να καλεί το PyTorch μοντέλο.
    def predict(x):
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32).to(device)
            output = model(tensor).cpu().numpy()
        return output.flatten()

    # Background dataset: μικρό δείγμα του train (ο KernelExplainer είναι αργός -> περιορίζουμε).
    n_bg = min(n_background, X_train.shape[0])
    background = shap.sample(X_train, n_bg, random_state=42)

    # Περιορίζουμε και τα test δείγματα για λόγους ταχύτητας.
    n_te = min(n_test, X_test.shape[0])
    X_test_sample = X_test[:n_te]

    explainer = shap.KernelExplainer(predict, background)
    shap_values = explainer.shap_values(X_test_sample)

    # --- Plot 1: Bar plot (mean |SHAP| ανά χαρακτηριστικό) -> κατάταξη σημαντικότητας ---
    plt.figure()
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance (mean |SHAP|)\n{exp_name}")
    plt.xlabel("mean(|SHAP value|)  [επίδραση στην πρόβλεψη]")
    plt.tight_layout()
    bar_path = os.path.join(plots_dir, f"05_SHAP_bar_{exp_name}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Plot 2: Beeswarm plot (τιμή χαρακτηριστικού vs επίδραση SHAP) -> κατεύθυνση επίδρασης ---
    plt.figure()
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    plt.title(f"SHAP Beeswarm Plot\n{exp_name}")
    plt.tight_layout()
    bee_path = os.path.join(plots_dir, f"06_SHAP_beeswarm_{exp_name}.png")
    plt.savefig(bee_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Εκτύπωση κατάταξης (για τη συζήτηση φυσικής συνέπειας στην παρουσίαση).
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    print(f"[SHAP] Κατάταξη σημαντικότητας χαρακτηριστικών ({exp_name}):")
    for rank, idx in enumerate(order, 1):
        print(f"   {rank}. {feature_names[idx]:<25} mean|SHAP| = {mean_abs[idx]:.4f}")

    return bar_path, bee_path

# ==========================================
# 4c. ARCHITECTURE ABLATION STUDY (Part 2, Step 5)
# ==========================================
def _train_eval_simple(t_loader, v_loader, test_loader, in_dim, scaler_y,
                       layer_units, lr=5e-4, epochs=40):
    """Βοηθητική: χτίζει/εκπαιδεύει ένα MLP με δοθείσα λίστα νευρώνων ανά layer και
    επιστρέφει το test MSE (σε κανονικοποιημένη κλίμακα) για σύγκριση μεταξύ αρχιτεκτονικών."""
    layers, curr = [], in_dim
    for u in layer_units:
        layers.append(nn.Linear(curr, u))
        layers.append(nn.ReLU())
        curr = u
    layers.append(nn.Linear(curr, 1))
    model = nn.Sequential(*layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for bx, by in t_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx.to(device)), by.to(device))
            loss.backward(); optimizer.step()

    # Test MSE (κανονικοποιημένο) -> ενιαία μετρική "total loss" για τον πίνακα ablation.
    model.eval()
    test_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            test_loss += criterion(model(bx.to(device)), by.to(device)).item()
            n_batches += 1
    return test_loss / max(n_batches, 1)

def run_architecture_ablation(df, target, plots_dir, base_dir):
    """Ελεγχόμενη μελέτη: μεταβάλλουμε ΜΙΑ αλλαγή τη φορά (πλήθος νευρώνων, πλήθος επιπέδων)
    και καταγράφουμε το test loss. Παράγει γράφημα τάσης + πίνακα, όπως ζητά το Step 5
    ('περισσότεροι νευρώνες -> χαμηλότερο σφάλμα', 'περισσότερα επίπεδα -> επίδραση')."""
    print(f"\n" + "="*55 + f"\n[ABLATION] Σάρωση αρχιτεκτονικής για: {target}\n" + "="*55)
    t_loader, v_loader, test_loader, in_dim, scaler_y, _, _, _ = prepare_loaders_part_b(df, target)

    # (A) Επίδραση πλήθους νευρώνων σε 1 hidden layer.
    neuron_grid = [5, 10, 20, 50, 100]
    neuron_losses = []
    for u in neuron_grid:
        loss = _train_eval_simple(t_loader, v_loader, test_loader, in_dim, scaler_y, [u])
        neuron_losses.append(loss)
        print(f"   1 layer x {u:>3} neurons -> test MSE = {loss:.6f}")

    # (B) Επίδραση πλήθους επιπέδων (σταθερό πλάτος 50 νευρώνες).
    depth_configs = {'1 layer (50)': [50], '2 layers (50->50)': [50, 50],
                     '3 layers (50->50->50)': [50, 50, 50]}
    depth_losses = {}
    for name, units in depth_configs.items():
        loss = _train_eval_simple(t_loader, v_loader, test_loader, in_dim, scaler_y, units)
        depth_losses[name] = loss
        print(f"   {name:<22} -> test MSE = {loss:.6f}")

    # --- Γράφημα τάσης (δύο υποδιαγράμματα: νευρώνες & βάθος) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(neuron_grid, neuron_losses, 'o-', color='teal', label='Test MSE')
    ax1.set_xlabel('Neurons (1 hidden layer)'); ax1.set_ylabel('Test MSE (normalized)')
    ax1.set_title(f'Effect of Width — {target}'); ax1.grid(True); ax1.legend()

    ax2.bar(range(len(depth_losses)), list(depth_losses.values()), color='indianred')
    ax2.set_xticks(range(len(depth_losses))); ax2.set_xticklabels(list(depth_losses.keys()), rotation=20, ha='right')
    ax2.set_ylabel('Test MSE (normalized)')
    ax2.set_title(f'Effect of Depth — {target}'); ax2.grid(True, axis='y')

    fig.suptitle(f'Architecture Ablation Study — {target}', fontweight='bold')
    plt.tight_layout()
    ablation_path = os.path.join(plots_dir, f"07_Ablation_{target}.png")
    plt.savefig(ablation_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Καταγραφή των αποτελεσμάτων ablation σε CSV (πίνακας 'αλλαγή -> test loss') ---
    ablation_csv = os.path.join(base_dir, 'ablation_results.csv')
    write_header = not os.path.exists(ablation_csv)
    with open(ablation_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['target', 'feature_mode', 'change', 'config', 'test_MSE_norm'])
        for u, loss in zip(neuron_grid, neuron_losses):
            w.writerow([target, FEATURE_ENG_MODE, 'width', f'1L x {u}', f'{loss:.6f}'])
        for name, loss in depth_losses.items():
            w.writerow([target, FEATURE_ENG_MODE, 'depth', name, f'{loss:.6f}'])

    print(f"[ABLATION] Γράφημα: {ablation_path}")
    return ablation_path

def log_final_metrics_csv(base_dir, target, best, mae, r2, rmse):
    """Προσθέτει μία γραμμή ανά τελικό μοντέλο σε συγκεντρωτικό CSV. Εκτελώντας τον κώδικα
    στα 3 modes (none/poly/gplearn) χτίζεται αυτόματα ο συγκριτικός πίνακας Feature Engineering."""
    results_csv = os.path.join(base_dir, 'final_results.csv')
    write_header = not os.path.exists(results_csv)
    with open(results_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['target', 'feature_mode', 'n_layers', 'units', 'lr',
                        'test_MAE', 'test_R2', 'test_RMSE'])
        units = '->'.join(str(best[f'n_units_l{i}']) for i in range(best['n_layers']))
        w.writerow([target, FEATURE_ENG_MODE, best['n_layers'], units, f"{best['lr']:.2e}",
                    f'{mae:.4f}', f'{r2:.4f}', f'{rmse:.4f}'])

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOTS_DIR = os.path.join(BASE_DIR, 'plots_part_b')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    df = load_and_clean_hippo_data('Data2026.csv')
    targets = ['NOx', 'Fuel_Consumption','Intake_Pressure', 'lambda']
    
    for target in targets:
        print(f"\n" + "#"*60 + f"\nSTARTING PART B FOR: {target.upper()}\n" + "#"*60)
        t_loader, v_loader, test_loader, in_dim, scaler_y, \
            X_train_np, X_test_np, feat_names = prepare_loaders_part_b(df, target)

        exp_name = f"{target}_{FEATURE_ENG_MODE.upper()}_PartB"
        db_path = os.path.join(BASE_DIR, f"study_{exp_name}.db")
        db_path_clean = db_path.replace('\\', '/')
        db_file = f"sqlite:///{db_path_clean}"
        
        # --- ΝΕΟ: Ενεργοποίηση του W&B Callback για το Optuna ---
        # Ρυθμίζουμε πού θα σωθούν τα δεδομένα του Optuna στο W&B
        wandb_kwargs = {
            "project": "Advanced_Control_Systems", 
            "entity": "dagklis-",
            "name": f"Sweep_{exp_name}" # π.χ. Sweep_NOx_GPLEARN_PartB
        }
        wandbc = WeightsAndBiasesCallback(metric_name="val_loss", wandb_kwargs=wandb_kwargs)

        # Δημιουργία του study για την εύρεση της βέλτιστης αρχιτεκτονικής.
        study = optuna.create_study(study_name=f"opt_{exp_name}", storage=db_file, load_if_exists=True, direction="minimize")
        
        if len(study.trials) < N_TRIALS:
            study.optimize(lambda t: objective(t, t_loader, v_loader, in_dim), n_trials=N_TRIALS, callbacks=[wandbc])
            
            # Τερματισμός του sweep run.
            wandb.finish()
        best = study.best_params
        print(f"Best Configuration: {best}")
        
        # --- DYNAMIC MODEL RECONSTRUCTION ---
        final_layers = []
        curr_in = in_dim
        for i in range(best['n_layers']):
            final_layers.append(nn.Linear(curr_in, best[f'n_units_l{i}']))
            final_layers.append(nn.ReLU())
            final_layers.append(nn.Dropout(best[f'dropout_l{i}']))
            curr_in = best[f'n_units_l{i}']
        final_layers.append(nn.Linear(curr_in, 1))
        
        model = nn.Sequential(*final_layers).to(device)
        t_log, v_log = train_final_model(model, t_loader, v_loader, exp_name, best['lr'])
        
        # Καμπύλες σύγκλισης: ΚΑΙ Training ΚΑΙ Validation loss vs εποχή (απαίτηση P1.5 / P2.4).
        plt.figure(figsize=(10, 4))
        plt.plot(t_log, color='steelblue', label='Training Loss')
        plt.plot(v_log, color='orange', label='Validation Loss')
        plt.title(f'Part B: {target} Deep Model Convergence')
        plt.xlabel('Epoch'); plt.ylabel('MSE Loss (normalized)')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(PLOTS_DIR, f"03_Loss_{exp_name}.png"), bbox_inches='tight'); plt.close()
        
        model.load_state_dict(torch.load(f'best_model_{exp_name}.pth'))
        model.eval()
        # Τελική αξιολόγηση στο Test Set (τα άγνωστα δεδομένα των 1400 RPM).
        preds, actuals = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                preds.extend(model(bx.to(device)).cpu().numpy())
                actuals.extend(by.cpu().numpy())
        
        preds_real = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1))
        # Αντιστροφή κανονικοποίησης: Επιστρέφουμε στις φυσικές μονάδες (π.χ. ppm ή g/h) για να καταλαβαίνουμε το σφάλμα.
        actuals_real = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1))
        mae = mean_absolute_error(actuals_real, preds_real)
        r2 = r2_score(actuals_real, preds_real)
        real_mse = mean_squared_error(actuals_real, preds_real)
        
        rmse = np.sqrt(real_mse)
        print(f"\n>>> FINAL TEST METRICS FOR {target} (REAL PHYSICAL UNITS):")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R-squared: {r2:.4f}")
        print(f"MSE: {rmse:.2f}")

        # Καταγραφή των τελικών μετρικών σε CSV (συγκριτικός πίνακας Feature Engineering ανά mode).
        log_final_metrics_csv(BASE_DIR, target, best, mae, r2, rmse)

        plt.figure(figsize=(7, 7))
        plt.scatter(actuals_real, preds_real, alpha=0.4, color='teal')
        plt.plot([actuals_real.min(), actuals_real.max()], [actuals_real.min(), actuals_real.max()], 'r--')
        plt.title(f'Part B: {target} Prediction Performance\nMAE: ±{mae:.2f}'); plt.grid(True)
        plt.xlabel('Actual (Physical)'); plt.ylabel('Predicted (Physical)')
        plt.savefig(os.path.join(PLOTS_DIR, f"04_Scatter_{exp_name}.png")); plt.close()
        
        # --- SHAP: ερμηνευσιμότητα του best μοντέλου (Part 2, Step 6) ---
        shap_bar_path, shap_bee_path = run_shap_analysis(
            model, X_train_np, X_test_np, feat_names, exp_name, PLOTS_DIR)

        # --- ΝΕΟ: Μαζική αποστολή γραφημάτων στο W&B ---
        # Το heatmap περιλαμβάνει το mode στο όνομα (συνέπεια με το save στο prepare_loaders_part_b).
        heatmap_path = os.path.join(PLOTS_DIR, f"02_Post_Eng_Heatmap_{target}_{FEATURE_ENG_MODE}.png")

        log_dict = {
            f"Loss_Plot_{exp_name}": wandb.Image(os.path.join(PLOTS_DIR, f"03_Loss_{exp_name}.png")),
            f"Scatter_Plot_{exp_name}": wandb.Image(os.path.join(PLOTS_DIR, f"04_Scatter_{exp_name}.png")),
            f"SHAP_Bar_{exp_name}": wandb.Image(shap_bar_path),
            f"SHAP_Beeswarm_{exp_name}": wandb.Image(shap_bee_path)
        }

        if os.path.exists(heatmap_path):
            log_dict[f"Post-Engineering_Heatmap_{target}"] = wandb.Image(heatmap_path)

        wandb.log(log_dict)
        
        # ΤΩΡΑ κλείνουμε το W&B Run του τελικού μοντέλου με ασφάλεια
        wandb.finish()

    # ==========================================
    # ABLATION STUDY (Part 2, Step 5): επίδραση κάθε αρχιτεκτονικής αλλαγής στο test loss.
    # Εκτελείται για τους δύο επιλεγμένους στόχους του Part 1 (NOx, Fuel_Consumption).
    # ==========================================
    if RUN_ABLATION:
        for target in ['NOx', 'Fuel_Consumption']:
            run_architecture_ablation(df, target, PLOTS_DIR, BASE_DIR)

    print("\n--- Project Completed Successfully ---")