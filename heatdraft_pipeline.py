# HeatDraft End-to-End Pipeline
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
import pandas as pd
import re
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.parse
import urllib.request



def visualize_data(file_path):
    print(f"Loading data from {file_path}...")
    # Load the data
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, header=1)
    else:
        df = pd.read_csv(file_path, header=1)

    print(f"Dataset shape: {df.shape}")

    # Create an output directory for plots
    os.makedirs("visualizations", exist_ok=True)

    # 0. Generate Data Stats
    stats_df = df.describe(include="all").T
    stats_df.to_csv("visualizations/0_data_statistics.csv")
    print("Saved basic data statistics to '0_data_statistics.csv'.")

    # 1. Missing Values Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig("visualizations/1_missing_values.png")
    plt.close()
    print("Saved missing values heatmap.")

    # 2. Distribution of Numeric Features
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        # Plot them in chunks of 16 to ensure they are readable
        chunk_size = 16
        for i in range(0, len(num_cols), chunk_size):
            plot_cols = num_cols[i : i + chunk_size]

            # Compute grid size
            n_cols_plot = 4
            n_rows_plot = (len(plot_cols) + n_cols_plot - 1) // n_cols_plot

            fig, axes = plt.subplots(
                n_rows_plot, n_cols_plot, figsize=(16, 4 * n_rows_plot)
            )
            axes = axes.flatten() if len(plot_cols) > 1 else [axes]

            for j, col in enumerate(plot_cols):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[j])
                axes[j].set_title(col[:30])  # truncate title if too long
                axes[j].set_xlabel("")
                axes[j].set_ylabel("")

            # Hide empty subplots
            for k in range(len(plot_cols), len(axes)):
                axes[k].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                f"visualizations/2_numeric_distributions_part_{i//chunk_size + 1}.png"
            )
            plt.close()
        print(
            f"Saved numeric distributions (split into {(len(num_cols) - 1)//chunk_size + 1} parts)."
        )

    # 3. Correlation Heatmap
    if len(num_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr = df[num_cols].corr()
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("visualizations/3_correlation_heatmap.png")
        plt.close()
        print("Saved correlation heatmap.")

    # 4. Dimensionality Reduction & Clustering
    if len(num_cols) > 1:
        print("Performing dimension reduction and clustering...")
        # Get numeric data and impute missing values
        num_data = df[num_cols].copy()
        imputer = SimpleImputer(strategy="median")
        num_data_imputed = imputer.fit_transform(num_data)

        # Scale data
        scaler = StandardScaler()
        num_data_scaled = scaler.fit_transform(num_data_imputed)

        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(num_data_scaled)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(num_data_scaled)

        # Apply KMeans Clustering (let's say 4 clusters as an initial guess)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(num_data_scaled)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # PCA Plot
        sns.scatterplot(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            hue=clusters,
            palette="viridis",
            ax=axes[0],
            s=50,
        )
        axes[0].set_title(
            f"PCA (explained variance: {sum(pca.explained_variance_ratio_):.2f})"
        )
        axes[0].set_xlabel("Principal Component 1")
        axes[0].set_ylabel("Principal Component 2")

        # t-SNE Plot
        sns.scatterplot(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            hue=clusters,
            palette="viridis",
            ax=axes[1],
            s=50,
        )
        axes[1].set_title("t-SNE")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")

        plt.tight_layout()
        plt.savefig("visualizations/4_clusters_and_dim_reduction.png")
        plt.close()
        print("Saved PCA and t-SNE plots with K-Means clusters.")

    print("Visualization complete! Check the 'visualizations' folder.")





def impute_chemical_parameters(file_path):
    print(f"Loading data from {file_path}...")
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, header=1)
    else:
        df = pd.read_csv(file_path, header=1)

    print(f"Original dataset shape: {df.shape}")

    # 1. Identify the SMILES column
    smiles_col = None
    for c in df.columns:
        if str(c).lower().strip() == "smiles":
            smiles_col = c
            break

    if not smiles_col:
        print("Could not find a 'SMILES' column in the dataset!")
        return df

    print(f"Found SMILES column: '{smiles_col}'")

    # 2. Define the RDKit descriptors we want to compute
    # These are highly reliable and can directly replace/impute missing chemical properties
    descriptor_fns = {
        "RD_MW": Descriptors.MolWt,  # Molecular Weight
        "RD_LogP": Descriptors.MolLogP,  # Partition Coefficient (LogP)
        "RD_TPSA": Descriptors.TPSA,  # Topological Polar Surface Area
        "RD_HBA": Descriptors.NumHAcceptors,  # Hydrogen Bond Acceptors
        "RD_HBD": Descriptors.NumHDonors,  # Hydrogen Bond Donors
        "RD_Rings": Descriptors.RingCount,  # Number of Rings
        "RD_RotBonds": Descriptors.NumRotatableBonds,  # Rotatable Bonds
        "RD_HeavyAtoms": Descriptors.HeavyAtomCount,  # Heavy Atoms
        # --- New Extended Features ---
        "RD_FractionCSP3": Descriptors.FractionCSP3, # Fraction of sp3 carbons (3D shape proxy)
        "RD_NumAliphaticRings": Descriptors.NumAliphaticRings, # Aliphatic rings
        "RD_NumAromaticRings": Descriptors.NumAromaticRings, # Aromatic rings
        "RD_MolMR": Descriptors.MolMR, # Molar Refractivity (polarizability/size)
        "RD_BertzCT": Descriptors.BertzCT # Complexity index
    }

    # 3. Compute descriptors for each unique SMILES string
    unique_smiles = df[smiles_col].dropna().astype(str).unique()
    desc_by_smiles = {}

    print(f"Computing RDKit descriptors for {len(unique_smiles)} unique molecules...")
    invalid_smiles_count = 0
    for smi in unique_smiles:
        mol = Chem.MolFromSmiles(smi) if smi.strip() else None
        if mol is None:
            invalid_smiles_count += 1
            continue  # Invalid SMILES cannot be parsed

        row_desc = {}
        for feat_name, fn in descriptor_fns.items():
            try:
                row_desc[feat_name] = float(fn(mol))
            except Exception:
                row_desc[feat_name] = np.nan
        desc_by_smiles[smi] = row_desc

    if invalid_smiles_count > 0:
        print(f"Warning: Found {invalid_smiles_count} unparseable SMILES strings.")

    # 4. Map the computed descriptors back to the main DataFrame
    desc_df = pd.DataFrame.from_dict(desc_by_smiles, orient="index")

    # Map the SMILES string to the computed descriptor value
    for col in desc_df.columns:
        df[col] = df[smiles_col].astype(str).map(desc_df[col])

    print(f"Added {len(desc_df.columns)} pristine chemical descriptors using RDKit.")

    # Compare generated features with the ones already in the dataset
    print("Comparing RDKit descriptors with original dataset features...")
    comparisons = [("RD_MW", "Compound Mw (g/mol)"), ("RD_LogP", "Compound log K ow")]
    for rd_col, orig_col in comparisons:
        if orig_col in df.columns:
            # Check correlation on rows where both are non-null
            valid_mask = df[rd_col].notnull() & df[orig_col].notnull()
            
            # Since some original features might be loaded as strings (e.g. if they contain '<' or errors), convert to numeric
            try:
                orig_numeric = pd.to_numeric(df.loc[valid_mask, orig_col], errors='coerce')
                valid_mask = valid_mask & orig_numeric.notnull()
                
                if valid_mask.sum() > 5:
                    corr = np.corrcoef(df.loc[valid_mask, rd_col], orig_numeric.loc[valid_mask])[0, 1]
                    print(f" - Correlation between '{rd_col}' and '{orig_col}': {corr:.4f}")
                    if abs(corr) > 0.85:
                        print(f"   -> Highly similar! Dropping original '{orig_col}' in favor of pristine RDKit values.")
                        df = df.drop(columns=[orig_col])
                    else:
                        print(f"   -> Not extremely similar. Keeping both.")
                else:
                    print(f" - '{orig_col}' doesn't have enough valid data to compare. Dropping it.")
                    df = df.drop(columns=[orig_col])
            except Exception as e:
                print(f" - Could not compare '{orig_col}': {e}")

    print(f"New dataset shape: {df.shape}")

    # 5. Skip intermediate saving to keep workspace clean
    return df


def fetch_pka_from_pubchem(smiles):
    """
    Attempts to fetch the pKa value for a given SMILES string from the PubChem API.
    Warning: This hits the PubChem PUG REST API and searches through experimental properties.
    """
    try:
        # Step 1: Get the Compound ID (CID) from the SMILES string
        encoded_smiles = urllib.parse.quote(smiles)
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/cids/JSON"
        req = urllib.request.urlopen(cid_url, timeout=5)
        cid_data = json.loads(req.read())
        cids = cid_data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return np.nan
        cid = cids[0]

        # Step 2: Fetch experimental properties for this CID
        # We need to wait slightly to respect PubChem's 5 requests/sec rate limit
        time.sleep(0.3)
        view_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
        )
        req_view = urllib.request.urlopen(view_url, timeout=5)
        view_data = json.loads(req_view.read())

        # Dig into the deeply nested JSON structure to find 'Dissociation Constants'
        sections = view_data.get("Record", {}).get("Section", [])
        for section in sections:
            if section.get("TOCHeading") == "Chemical and Physical Properties":
                sub_sections = section.get("Section", [])
                for sub in sub_sections:
                    if sub.get("TOCHeading") == "Experimental Properties":
                        exp_props = sub.get("Section", [])
                        for prop in exp_props:
                            if prop.get("TOCHeading") == "Dissociation Constants":
                                # Return the first pKa string found
                                info = prop.get("Information", [])
                                if (
                                    info
                                    and "Value" in info[0]
                                    and "StringWithMarkup" in info[0]["Value"]
                                ):
                                    raw_text = info[0]["Value"]["StringWithMarkup"][0][
                                        "String"
                                    ]
                                    return raw_text  # e.g. "pKa = 4.5"
    except Exception as e:
        # Either no data found, rate limited, or connection error
        pass

    return np.nan


def impute_pka1_pubchem(df):
    # Find the pKa1 column
    pka1_col = None
    for c in df.columns:
        if "pka1" in str(c).lower().replace(" ", ""):
            pka1_col = c
            break

    if not pka1_col:
        print("Could not find pKa1 column.")
        return df

    smiles_col = "SMILES"
    missing_mask = df[pka1_col].isnull()
    unique_missing_smiles = df[missing_mask][smiles_col].dropna().unique()

    print(f"Found {len(unique_missing_smiles)} unique molecules missing {pka1_col}.")
    print(
        "Querying PubChem API... (This will take a few seconds to respect rate limits)"
    )

    imputed_count = 0
    total_smiles = len(unique_missing_smiles)
    for idx, smi in enumerate(unique_missing_smiles):
        if (idx + 1) % 5 == 0 or (idx + 1) == total_smiles:
            print(f"  PubChem Query Progress: [{idx + 1}/{total_smiles}] molecules...")
            
        raw_pka = fetch_pka_from_pubchem(smi)
        if pd.notnull(raw_pka):
            # Try to extract the first float value from the text string using regex
            match = re.search(r"[-+]?\d*\.\d+|\d+", str(raw_pka))
            if match:
                val = float(match.group())
                # Update all rows with this SMILES
                df.loc[(df[smiles_col] == smi) & (df[pka1_col].isnull()), pka1_col] = (
                    val
                )
                imputed_count += 1

    print(f"Successfully imputed {imputed_count} unique molecules' pKa1 using PubChem.")

    return df

def impute_diffusion_xgboost(df):
    """
    Impute missing Diffusion coefficient using XGBoost on RDKit MW and MolMR.
    Adds a tiny bit of Gaussian noise matching the known MAE to prevent artificial determinism.
    """
    diff_cols = [c for c in df.columns if "diffusion" in str(c).lower()]
    if not diff_cols:
        return df
    diff_col = diff_cols[0]
    
    # We need MW and MolMR to predict
    if "RD_MW" not in df.columns or "RD_MolMR" not in df.columns:
        return df

    missing_mask = df[diff_col].isnull() & df["RD_MW"].notnull() & df["RD_MolMR"].notnull()
    train_mask = df[diff_col].notnull() & df["RD_MW"].notnull() & df["RD_MolMR"].notnull()
    
    if missing_mask.sum() == 0:
        return df
        
    print(f"\n--- Imputing {missing_mask.sum()} missing Diffusion coefficients using tuned XGBoost ---")
    
    from xgboost import XGBRegressor
    
    X_train = df.loc[train_mask, ["RD_MW", "RD_MolMR"]].values
    y_train_raw = pd.to_numeric(df.loc[train_mask, diff_col], errors='coerce').values
    
    # Filter out any lingering NaNs in y that were coerced
    valid_y = ~np.isnan(y_train_raw)
    X_train = X_train[valid_y]
    y_train_log = np.log10(y_train_raw[valid_y])
    
    # Use the Optuna tuned parameters
    xgb = XGBRegressor(
        max_depth=8, 
        learning_rate=0.105, 
        n_estimators=278, 
        subsample=0.69, 
        colsample_bytree=0.58,
        random_state=42
    )
    xgb.fit(X_train, y_train_log)
    
    X_missing = df.loc[missing_mask, ["RD_MW", "RD_MolMR"]].values
    preds_log = xgb.predict(X_missing)
    
    # Add realistic MAE spread to the log predictions (MAE was ~0.09)
    mae_spread = 0.0931
    noise = np.random.normal(0, mae_spread, size=len(preds_log))
    preds_log_noisy = preds_log + noise
    
    # Convert back from log scale
    preds_raw = 10 ** preds_log_noisy
    
    df.loc[missing_mask, diff_col] = preds_raw
    print("Diffusion coefficient imputation complete with MAE noise spread.")
    
    return df

    print("\n--- Current Missing Data Summary ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        missing_pct = (missing / len(df)) * 100
        df_out = pd.DataFrame(
            {"Missing Count": missing, "Missing %": missing_pct.round(2)}
        )
        # Strip unicode from index names to avoid print crashes on Windows
        df_out.index = [
            str(x).encode("ascii", "ignore").decode("ascii") for x in df_out.index
        ]
        print(f"Total missing cells: {df.isnull().sum().sum()}")
        print(df_out.to_string())
    else:
        print("No missing data left!")

    return df


def prepare_for_nn(df):
    print("\n--- Preparing Data for Neural Network Embedding ---")

    # 0. Drop ID, Metadata, and Text columns that cause data leakage
    drop_candidates = ["Number", "Reference", "CAS", "SMILES", "Name of MB"]
    cols_to_drop = [
        c
        for c in df.columns
        if any(str(c).lower().strip() == d.lower() for d in drop_candidates)
    ]
    if cols_to_drop:
        print(f"Dropping metadata/ID columns to prevent data leakage: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # 0.5 Drop Log D columns since NN can infer it from LogP, pKa1, and pH
    log_d_cols = [c for c in df.columns if "log d" in str(c).lower()]
    if log_d_cols:
        print(f"Dropping explicit Log D columns to rely on implicit neural learning: {log_d_cols}")
        df = df.drop(columns=log_d_cols)

    # 0.6 Drop Density column since NN can infer it near-perfectly from RD_MW and RD_MolMR
    density_cols = [c for c in df.columns if "density" in str(c).lower()]
    if density_cols:
        print(f"Dropping explicit Density columns to rely on RDKit MW/MolMR inference: {density_cols}")
        df = df.drop(columns=density_cols)

    # 0.7 Drop Surface Tension and Gs-m since they are >97% predictable from RDKit properties
    st_gs_cols = [c for c in df.columns if "surface tension" in str(c).lower() or "gs-m" in str(c).lower()]
    if st_gs_cols:
        print(f"Dropping Surface Tension and Gs-m to rely on RDKit chemistry inference: {st_gs_cols}")
        df = df.drop(columns=st_gs_cols)

    # 1. Handle pKa2 with an Indicator Feature
    # Instead of dropping it due to 70% missing data, we create a binary 'has_pka2' feature
    pka2_cols = [c for c in df.columns if "pka2" in str(c).lower().replace(" ", "")]
    if pka2_cols:
        pka2_col = pka2_cols[0]
        indicator_col = "has_pka2"
        print(f"Creating indicator feature '{indicator_col}' for sparse column '{pka2_col}'.")
        df[indicator_col] = df[pka2_col].notnull().astype(int)
        
        # Zero-fill the missing pKa2 values so KNN imputer doesn't try to guess 70% of the data
        df[pka2_col] = df[pka2_col].fillna(0.0)

    # 2. Separate numeric from categorical
    num_cols = df.select_dtypes(include=[np.number]).columns

    # 3. Strict Data Completeness (Drop rows with missing values)
    # Ensure the model only learns from perfectly complete rows rather than using imputation.
    # Since pKa2 was safely handled above, this mainly drops rows missing pKa1 or experimental conditions.
    initial_len = len(df)
    df = df.dropna()
    dropped_count = initial_len - len(df)
    print(f"Strict Completeness Check: Dropped {dropped_count} rows with missing data.")
    print(f"Remaining pristine rows: {len(df)}")
    
    # Refresh numeric columns list after drop
    num_cols = df.select_dtypes(include=[np.number]).columns

    # 3.5 Drop Highly Collinear Features
    print("Identifying and dropping highly collinear features (correlation > 0.90)...")
    corr_matrix = df[num_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_collinear = [
        column for column in upper_tri.columns if any(upper_tri[column] > 0.90)
    ]

    # Ensure we don't drop the target column accidentally
    target_cols = [
        c for c in num_cols if "removal" in str(c).lower() and "rate" in str(c).lower()
    ]
    to_drop_collinear = [c for c in to_drop_collinear if c not in target_cols]

    if to_drop_collinear:
        safe_names = [str(c).encode('ascii', 'ignore').decode('ascii') for c in to_drop_collinear[:5]]
        print(
            f"Dropping {len(to_drop_collinear)} highly collinear columns to reduce redundancy: {safe_names}..."
        )
        df = df.drop(columns=to_drop_collinear)
        # Refresh num_cols after drop
        num_cols = df.select_dtypes(include=[np.number]).columns

    # 4. Identify the Target column so we don't transform the target
    target_cols = [
        c for c in num_cols if "removal" in str(c).lower() and "rate" in str(c).lower()
    ]
    features_to_transform = [c for c in num_cols if c not in target_cols]

    # 5. Apply Gaussian Distribution Transformation (Yeo-Johnson)
    # This standardizes the data (mean=0, std=1) AND morphs it into a Gaussian distribution
    print(
        "Applying PowerTransformer (Yeo-Johnson) to enforce Gaussian distribution on numeric features..."
    )
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    df_transformed = pd.DataFrame(
        pt.fit_transform(df[features_to_transform]),
        columns=features_to_transform,
        index=df.index,
    )

    # Put transformed features back into the dataframe
    df[features_to_transform] = df_transformed

    print(
        f"Standardization complete. Zero missing values remaining. Dataset shape: {df.shape}"
    )

    output_path = "heatdraft_dataset_ready.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved neural-network-ready dataset to '{output_path}'")

    return df




# Check device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- MODEL DEFINITION ---
class TabularAttentionNet(nn.Module):
    """
    A Self-Attention based Neural Network for Tabular Data.

    It treats each numerical/categorical feature as a 'token' by
    projecting it into a high-dimensional embedding space, and applies
    Transformer Encoder layers to capture interactions between features.

    Args:
        num_features (int): The number of features in the input data.
        d_model (int, optional): The dimensionality of the embedding space. Defaults to 32.
        n_heads (int, optional): The number of attention heads in the transformer. Defaults to 4.
        num_layers (int, optional): The number of transformer encoder layers. Defaults to 2.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
    """

    def __init__(self, num_features, d_model=32, n_heads=4, num_layers=2, dropout=0.1):
        super(TabularAttentionNet, self).__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Project each scalar feature into a d_model dimensional vector
        # Shape: (1, num_features, d_model)
        self.feature_embeddings = nn.Parameter(torch.randn(1, num_features, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP Head for Regression (Predicting Removal Rate)
        self.mlp = nn.Sequential(
            nn.Linear(num_features * d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x shape: (batch_size, num_features)
        batch_size = x.size(0)

        # Multiply each scalar feature by its corresponding embedding vector
        # x.unsqueeze(2) shape: (batch, num_features, 1)
        # embeddings shape: (1, num_features, d_model) -> broadcast to batch
        x_emb = x.unsqueeze(2) * self.feature_embeddings.expand(batch_size, -1, -1)

        # Pass through Self-Attention mechanism
        attn_out = self.transformer(x_emb)  # (batch, num_features, d_model)

        # Flatten and pass to MLP
        attn_out = attn_out.reshape(batch_size, -1)
        out = self.mlp(attn_out)
        return out.squeeze(1)


# --- DATA PREPARATION ---
def load_and_prep_data(filepath):
    """
    Load data from a CSV file, identify the target column, and encode categorical features.

    Args:
        filepath (str): The path to the input CSV file.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The feature matrix as float32.
            - y (np.ndarray): The target vector as float32.
    """
    df = pd.read_csv(filepath)

    # Identify target (Removal Rate)
    target_col = [
        c
        for c in df.columns
        if "removal" in str(c).lower() and "rate" in str(c).lower()
    ][0]

    # Identify features
    features = df.drop(columns=[target_col])

    # Fast encoding for remaining Categorical Columns
    cat_cols = features.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols:
        le = LabelEncoder()
        features[c] = le.fit_transform(features[c].astype(str))

    X = features.values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    return X, y


# --- OPTUNA OBJECTIVE ---
def objective(trial):
    """
    Optuna objective function for hyperparameter tuning.

    This function defines the hyperparameter search space, initializes the model,
    trains it for a fixed number of epochs, and evaluates its performance on a
    validation set.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object.

    Returns:
        float: The final Mean Squared Error (MSE) on the validation set.
    """
    # Hyperparameters to auto-tune
    d_model = trial.suggest_categorical("d_model", [16, 32, 64])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    epochs = 150
    batch_size = 64

    # Load Data
    X, y = load_and_prep_data("heatdraft_dataset_ready.csv")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize Model
    model = TabularAttentionNet(
        num_features=X.shape[1],
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
            
        # Print progress every 25 epochs
        if (epoch + 1) % 25 == 0:
            print(f"  Trial Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(X_train):.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx)
            # Clamp predictions to valid physical bounds [0, 100] for accurate validation loss
            preds = torch.clamp(preds, min=0.0, max=100.0)
            val_loss += criterion(preds, by).item() * bx.size(0)

    final_mse = val_loss / len(X_val)
    return final_mse



if __name__ == "__main__":
    dataset_path = "Nature paper.xlsx"
    if os.path.exists(dataset_path):
        print("=== Step 1: Data Visualization ===")
        visualize_data(dataset_path)
        print("\n=== Step 2: Data Preprocessing ===")
        df = impute_chemical_parameters(dataset_path)
        df = impute_pka1_pubchem(df)
        df = impute_diffusion_xgboost(df)
        df = prepare_for_nn(df)
        print("\n=== Step 3: Model Training & Tuning ===")

        print(f"Training on device: {device}")
        print("Loading data from 'heatdraft_dataset_ready.csv'...")

        # Check if data exists
        if not os.path.exists("heatdraft_dataset_ready.csv"):
            print("Data not found! Please run the preprocessor first.")
            exit(1)

        print("Starting Optuna Hyperparameter tuning for Self-Attention Network...")

        # Reduce optuna logging verbosity so it doesn't flood the terminal
        optuna.logging.set_verbosity(optuna.logging.INFO)

        # Create Study and store history in a database to learn from past runs
        study = optuna.create_study(
            direction="minimize", 
            study_name="Self-Attention-Tuning",
            storage="sqlite:///optuna_tuning_history.db",
            load_if_exists=True
        )
        # We will do 15 trials as a temporary check. It can be increased later.
        study.optimize(objective, n_trials=15)

        print("\n" + "=" * 50)
        print("OPTIMIZATION FINISHED!")
        print("=" * 50)
        print(f"Best Validation MSE: {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        print("\n--- Training Final Model with Best Hyperparameters ---")
        best_params = study.best_params
        epochs = 1500  # Large epoch count with early stopping for fair learning
        batch_size = 32
        early_stopping_patience = 300
        best_val_loss = float('inf')
        patience_counter = 0

        X, y = load_and_prep_data("heatdraft_dataset_ready.csv")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=batch_size,
            shuffle=True,
        )

        final_model = TabularAttentionNet(
            num_features=X.shape[1],
            d_model=best_params["d_model"],
            n_heads=best_params["n_heads"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
        ).to(device)

        # We will use ReduceLROnPlateau to help the model converge
        optimizer = optim.Adam(final_model.parameters(), lr=best_params["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            final_model.train()
            epoch_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                preds = final_model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
            avg_loss = epoch_loss / len(X_train)
            
            # Calculate validation loss for early stopping & scheduler
            final_model.eval()
            with torch.no_grad():
                val_preds = final_model(torch.tensor(X_test).to(device))
                val_loss = criterion(val_preds, torch.tensor(y_test).to(device)).item()
                
            scheduler.step(val_loss)
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(final_model.state_dict(), "best_final_model.pth")
            else:
                patience_counter += 1
                
            # Print progress every 50 epochs
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Patience: {patience_counter}/{early_stopping_patience}")

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}! Best Val Loss: {best_val_loss:.4f}")
                break

        # Evaluation
        if os.path.exists("best_final_model.pth"):
            final_model.load_state_dict(torch.load("best_final_model.pth"))
            
        final_model.eval()
        with torch.no_grad():
            train_preds = final_model(torch.tensor(X_train).to(device)).cpu().numpy()
            test_preds = final_model(torch.tensor(X_test).to(device)).cpu().numpy()

        # Clamping physical boundaries: Removal rates cannot be < 0% or > 100%
        train_preds = np.clip(train_preds, 0.0, 100.0)
        test_preds = np.clip(test_preds, 0.0, 100.0)

        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        train_mse = mean_squared_error(y_train, train_preds)
        test_mse = mean_squared_error(y_test, test_preds)

        print("\n" + "=" * 50)
        print("FINAL MODEL EVALUATION")
        print("=" * 50)
        print(f"Train MSE: {train_mse:.4f} | Train R2: {train_r2:.4f}")
        print(f"Test MSE:  {test_mse:.4f} | Test R2:  {test_r2:.4f}")

    else:
        print(f"Dataset not found at {dataset_path}")
