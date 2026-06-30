# Part 1: Data Acquisition, Imputation, and Exploratory Physics

This document details exactly what I implemented in the first half of the HeatDraft ML Pipeline, and the scientific rationale behind why I made these specific architectural and data processing decisions.

---

## 1. Exploratory Data Analysis & Dimensionality Reduction

### What I did:
I built an automated `visualize_data()` module that generates `.describe()` statistical profiles, missing value heatmaps, feature distribution histograms, and correlation matrices. Furthermore, I implemented Principal Component Analysis (PCA) and t-SNE to compress the 38+ chemical features into a 2D plane, grouping them using K-Means clustering (n=4).

### Why I did it:
I needed to mathematically verify the skewness of the raw chemical properties before feeding them to the neural network. If a property like Pressure is heavily skewed, it justifies applying the Yeo-Johnson transformation later. I implemented PCA and t-SNE because chemical datasets exist in high-dimensional space; projecting them down to 2D allowed me to visually prove that distinct structural clusters of membrane-contaminant interactions actually exist in the data, proving that a neural network could theoretically learn these boundaries.

---

## 2. Deterministic Chemical Imputation (RDKit)

### What I did:
Instead of using standard data science techniques (like mean/median filling or KNN imputation) to fill missing values, I integrated **RDKit**, an open-source chemoinformatics engine. I wrote a loop that reads the `SMILES` string of the contaminant and mathematically calculates intrinsic properties such as Molecular Weight (`MolWt`), Hydrophobicity (`MolLogP`), Topological Polar Surface Area (`TPSA`), and Hydrogen Bond capabilities.

### Why I did it:
Standard statistical imputation is scientifically invalid for hard chemistry. A molecule's molecular weight or polar surface area is a fixed physical constant governed by its atomic graph, not an average of its neighbors in an Excel sheet. I chose to use RDKit because calculating these values deterministically guarantees 100% accuracy for missing properties, vastly improving the integrity of the training data.

---

## 3. External API Fallback (PubChem)

### What I did:
For properties that RDKit could not calculate locally from topological structure (specifically the Acid Dissociation Constant, `pKa1`), I built a dynamic web-scraper that connects to the **NIH PubChem REST API**. My code queries the chemical string, retrieves the global Compound ID (CID), and extracts the empirical laboratory `pKa` value from the official JSON records.

### Why I did it:
I realized that local software calculation for pKa is notoriously inaccurate because it depends on complex aqueous thermodynamics. By falling back to the NIH PubChem database, I ensured that the neural network was learning from true, peer-reviewed empirical laboratory values rather than estimated noise.

---

## 4. Non-Linear Surrogate Imputation (XGBoost)

### What I did:
To impute the missing **Diffusion Coefficient** values, I built a "model within a model." I trained a standalone **XGBoost Regressor** on the clean subset of the data, using topological properties (`RD_MW`, `RD_LogP`, `RD_TPSA`) as inputs. Crucially, after generating the predictions, I injected artificial Gaussian noise (`np.random.normal(0, 0.0931)`) based on the XGBoost model's known Mean Absolute Error (MAE).

### Why I did it:
The Diffusion Coefficient in water is a non-linear property that is not reliably available on PubChem for niche pharmaceutical compounds. I chose XGBoost because its decision-tree structure excels at finding complex topological relationships in small tabular datasets. 
However, I explicitly injected the MAE noise because if I fed perfectly smooth, deterministic XGBoost predictions into the final Neural Network, the network would realize the data was "fake" (artificially imputed) and exploit that signal. Injecting noise forces the imputed data to mimic the natural variance of real-world laboratory experiments.
