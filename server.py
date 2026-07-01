import pandas as pd
import numpy as np
import torch
import joblib
import optuna
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from heatdraft_pipeline import TabularAttentionNet
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='app')
CORS(app)

print("Loading reference database and preprocessors...")
try:
    ref_df = pd.read_csv("heatdraft_reference_database.csv")
    pt = joblib.load("power_transformer.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    study = optuna.load_study(study_name="Self-Attention-Tuning", storage="sqlite:///optuna_tuning_history.db")
    best_params = study.best_params
    
    model = TabularAttentionNet(
        num_features=len(feature_names),
        d_model=best_params["d_model"],
        n_heads=best_params["n_heads"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
    ).to(device)
    
    model.load_state_dict(torch.load("best_final_model.pth", map_location=device))
    model.eval()
    print("PyTorch Backend successfully loaded and ready for real-time inference.")
except Exception as e:
    print(f"Warning: Ensure heatdraft_pipeline.py has been run to generate the required files. Error: {e}")

@app.route('/')
def index():
    return send_from_directory('app', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('app', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    cont_name = data.get('contName')
    mb_name = data.get('mbName')
    
    # Find matching reference row for baseline properties
    match = ref_df[(ref_df['Types of contaminants'] == cont_name) & (ref_df['Name of MB'] == mb_name)]
    if len(match) == 0:
        # Fallback: combine the first occurrence of each
        cont_match = ref_df[ref_df['Types of contaminants'] == cont_name]
        mb_match = ref_df[ref_df['Name of MB'] == mb_name]
        if len(cont_match) == 0 or len(mb_match) == 0:
            return jsonify({"error": "Unknown contaminant or membrane."}), 400
        row = cont_match.iloc[0].copy()
        for col in mb_match.columns:
            if "MB " in col or "pore" in col.lower() or "zeta" in col.lower() or "Type of MB" in col:
                row[col] = mb_match.iloc[0][col]
    else:
        row = match.iloc[0].copy()
        
    # Override with live user inputs
    if data.get('pressure') is not None:
        row['Pressure (kPa)'] = float(data.get('pressure'))
    if data.get('time') is not None:
        row['Measurement time (min)'] = float(data.get('time'))
    if data.get('conc') is not None:
        row['Initial concentration of compound (mg/L)'] = float(data.get('conc'))
    if data.get('ph') is not None:
        row['pH'] = float(data.get('ph'))
        
    # Recreate engineered features
    safe_pore = 1e-6 if row.get("MB pore radius rp (nm)", 0) == 0 else row["MB pore radius rp (nm)"]
    row["Steric_Hindrance_Ratio"] = row.get("Compound size (nm)", 0) / safe_pore
    
    df_single = pd.DataFrame([row])
    
    # Handle pka2 Indicator
    pka2_cols = [c for c in df_single.columns if "pka2" in str(c).lower().replace(" ", "")]
    if pka2_cols:
        pka2_col = pka2_cols[0]
        df_single["has_pka2"] = df_single[pka2_col].notnull().astype(int)
        df_single[pka2_col] = df_single[pka2_col].fillna(0.0)
        
    # Process numeric & categorical
    input_dict = {}
    for col in feature_names:
        val = df_single.iloc[0].get(col, 0)
        if col in label_encoders:
            try:
                val_encoded = label_encoders[col].transform([str(val)])[0]
                input_dict[col] = val_encoded
            except:
                input_dict[col] = 0
        else:
            input_dict[col] = float(val) if pd.notnull(val) else 0.0
            
    final_df = pd.DataFrame([input_dict])[feature_names]
    
    # PowerTransform continuous features
    num_cols_for_pt = [c for c in final_df.columns if c not in label_encoders]
    final_df[num_cols_for_pt] = pt.transform(final_df[num_cols_for_pt])
    
    # PyTorch Inference
    X_tensor = torch.tensor(final_df.values.astype(np.float32)).to(device)
    
    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy()[0]
        
    pred = float(np.clip(pred, 0.0, 100.0))
    
    return jsonify({
        "predicted_removal": pred, 
        "confidence": "PyTorch High-Rate Optimized"
    })

if __name__ == '__main__':
    print("\nStarting local PyTorch backend on http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=False)
