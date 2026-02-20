html_content = """<!DOCTYPE html>
<html>
<head>
    <title>HeatDraft ML Pipeline Flowchart</title>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({ startOnLoad: true, theme: 'default' });
    </script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; padding: 20px; text-align: center; }
        h2 { color: #333; }
        .mermaid { margin: 0 auto; display: flex; justify-content: center; }
    </style>
</head>
<body>
    <h2>HeatDraft ML Pipeline Architecture</h2>
    <div class="mermaid">
    graph TD
        %% Core Data Loading
        Input("Raw Data:\\n.csv / .xlsx") --> Load["Load Dataframe\\nNormalize Column Names\\nFallback headers if needed"]
        Load --> SMILES{"Contains 'SMILES' ?"}
        
        %% Cheminformatics
        SMILES -- Yes --> RDKit["RDKit Cheminformatics Engine\\nGenerate 8 Physical Descriptors:\\nMW, LogP, TPSA, HBD, HBA, Rings..."]
        SMILES -- No --> Clean["Initial Cleansing"]
        RDKit --> Clean
        
        %% Initial Cleaning
        Clean --> RowDrop["Drop rows missing Target Value\\nDrop Data-Leakage columns 'Unnamed: 0'"]
        
        %% Data Splitting
        RowDrop --> Split["Stratified Train/Test Split\\n80% Train | 20% Test\\nPreserves 'High Performance' ratios"]
        
        %% Drop Out High Cardinality
        Split --> CatDrop["Categorical Culling:\\nDrop text vars with >60 unique values\\nDrop variables with >95% uniqueness\\nPrevents UUID Memorization"]
        
        %% Strict Multicollinearity Protocol
        CatDrop --> FeatureSelection["Feature Selection Protocol\\n(Executing on Train Set Only)"]
        
        subgraph Mathematical Feature Culling
        FeatureSelection --> Sparsity["Drop Sparse Features >95% missing"]
        Sparsity --> NZV["Drop Near-Zero Variance Numbers"]
        NZV --> Pearson["Pearson Correlation Clustering > 0.92"]
        Pearson --> MI["Mutual Information Tie-Breaker:\\nKeep mathematically best feature\\nin identical clusters, drop rest"]
        MI --> VIF["Variance Inflation Factor Recursion:\\nRecursively drop worst feature\\nuntil matrix VIF < 10.0"]
        end
        
        %% Target Adjustments
        VIF --> TargetTransform["Logit Target Bound Transformation:\\nForce limits to 0.0 - 1.0 probability\\n'log(p / (1-p))'"]
        TargetTransform --> Impute["Median Imputation & Robust Scaling"]
        
        %% Training Configuration
        Impute --> Tuning["Bayesian Hyperparameter Engine\\n(Optuna - 5-Fold Cross Validation)"]
        
        subgraph Intelligent AI Tuning
        Tuning --> XGB["XGBoost base modeling"]
        Tuning --> Extra["ExtraTrees base modeling"]
        Tuning --> RF["Random Forest base modeling"]
        Tuning --> Hist["HistGradientBoosting modeling"]
        
        XGB -.-> DynamicWeights
        Extra -.-> DynamicWeights
        RF -.-> DynamicWeights
        Hist -.-> DynamicWeights
        end
        
        %% Penalties
        DynamicWeights(("Dynamic High-Zone Weighting:\\nHeavily penalize the AI\\nif it hallucinates >90% successes"))
        
        %% Final Ensemble Assembly
        DynamicWeights --> Top3["Extract Top 3 Best Base Architectures"]
        Top3 --> Ensemble["Stacking Regressor Assembly:\\nFuse optimal predictions into\\nRidgeCV Meta-Model"]
        
        %% Mixture of Experts
        Ensemble --> MoE{"Mixture of Experts\\nEnabled?"}
        MoE -- Yes --> PCA["Principal Component Dimensionality Reduction"]
        PCA --> KMeans["K-Means Clustering\\nIdentify 2 unique chemical physics zones"]
        KMeans --> Expert1["Train Full Ensemble\\nClone strictly on Zone 1"]
        KMeans --> Expert2["Train Full Ensemble\\nClone strictly on Zone 2"]
        
        %% Execution and Output
        Expert1 --> Output
        Expert2 --> Output
        MoE -- No --> Output
        
        Output["Generate Analytics:\\nInverse-Transform Logits back to 0-100%"]
        Output --> Dash["Generate Dashboard PNG\\nWrite JSON Model Artifacts\\nBuild Correlation Heatmaps"]
        Dash --> LowFail["Diagnose Low Performance <90%\\nCalculate Gap Medians"]
        LowFail --> Save[/"Saved in outputs/ directory"/]
        
        %% Styles
        classDef primary fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px,color:#0b3c5d
        classDef danger fill:#ffebee,stroke:#e53935,stroke-width:2px,color:#b71c1c
        classDef success fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#1b5e20
        classDef ML fill:#fff3e0,stroke:#fb8c00,stroke-width:2px,color:#e65100
        
        class Load,SMILES,Clean,RowDrop,Split primary
        class Sparsity,NZV,Pearson,MI,VIF,CatDrop danger
        class TargetTransform,DynamicWeights success
        class Tuning,XGB,Extra,RF,Hist,Top3,Ensemble,MoE,PCA,KMeans,Expert1,Expert2 ML
    </div>
</body>
</html>
"""

with open("HeatDraft_Pipeline_Flowchart.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Flowchart HTML generated successfully.")
