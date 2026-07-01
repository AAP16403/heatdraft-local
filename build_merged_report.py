import os
import subprocess

parts = [
    "Part_1_Detailed.md",
    "Part_2_Detailed.md",
    "Part_3_Detailed.md",
    "Part_4_Detailed.md"
]

merged_content = "# HeatDraft Machine Learning Pipeline: Comprehensive Architecture Report\n\n"

for part in parts:
    if not os.path.exists(part):
        print(f"Warning: {part} not found. Skipping.")
        continue
        
    with open(part, "r", encoding="utf-8") as f:
        content = f.read()
        
        # Inject images where relevant for Part 1
        if "Part_1" in part:
            content = content.replace(
                "### Why I did it:\nI needed to mathematically", 
                "**Visual Proof:**\n\n<img src='visualizations/1_missing_values.png' width='500'/>\n\n*Figure 1: Missing values distribution prior to deterministic imputation.*\n\n<img src='visualizations/4_clusters_and_dim_reduction.png' width='700'/>\n\n*Figure 2: Dimensionality reduction (PCA & t-SNE) showing intrinsic clustering manifolds.*\n\n### Why I did it:\nI needed to mathematically"
            )
        
        # Inject images where relevant for Part 4
        if "Part_4" in part:
            content = content.replace(
                "## 1. Permutation Feature Importance",
                "## 0. Final Prediction Accuracy\n\n### What I did:\nI evaluated the final neural network weights on the completely unseen test set, plotting the Predicted vs Actual removal rates along an ideal 1:1 regression line to visually verify the Root Mean Squared Error (RMSE).\n\n**Visual Proof:**\n\n<img src='visualizations/5_predicted_vs_actual.png' width='600'/>\n\n*Figure 3: The final predictive accuracy of the neural network on unseen data.*\n\n---\n\n## 1. Permutation Feature Importance"
            )
            
            content = content.replace(
                "### Why I did it:\nNeural networks are notoriously", 
                "**Visual Proof:**\n\n<img src='visualizations/6_feature_importance.png' width='600'/>\n\n*Figure 4: Permutation Importance detailing exactly which physical traits drove the network's decisions.*\n\n### Why I did it:\nNeural networks are notoriously"
            )
            
            content = content.replace(
                "### Why I did it:\nReporting an average accuracy", 
                "**Visual Proof:**\n\n<img src='visualizations/7_error_analysis.png' width='800'/>\n\n*Figure 5: Scatter plot isolating absolute prediction error against key physical constraints to identify boundary failures.*\n\n### Why I did it:\nReporting an average accuracy"
            )
            
        merged_content += content + "\n\n<div style='page-break-after: always;'></div>\n\n"

# Write the final merged markdown
with open("HeatDraft_Comprehensive_Report.md", "w", encoding="utf-8") as f:
    f.write(merged_content)

print("Merged markdown created. Attempting to convert to PDF via npx md-to-pdf...")
try:
    subprocess.run(["npx", "--yes", "md-to-pdf", "HeatDraft_Comprehensive_Report.md"], shell=True, check=True)
    print("\nSUCCESS: HeatDraft_Comprehensive_Report.pdf has been generated!")
except subprocess.CalledProcessError:
    print("\nWARNING: npx md-to-pdf failed. The markdown file 'HeatDraft_Comprehensive_Report.md' is ready, but you may need to export it to PDF manually in VSCode.")
