# Part 4: Post-Training Diagnostics and Web Dashboard Integration

This document outlines the graphical diagnostics I built to prove the model's physical understanding, as well as the transition of the inference engine to the web-based React/JS application interface.

---

## 1. Permutation Feature Importance

### What I did:
I wrote a custom script at the end of the PyTorch training loop that mathematically shuffles the values of each physical feature (one by one) across the entire test set. For each shuffled column, it passes the corrupted data through the neural network and records how severely the Mean Squared Error (MSE) explodes. I then plotted the top 15 most important features as a bar chart (`6_feature_importance.png`).

### Why I did it:
Neural networks are notoriously "black boxes." I needed to scientifically prove that the model wasn't just guessing, but was actually relying on real physics to make decisions. By shuffling a feature and measuring the error explosion, I could definitively prove which variables the network cared about. If shuffling the Steric Hindrance Ratio caused a massive spike in error, I successfully proved that the network had learned size-exclusion mechanics.

---

## 2. Erratic Error Analysis Diagnostics

### What I did:
I built an automated Error Analysis visualization (`7_error_analysis.png`). My code calculates the Absolute Prediction Error for every single unseen molecule in the test set. It then plots these errors as a scatter plot against the actual values of the top 3 most important features (like pH or MWCO).

### Why I did it:
Reporting an average accuracy of "76%" is not enough for a complete industrial report. I needed to know exactly *where* and *why* the model failed. By plotting the errors against the physical features, I could visually detect edge-case failures—for example, proving if the model's accuracy remained tight during standard operation but became highly erratic when feed pH exceeded 12. This provided total transparency into the model's limitations and operational boundaries.

---

## 3. Web Dashboard K-Nearest Neighbors (KNN) Inference Engine

### What I did:
Before completing the deployment of the PyTorch API server, I heavily modified the frontend dashboard (`app.js`). Instead of relying on rigid, hard-coded exact string matching to filter results, I engineered a mathematical K-Nearest Neighbors (KNN) inference engine in pure JavaScript. 
The app takes the user's requested membrane and chemical properties, normalizes them, and calculates the multi-dimensional Euclidean Distance against all 1,140 rows of training data to return the 15 closest experimental matches.

### Why I did it:
When users input custom chemical properties into the web app, an exact experimental match rarely exists in the database. Using simple filters caused the app to return empty, broken results. I built the KNN distance engine so that the app could dynamically "reverse-engineer" predictions based on nearest similarity. This ensures the web dashboard always yields a mathematically sound prediction and visually shows the user exactly which historical experiments most closely resemble their input, acting as a highly accurate stop-gap until the raw PyTorch model is fully wired into the backend server.
