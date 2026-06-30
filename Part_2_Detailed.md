# Part 2: Feature Engineering, Tabular Attention, and Optimization

This document details exactly what I implemented in the second half of the HeatDraft ML Pipeline, focusing on why I engineered specific physical features and the rationale behind my custom Neural Network architecture and regularization strategies.

---

## 1. Feature Engineering & Standardization

### 1.1 The Steric Hindrance Ratio (Physics Injection)

**What I did:**
I engineered a brand new feature called the **Steric Hindrance Ratio ($S_h$)**. I wrote code to divide the physical size of the contaminant molecule by the physical pore radius of the membrane ($S_h = \frac{r_c}{r_p}$).

**Why I did it:**
Neural networks are blank slates; they do not inherently understand the physical laws governing nanofiltration. By manually injecting this ratio, I mathematically represented **steric exclusion** for the network. If the ratio is $> 1.0$, the molecule is physically larger than the pore, which mechanically forces the removal rate towards 100%. By feeding this explicit physics rule to the network, I saved the model from having to blindly guess the relationship between compound size and pore radius, significantly accelerating its convergence.

### 1.2 Yeo-Johnson Transformation & Gaussian Normalization

**What I did:**
I applied `sklearn.preprocessing.PowerTransformer(method="yeo-johnson")` to all numerical features. This mathematically morphs heavily skewed distributions into a standard Bell Curve and scales them to have a mean of 0 and a standard deviation of 1.

**Why I did it:**
If I fed raw feature scales into the network (where pH goes from 1-14 but Pressure goes from 100-2000 kPa), the loss gradients would explode or vanish, completely breaking the Adam optimizer during backpropagation. I specifically chose the Yeo-Johnson algorithm because it handles zero and negative values (unlike Box-Cox), ensuring that every single input feature carries an equal, stable mathematical weight into the first layer.

---

## 2. The Tabular Attention Neural Network Architecture

**What I did:**
Instead of using a standard Multi-Layer Perceptron (MLP) or Random Forest, I designed and implemented a custom **TabularAttentionNet** in PyTorch. I programmed it to project each scalar feature into a high-dimensional vector space (`d_model = 32`), essentially treating every spreadsheet column like a "token". I then passed these tokens through an `nn.TransformerEncoderLayer` featuring a Query-Key-Value (QKV) self-attention mechanism.

**Why I did it:**
Standard dense neural networks are historically terrible at tabular data because they mash all columns together indiscriminately in the first layer. I chose a Transformer-based Attention mechanism because it allows the network to dynamically calculate "Attention Weights" *between* different columns. For example, my architecture allows the network to learn to "pay high attention" to `Molecule Charge` *only* when it detects a strongly negative `Membrane Zeta Potential`. This perfectly simulates complex electrostatic repulsion physics dynamically, rather than relying on static weights.

---

## 3. Optuna Hyperparameter Optimization (TPE Algorithm)

**What I did:**
I integrated **Optuna** to optimize the network's structure (embedding width, number of attention heads, layers, and learning rate) using the Tree-structured Parzen Estimator (TPE) algorithm. I bound this study to a persistent SQLite database (`optuna_tuning_history.db`).

**Why I did it:**
Guessing the correct depth and width of a Transformer model is impossible. I used Optuna's Bayesian optimization to mathematically search the hyperparameter space efficiently. I specifically bound it to a persistent SQLite database so that if my training script crashed or I paused the training across multiple days, Optuna would remember exactly which architectural shapes failed previously. This ensured zero wasted compute time.

---

## 4. Dynamic Regularization & Shifting Early Stopping

**What I did:**
I implemented an aggressive multi-layered defense against overfitting:
1. I hardcoded an **L2 Weight Decay** (`1e-4`) in the Adam optimizer.
2. I forced Optuna to select a high **Dropout Minimum** (between 20% and 50%).
3. I programmed a custom **Tandem Shifting Patience** Early Stopping algorithm that only resets patience if the model improves over the *immediately previous* epoch, rather than comparing it to an all-time historical best.

**Why I did it:**
Because my dataset only contains ~1,140 rows, a massive neural network will inevitably try to cheat by memorizing the training data (overfitting). 
I added Weight Decay to penalize the network for relying too heavily on any single feature, and high Dropout to force the network to learn multiple redundant logic pathways.
I wrote the custom Tandem Early Stopping algorithm because standard early stopping failed. I was using a dynamic learning rate (`ReduceLROnPlateau`) that takes smaller and smaller steps. Standard early stopping would ruthlessly kill the training because it couldn't beat the all-time high score fast enough. My shifting patience logic allowed the model to take tiny, slow steps out of complex local minima without being prematurely terminated, ultimately allowing me to find a much better final set of weights.
