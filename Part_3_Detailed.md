# Part 3: Data Cleansing, Dimensional Pruning, and Physical Boundary Enforcement

This document outlines the strict data cleansing rules, mathematical feature pruning, and output enforcement I built into the HeatDraft pipeline to prevent neural network hallucination.

---

## 1. Sparse Data Preservation (The pKa2 Indicator)

### What I did:
The `pKa2` (Secondary Acid Dissociation Constant) column was missing over 70% of its data because many chemicals do not have a secondary dissociation point. Instead of dropping the column entirely or trying to guess the missing values using KNN, I created a binary indicator feature (`has_pka2`). I then zero-filled the missing values.

### Why I did it:
Dropping the column would throw away valuable chemical information for the 30% of molecules that *do* have a secondary pKa. By creating a binary flag (1 if present, 0 if missing), I allowed the Neural Network to learn the structural difference between single-dissociation and double-dissociation compounds, turning sparse data into a powerful classification signal without forcing the imputer to guess impossible physics.

---

## 2. Removing Collinearity (Redundancy Pruning)

### What I did:
I programmed the pipeline to calculate a complete Pearson Correlation Matrix of all numerical features before they reach the model. I wrote a filter that automatically drops any feature that has a correlation of $> 0.90$ with another feature (excluding the target removal rate).

### Why I did it:
If two features are 95% correlated (for example, Molecule Weight and Molecule Size), they contain practically identical mathematical information. Feeding both into a neural network confuses the attention mechanism and causes the model to "double-count" the importance of that trait. By ruthlessly pruning highly collinear features, I vastly reduced the dimensionality of the embedding space, forcing the network to focus only on unique physical interactions and drastically speeding up training times.

---

## 3. Strict Experimental Condition Completeness

### What I did:
While I used RDKit, PubChem, and XGBoost to impute missing chemical properties, I placed a hard ban on imputing *experimental conditions* (like applied pressure, feed pH, or cross-flow velocity). I wrote a strict `.dropna()` rule that drops any row missing these core operational parameters.

### Why I did it:
Chemical properties are intrinsic and can be calculated deterministically, but laboratory conditions are arbitrary choices made by the human scientist. If a row was missing the applied pressure, imputing it with a "median" pressure of 1500 kPa would mean feeding the network a completely fabricated laboratory experiment. I made the hard executive decision to drop these rows entirely so that the neural network only trained on pristine, verifiable experimental conditions, ensuring no artificial laboratory data contaminated the brain.

---

## 4. Physical Boundary Clamping

### What I did:
In the final evaluation block of the PyTorch pipeline, I added a mathematical clamp using `np.clip(test_preds, 0.0, 100.0)` to constrain the final output array.

### Why I did it:
Neural networks operate on infinite floating-point domains. Occasionally, if presented with an extreme outlier (like a molecule 10x larger than the pore), a raw regression network might predict a removal rate of `105%`, or a highly permeable molecule might yield `-2%`. Because a physical membrane cannot reject more than 100% or less than 0% of a contaminant, I implemented an absolute physical clamp to guarantee that the final predictions obey the boundaries of reality before calculating the true MSE and R² scores.
