from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'HeatDraft ML Pipeline: The "Why" and "How" Architecture', border=False, new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, num, title):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(220, 230, 255)
        self.cell(0, 8, f'{num}. {title}', 0, 1, 'L', fill=True)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('helvetica', '', 10)
        body = body.encode('ascii', 'ignore').decode('ascii')
        self.multi_cell(0, 6, body)
        self.ln(6)

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

content = [
    ("Why Sanitize Data & Handle Encodings (Lines 1-121)",
     "HOW IT WORKS: The script forces sys.stdout to UTF-8 and runs a dictionary replacement (`normalize_col_name`) sweeping through every column header, translating or deleting erratic unicode byte characters (e.g. converting Î³ to gamma).\n\n"
     "WHY WE DO THIS: Machine learning relies on dataframe dictionaries. In chemical research, scientists frequently copy/paste symbols from Word documents or proprietary tools into Excel. When Python attempts to access `df['MB surface energy, γm']`, strict OS-level Latin-1 decoders will aggressively crash the script if it encounters unmapped Greek symbols. Sanitizing the data at the exact moment of ingestion guarantees that automated ML pipelines can run unsupervised across hundreds of random CSVs without a human having to manually fix headers."),

    ("Why Integrate RDKit SMILES Descriptors (Lines 137-165)",
     "HOW IT WORKS: The `add_rdkit_descriptors` function hunts for a specific string column named 'SMILES'. If found, it routes that string sequence into a C++ chemistry engine (`Chem.MolFromSmiles`). It then executes eight unique topological graphing functions to map exactly how many rings, H-bonds, or fraction of Sp3 hybrid carbons exist in that literal molecule string.\n\n"
     "WHY WE DO THIS: An algorithm like XGBoost understands numbers, not strings. If you pass 'CCO' (Ethanol) to a Random Forest, it treats it as a random categorical ID, effectively useless. By forcing the integration of RDKit, we are directly extracting real-world physical and atomic properties out of a flat string. We are artificially expanding the model's intelligence by giving it the exact quantum and structural topography of the molecule rather than just its name."),

    ("Why Delete Strict Categories and Near-Zero Variance (Lines 168-177, 369-406)",
     "HOW IT WORKS: `drop_high_cardinality_categoricals` scans string columns. If a column has over 60 unique distinct text names, or if 95% of the values in the column are entirely unique (like a UUID or ID column), it dynamically drops it. `drop_near_zero_variance` handles numbers, dropping columns where standard deviation mathematically approaches 0.\n\n"
     "WHY WE DO THIS: This is strict defense against 'Memorization' (Overfitting). If we leave an 'Experiment ID' column in the dataset, a highly complex ExtraTrees regressor will realize that 'Experiment_412' always yields 99% efficiency. The model will literally memorize the ID rather than learning the actual chemistry. When exposed to future, unseen data ('Experiment_999'), the model will fail catastrophically because it learned the UUID, not the physics. Dropping near-zero numeric variance (e.g. an 'atmospheric pressure' column where every entry is 1.0 atm) simply removes useless noise that wastes computational training time."),

    ("Why we use Pearson, Mutual Information, AND VIF (Lines 180-315)",
     "HOW IT WORKS: This is a triple-layered defense system. First, Pearson correlation graphs every single feature against each other linearly. If Feature A and Feature B correlate at 0.95, they are mathematically identical. To choose which one to kill, it uses Non-Linear Mutual Information against the TARGET. Basically, it asks: 'Which of these two identical features actually helps predict our goal better?'. It keeps the winner. Finally, it uses Variance Inflation Factor (VIF), recursively analyzing the matrix to see if Feature D can be perfectly predicted by a combination of Features A + B + C. If VIF > 10, Feature D is killed.\n\n"
     "WHY WE DO THIS: This combats Multicollinearity. Machine learning algorithms (especially meta-estimators like Ridge applied in the stacking ensemble) rely on matrix inversion. If two columns are identical, matrix inversion mathematical stability shatters (causing weights to explode to infinity). Furthermore, if you feed a Random Forest 5 columns that all basically describe 'Temperature' in different metrics, the algorithm splits its attention 5 ways, severely weakening its ability to learn the true physical priority of Temperature. Stripping the dataset to completely mathematically unique (orthogonal) vectors forces the AI to learn efficiently."),

    ("Why use the Logit Transform and Dynamic Sample Weights (Lines 794-822)",
     "HOW IT WORKS: The dataset's target (Removal Rate) is bounded strictly between 0 and 100%. `target_transform` manually compresses this into a strict 0.0-1.0 probability scope and applies the formula `log(p / (1-p))`. Simultaneously, the script calculates how rare 'high-performing' (>90%) samples are. If they make up only 10% of the dataset, it creates a `weight_ratio` of 10.0x and attaches it to the high samples during the model's `reg.fit()` phase.\n\n"
     "WHY WE DO THIS: Standard ML regressors assume Target scales are infinite. Without a logit transform, a Random Forest could easily predict that a superb molecule has a '115% Removal Rate', which is physically impossible and ruins downstream analytics. The logit mathematically bans the model from exceeding the 0-100% boundary. As for sample weighting: ML models inherently try to minimize global average error. If 90% of your experiments fail, the model will become incredibly accurate at predicting failures and will ignore successes because successes are statistically irrelevant anomalies. By inflating the mathematical penalty of guessing a success wrong by 10x, we force the AI to hyper-focus on the physics required to succeed, which strictly aligns with real-world business value (we want to find good molecules, not exactly quantify bad ones)."),

    ("Why we use Optuna with K-Fold Cross Validation (Lines 436-529)",
     "HOW IT WORKS: The script establishes massive parameter boundary grids for XGBoost, Random Forest, ExtraTrees, etc. Optuna, a Bayesian Optimization engine, spins up an isolated 5-Fold Cross Validator (`KFold`). It guesses a random combination of architecture parameters (e.g. 1000 trees, 5 max depth). It slices the `X_train` matrix into 5 chunks, trains on 4, validates on 1, rotates 5 times, averages the Mean Absolute Error (MAE), and feeds that score back to Optuna. Optuna uses probabilistics to intelligently guess a better parameter layout on the next trial.\n\n"
     "WHY WE DO THIS: AI Architectures are highly sensitive completely blank slates out of the box. Default XGBoost parameters are essentially guessing. By forcing Optuna to execute thousands of K-Fold cross-validations, we are ensuring we locate the absolute mathematical limit of an algorithm's capability. We strictly use KFold inside the training loop so that hyperparameters are entirely blind to the global 20% holdout (`X_test`) set. If we tuned hyperparameters against the final test set, we would be mathematically 'cheating' and the ultimate global R2 score would be a fraudulent representation of real-world stability."),

    ("Why Build a RidgeCV Stacking Ensemble (Lines 871-903)",
     "HOW IT WORKS: The script isolates the top 3 best individual models (e.g. XGBoost, ExtraTrees, HistGB). It creates a `StackingRegressor`. In Phase 1, the three base models make their raw predictions. In Phase 2, a linear `RidgeCV` algorithm takes those 3 predictions, treats them as inputs, and calculates the exact optimal weighted blend to produce a final, single prediction.\n\n"
     "WHY WE DO THIS: Every algorithm has indigenous flaws. Forests struggle with extrapolation outside the train bounds. Gradient Boosting is heavily susceptible to severe localized outlier noise. By stacking them, we let a secondary Ridge algorithm figure out exactly how to correlate their strengths. If XGBoost is historically better at high-pressure physics but ExtraTrees is historically better at high-temperature physics, the Ridge mathematical blend inherently balances them seamlessly, achieving stability and accuracy that mathematically cannot be achieved by a single isolated framework."),

    ("Why Use a KMeans Mixture-of-Experts (MoE) (Lines 618-715)",
     "HOW IT WORKS: The core data is drastically dimensionally reduced via Principal Component Analysis (PCA). That simplified spatial map is routed to Unsupervised KMeans clustering which slices the dataset into 2 distinct physical zones. The generic StackingRegressor (our best meta-model) is explicitly duplicated. Clone A is trained exclusively on Zone 1 data. Clone B is trained exclusively on Zone 2 data. When a new molecule arrives during validation, KMeans acts as a dynamic 'Gate', instantly assessing the molecule's PCA coordinates and routing it strictly to the specific clone trained on that local physics.\n\n"
     "WHY WE DO THIS: In advanced material science, chemical behaviors are rarely universally linear. The physics governing a Low-Temperature, High-Pressure material reacts to catalysts completely differently than a High-Temperature, Low-Pressure material. If we attempt to train a single global ML model on both, it is forced to mathematically compromise, creating an 'average' rule curve that is ultimately mediocre at both. By executing a Mixture of Experts, we are acknowledging that heterogeneous physics exist. The gateway naturally organizes physical states into distinctly similar families, deploying highly specialized 'Expert' regressors capable of memorizing localized rules without interfering with each other.")
]

for i, (title, text) in enumerate(content, 1):
    pdf.chapter_title(i, title)
    pdf.chapter_body(text)

pdf.output("HeatDraft_Pipeline_Theory_Why_And_How.pdf")
print("Theory PDF generated.")
