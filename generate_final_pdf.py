from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'HeatDraft ML Pipeline: Project Defense Report', 0, 1, 'C')
        
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 51, 153)
        self.ln(5)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(2)
        
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body.encode('latin-1', 'replace').decode('latin-1'))
        self.ln(5)

pdf = PDF()
pdf.add_page()

# PART 1
pdf.chapter_title('Part 1: Data Acquisition & EDA')
pdf.chapter_body("What I did:\nI built an automated module that generates statistical profiles, missing value heatmaps, and correlation matrices. Furthermore, I implemented Principal Component Analysis (PCA) and t-SNE to compress the chemical features into a 2D plane, grouping them using K-Means clustering (n=4).")
pdf.chapter_body("Why I did it:\nI needed to mathematically verify the skewness of the raw chemical properties before feeding them to the neural network. I implemented PCA and t-SNE to visually prove that distinct structural clusters of membrane-contaminant interactions actually exist in the data.")
try:
    pdf.image('visualizations/1_missing_values.png', w=170)
    pdf.ln(5)
    pdf.image('visualizations/4_clusters_and_dim_reduction.png', w=170)
except:
    pass

pdf.add_page()
# PART 2
pdf.chapter_title('Part 2: Deterministic Chemical Imputation')
pdf.chapter_body("What I did:\nInstead of using standard data science techniques (like mean filling), I integrated RDKit to mathematically calculate intrinsic properties (MolWt, MolLogP) directly from SMILES strings. For properties that RDKit could not calculate locally, I built a dynamic web-scraper that connects to the NIH PubChem REST API to extract empirical pKa values.")
pdf.chapter_body("Why I did it:\nStandard statistical imputation is scientifically invalid for hard chemistry. A molecule's weight is a fixed physical constant, not an average of its neighbors in an Excel sheet. By falling back to the NIH PubChem database, I ensured the neural network was learning from true, peer-reviewed empirical laboratory values.")

# PART 3
pdf.chapter_title('Part 3: The Tabular Attention Neural Network')
pdf.chapter_body("What I did:\nInstead of using a standard Multi-Layer Perceptron (MLP), I designed and implemented a custom TabularAttentionNet in PyTorch. I programmed it to project each scalar feature into a high-dimensional vector space, treating every spreadsheet column like a 'token'. I then passed these tokens through an nn.TransformerEncoderLayer featuring a Query-Key-Value (QKV) self-attention mechanism.")
pdf.chapter_body("Why I did it:\nStandard dense neural networks mash all columns together indiscriminately in the first layer. I chose a Transformer-based Attention mechanism because it allows the network to dynamically calculate 'Attention Weights' between different columns, simulating complex physics (like electrostatic repulsion) dynamically rather than relying on static weights.")

pdf.add_page()
# PART 4
pdf.chapter_title('Part 4: Final Evaluation and Post-Training Diagnostics')
pdf.chapter_body("What I did:\nI evaluated the final neural network weights on the completely unseen test set. I wrote a custom Permutation Feature Importance script to mathematically shuffle physics columns and measure the MSE explosion. I also built an Erratic Error Analysis visualization that plots Absolute Prediction Error against key physical constraints to identify boundary failures.")

try:
    pdf.image('visualizations/5_predicted_vs_actual.png', w=150)
    pdf.ln(5)
    pdf.image('visualizations/6_feature_importance.png', w=150)
    pdf.ln(5)
    pdf.image('visualizations/7_error_analysis.png', w=150)
except:
    pass

pdf.output('HeatDraft_Project_Defense_Report.pdf')
