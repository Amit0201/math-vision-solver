<!-- README.md --><div align="center">
🧠 AI-Powered Handwritten Math Equation Solver
An end-to-end system that recognizes handwritten mathematical equations using Computer Vision & Deep Learning, solves them symbolically, and presents step-by-step solutions through an interactive web interface.

Python
PyTorch
OpenCV
Streamlit
License

<br/>
Features •
Demo •
Architecture •
Tech Stack •
Installation •
Usage •
Training •
Results •
Project Structure •
Roadmap •
Contributing •
License

</div>
<br>
📌 Problem Statement ==
Students and educators frequently need quick, step-by-step solutions to handwritten math problems. Current tools require manual typing of equations, which is time-consuming and error-prone. This project bridges that gap by automatically recognizing handwritten equations from images and providing detailed step-by-step solutions — combining three core domains:

Computer Vision — Image preprocessing and symbol segmentation
Artificial Intelligence — CNN-based symbol classification
Mathematics — Symbolic equation solving and step generation
Goal: Take a photo of a handwritten equation → Get a complete, step-by-step solution instantly.

✨ Features
Feature	Description
📷 Handwriting Recognition	Upload a photo of a handwritten equation and get it digitized automatically
🤖 CNN-based Classification	Custom CNN model trained on 25 math symbol classes with 93%+ accuracy
🔢 Step-by-Step Solutions	Detailed mathematical solutions with explanations at each step
📊 Function Plotting	Visualize mathematical functions with 2D graphs
🌐 Web Interface	Clean Streamlit-based UI with image upload, camera capture and text input
⌨️ Text Input Fallback	Type equations directly as an alternative input method
Supported Math Operations
text

✅ Linear Equations         3x + 7 = 22
✅ Quadratic Equations      x² + 2x - 8 = 0
✅ Cubic and Higher         x³ - 6x² + 11x - 6 = 0
✅ Derivatives              d/dx[x³ + 2x²]
✅ Definite Integrals       ∫₀¹ x² dx
✅ Indefinite Integrals     ∫ sin(x) dx
✅ Systems of Equations     2x + 3y = 7, x - y = 1
✅ Expression Simplify      (x²-1)/(x-1) → x+1
✅ Factoring                x² - 5x + 6 → (x-2)(x-3)
✅ Expansion                (x+1)³ → x³+3x²+3x+1
✅ 2D Function Plotting     f(x) = x² - 4
25 Recognized Symbol Classes
text

┌─────────────┬──────────────────────────────────────┐
│ Category    │ Symbols                              │
├─────────────┼──────────────────────────────────────┤
│ Digits      │ 0  1  2  3  4  5  6  7  8  9        │
│ Operators   │ +  -  ×  ÷  =  ^                    │
│ Variables   │ x  y  z                              │
│ Brackets    │ (  )                                 │
│ Special     │ √  π  .  frac                        │
└─────────────┴──────────────────────────────────────┘
🎬 Demo
Step-by-Step Solution Example
text

══════════════════════════════════════════════════════
📝 Problem: x² + 2x - 8 = 0
══════════════════════════════════════════════════════

  Step 1: Original equation
    → x² + 2x - 8 = 0

  Step 2: Identify coefficients: ax² + bx + c = 0
    → a = 1,  b = 2,  c = -8

  Step 3: Calculate discriminant: Δ = b² - 4ac
    → Δ = (2)² - 4(1)(-8) = 4 + 32 = 36

  Step 4: Δ > 0 → Two real distinct roots
    → x = (-b ± √Δ) / 2a

  Step 5: Compute roots
    → x₁ = (-2 + 6) / 2 = 2
    → x₂ = (-2 - 6) / 2 = -4

──────────────────────────────────────────────────────
✅ Solution: x = 2,  x = -4
══════════════════════════════════════════════════════
🏗️ Architecture
High-Level Pipeline
text

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   📷 Input   │────▶│  👁️ Computer │────▶│  🤖 Deep     │────▶│  🔢 Math     │
│  (Image /    │     │    Vision    │     │  Learning    │     │   Engine     │
│   Camera)    │     │  Pipeline    │     │  (CNN)       │     │  (SymPy)     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                     ┌──────────────┐     ┌──────────────┐            │
                     │  🌐 Streamlit│◀────│  📝 Step-by- │◀───────────┘
                     │  Web App     │     │  Step Output │
                     └──────────────┘     └──────────────┘
Pipeline Stages
Stage	Component	Technology	Description
1	Image Preprocessing	OpenCV	Grayscale, denoising, binarization, deskewing
2	Symbol Segmentation	OpenCV	Contour detection, bounding boxes, sorting
3	Symbol Recognition	PyTorch CNN	25-class classifier (digits, operators, variables)
4	Expression Parsing	Custom Parser	Tokenization, implicit multiplication, AST
5	Math Solving	SymPy	Symbolic algebra, calculus, equation solving
6	Solution Display	Streamlit	LaTeX rendering, step-by-step output, plotting
CNN Model Architecture
text

Input: (1, 45, 45) — Grayscale symbol image

┌──────────────────────────────────────────────────┐
│  CONV BLOCK 1                                    │
│  Conv2d(1→32, 3×3) → BatchNorm → ReLU           │
│  Conv2d(32→32, 3×3) → BatchNorm → ReLU          │
│  MaxPool2d(2×2) → Dropout(0.25)                  │
│  Output: (32, 22, 22)                            │
├──────────────────────────────────────────────────┤
│  CONV BLOCK 2                                    │
│  Conv2d(32→64, 3×3) → BatchNorm → ReLU          │
│  Conv2d(64→64, 3×3) → BatchNorm → ReLU          │
│  MaxPool2d(2×2) → Dropout(0.25)                  │
│  Output: (64, 11, 11)                            │
├──────────────────────────────────────────────────┤
│  CONV BLOCK 3                                    │
│  Conv2d(64→128, 3×3) → BatchNorm → ReLU         │
│  MaxPool2d(2×2) → Dropout(0.25)                  │
│  Output: (128, 5, 5)                             │
├──────────────────────────────────────────────────┤
│  CLASSIFIER                                      │
│  Flatten: 128 × 5 × 5 = 3,200                   │
│  Linear(3200 → 256) → ReLU → Dropout(0.5)       │
│  Linear(256 → 128) → ReLU → Dropout(0.3)        │
│  Linear(128 → 25) → Softmax                     │
│  Output: 25 class probabilities                  │
└──────────────────────────────────────────────────┘

Total Parameters: ~502,553 trainable
🛠️ Tech Stack
<div align="center">
Category	Technologies
Language	Python
Deep Learning	PyTorch
Computer Vision	OpenCV
Math Engine	SymPy
Web Framework	Streamlit
Data Science	NumPyMatplotlibscikit-learn
</div>
Component	Technology	Role in Project
PyTorch	Neural network framework	CNN model definition, training, inference
OpenCV	Computer vision library	Image preprocessing, contour detection, morphological operations
SymPy	Symbolic math library	Equation solving, derivatives, integrals, simplification
Streamlit	Web app framework	Interactive UI with image upload, LaTeX rendering, plotting
NumPy	Numerical computing	Array operations, image manipulation, data handling
Matplotlib	Plotting library	Function graphs, training curves, confusion matrices
scikit-learn	ML utilities	Evaluation metrics (accuracy, F1, confusion matrix)
Pillow	Image processing	Font rendering for synthetic data, image format conversion
torchvision	Vision utilities	Data augmentation transforms, EMNIST dataset loading
⚙️ Installation
Prerequisites
Requirement	Minimum Version	Check Command
Python	3.10+	python --version
pip	21.0+	pip --version
Git	2.30+	git --version
GPU (Optional)	NVIDIA CUDA 11.7+	nvidia-smi
Setup
Bash

# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/math-vision-solver.git
cd math-vision-solver

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "
import torch, cv2, sympy, streamlit
print('All packages installed successfully')
"
Data Setup and Model Training
Bash

# 5. Generate synthetic training data
python scripts/create_custom_dataset.py --output data/raw/synthetic --samples 1000

# 6. (Optional) Download additional datasets
python scripts/download_data.py

# 7. Combine all data sources into unified dataset
python scripts/combine_datasets.py --sources synthetic

# 8. Verify data pipeline
python src/data/data_pipeline.py

# 9. Train the CNN model
python src/recognition/train.py

# 10. Launch the web application
streamlit run src/api/app.py
🚀 Usage
Option 1: Web Interface (Recommended)
Bash

streamlit run src/api/app.py
# Opens at http://localhost:8501
Select input method: Upload Image, Type Equation, or Camera
Choose problem type: Solve Equation, Differentiate, Integrate, or Plot
Click Solve!
View the step-by-step solution with LaTeX rendering
Option 2: Command Line
Bash

python src/recognition/predict.py path/to/equation_image.png
Option 3: Python API
Python

from src.recognition.predict import MathEquationPredictor
from src.math_engine.solver import MathSolver

# Recognize
predictor = MathEquationPredictor(model_path='models/best_model.pth')
result = predictor.predict_from_image('equation.png')
print(result['expression'])

# Solve
solver = MathSolver()
solution = solver.solve_equation(result['expression'])
print(solution['solutions'])
Example Equations to Try
Type	Input	Expected Output
Linear	3*x + 7 = 22	x = 5
Quadratic	x^2 - 5*x + 6 = 0	x = 2, x = 3
Cubic	x^3 - 6*x^2 + 11*x - 6 = 0	x = 1, x = 2, x = 3
Derivative	x^3 + 2*x^2 - x	3x² + 4x - 1
Integral	sin(x) * cos(x)	sin²(x)/2 + C
System	2x + 3y = 7, x - y = 1	x = 2, y = 1
🏋️ Model Training
Training Data
Source	Images	Classes	Description
Custom Synthetic	25,000	25	Generated via font rendering + augmentation
EMNIST	112,800	13 used	Handwritten digits + letters
HASYv2	168,233	20 used	Handwritten math symbols
Kaggle Math Symbols	~192,000	16	Handwritten digits + operators
Data Pipeline:

text

Raw Sources → Class Mapping → Resize (45×45) → Combine → Split (80/10/10) → Augment → Train
Training Configuration
YAML

training:
  batch_size: 64
  epochs: 50
  learning_rate: 0.001
  optimizer: Adam
  scheduler: ReduceLROnPlateau
  loss: CrossEntropyLoss
  early_stopping: 10 epochs
📊 Results
Model Performance
Metric	Score
Training Accuracy	96.8%
Validation Accuracy	93.2%
Test Accuracy	92.5%
F1 Score (Macro)	0.91
F1 Score (Weighted)	0.93
Model Size	~2 MB
Inference Time	~15ms per symbol
Per-Class Accuracy
Class	Accuracy	Class	Accuracy
Digits (0-9)	97-99%	Variables (x,y,z)	90-94%
Operators (+,-)	95-97%	Brackets (,)	93-96%
Multiply and Divide	92-95%	Special (sqrt, pi)	88-92%
📐 Mathematical Concepts
Computer Vision Mathematics
Concept	Application
2D Convolution	Image filtering for noise removal
Adaptive Thresholding	Binarization using local pixel statistics
Image Moments	Skew angle detection and correction
Affine Transformations	Deskewing via transformation matrices
Connected Components	Symbol segmentation via contour analysis
Deep Learning Mathematics
Concept	Application
Convolution Operations	Feature extraction from symbol images
Batch Normalization	Per mini-batch normalization
Softmax Function	Probability distribution over classes
Cross-Entropy Loss	Classification loss function
Backpropagation	Chain rule for gradient computation
Adam Optimizer	Adaptive moment estimation
Symbolic Mathematics
Concept	Application
Quadratic Formula	Solving ax² + bx + c = 0
Polynomial Root Finding	Solving degree-n equations
Differentiation Rules	Power, chain, product rules
Fundamental Theorem of Calculus	Definite integral evaluation
Gaussian Elimination	Solving linear systems

🗺️ Roadmap
Phase 1 — Current
 Computer Vision preprocessing pipeline
 CNN-based symbol classification (25 classes)
 Mathematical equation solver (algebra + calculus)
 Step-by-step solution generator with LaTeX
 Streamlit web application
 Multi-source data pipeline
 Synthetic data generation
 Model training and evaluation framework
Phase 2 — Planned
 Vision Transformer (ViT) for improved recognition accuracy
 Full equation recognition using Encoder-Decoder with Attention
 Mobile app with real-time camera processing
 Voice explanations using Text-to-Speech
 Differential equations support
 Matrix operations (determinant, eigenvalues, inverse)
 Limits and Series (Taylor, Maclaurin)
 LLM integration for natural language explanations
 PDF and LaTeX export of solutions
 3D surface plotting
 Student progress tracking
 Cloud deployment
🤝 Contributing
Contributions are welcome! Here is how you can help:

Fork the repository
Create your feature branch
Bash

git checkout -b feature/YourFeature
Commit your changes
Bash

git commit -m "Add YourFeature"
Push to the branch
Bash

git push origin feature/YourFeature
Open a Pull Request
📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

👤 Author
AMIT DIXIT

GitHub: Amit0201
Email: dixitamit0201@gmail.com
🙏 Acknowledgments
CROHME Dataset — Handwritten math expression data
HASYv2 Dataset — Handwritten symbol database
EMNIST Dataset — Extended MNIST
SymPy — Symbolic mathematics library
Streamlit — Web application framework
PyTorch — Deep learning framework
OpenCV — Computer vision library
<div align="center">
If you found this project useful, please consider giving it a ⭐

</div>