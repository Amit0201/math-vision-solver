# src/api/app.py

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import sys
sys.path.append('..')

from src.preprocessing.image_processor import MathImageProcessor
from src.recognition.model import MathSymbolCNN
from src.math_engine.solver import MathSolver
from src.math_engine.step_generator import StepByStepGenerator
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
#  Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="🧮 AI Math Solver",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Powered Handwritten Math Solver")
st.markdown("*Upload a handwritten equation or type one in!*")

# ──────────────────────────────────────────────
#  Initialize Components
# ──────────────────────────────────────────────

@st.cache_resource
def load_models():
    processor = MathImageProcessor()
    model = MathSymbolCNN(num_classes=25)
    # model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    solver = MathSolver()
    formatter = StepByStepGenerator()
    return processor, model, solver, formatter

processor, model, solver, formatter = load_models()

# ──────────────────────────────────────────────
#  Sidebar - Input Method Selection
# ──────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")
input_method = st.sidebar.radio(
    "Input Method",
    ["📷 Upload Image", "⌨️ Type Equation", "📹 Live Camera"]
)

problem_type = st.sidebar.selectbox(
    "Problem Type",
    ["Auto Detect", "Solve Equation", "Differentiate",
     "Integrate", "System of Equations", "Plot Function"]
)

# ──────────────────────────────────────────────
#  Main Content
# ──────────────────────────────────────────────

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📥 Input")

    if input_method == "📷 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload handwritten equation",
            type=['png', 'jpg', 'jpeg']
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Equation",
                     use_column_width=True)

            # Process image
            img_array = np.array(image)
            processed = processor.preprocess_from_array(img_array)
            st.image(processed, caption="Preprocessed",
                     use_column_width=True, clamp=True)

            # Recognition would happen here
            st.info("🔍 Recognized: `x^2 + 2*x - 8 = 0`")
            equation_str = "x^2 + 2*x - 8 = 0"  # Placeholder

    elif input_method == "⌨️ Type Equation":
        equation_str = st.text_input(
            "Enter equation:",
            value="x^2 + 2*x - 8 = 0",
            help="Examples: '2x + 5 = 13', 'x^2 - 4 = 0', 'sin(x) + x'"
        )

    elif input_method == "📹 Live Camera":
        camera_input = st.camera_input("Take a photo of the equation")
        equation_str = ""
        if camera_input:
            image = Image.open(camera_input)
            st.image(image, caption="Captured")
            equation_str = "x^2 + 2*x - 8 = 0"  # Placeholder

with col2:
    st.header("📤 Solution")

    if st.button("🚀 Solve!", type="primary", use_container_width=True):
        if equation_str:
            with st.spinner("Solving..."):
                try:
                    # ─── Route to appropriate solver ───
                    if problem_type in ["Solve Equation", "Auto Detect"]:
                        result = solver.solve_equation(equation_str)

                        st.success("✅ Solution Found!")

                        # Display steps
                        for step in result['steps']:
                            st.markdown(
                                f"**Step {step['step']}:** {step['description']}"
                            )
                            st.latex(step['expression'])

                        st.markdown("---")
                        st.markdown("### 🎯 Final Answer")
                        for sol in result.get('solutions_latex', []):
                            st.latex(f"x = {sol}")

                    elif problem_type == "Differentiate":
                        result = solver.differentiate(equation_str)
                        st.success("✅ Derivative Computed!")
                        for step in result['steps']:
                            st.markdown(
                                f"**Step {step['step']}:** "
                                f"{step['description']}"
                            )
                            st.latex(step['expression'])

                    elif problem_type == "Integrate":
                        result = solver.integrate_expr(equation_str)
                        st.success("✅ Integral Computed!")
                        for step in result['steps']:
                            st.markdown(
                                f"**Step {step['step']}:** "
                                f"{step['description']}"
                            )
                            st.latex(step['expression'])

                    elif problem_type == "Plot Function":
                        plot_data = solver.generate_plot_data(
                            equation_str, (-10, 10)
                        )
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(plot_data['x'], plot_data['y'],
                                'b-', linewidth=2)
                        ax.set_xlabel('x')
                        ax.set_ylabel('f(x)')
                        ax.set_title(f'f(x) = {equation_str}')
                        ax.grid(True, alpha=0.3)
                        ax.axhline(y=0, color='k', linewidth=0.5)
                        ax.axvline(x=0, color='k', linewidth=0.5)
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter an equation first!")

# ──────────────────────────────────────────────
#  Footer with examples
# ──────────────────────────────────────────────

st.markdown("---")
st.markdown("### 💡 Try These Examples")

examples_col1, examples_col2, examples_col3 = st.columns(3)

with examples_col1:
    st.markdown("**Algebra**")
    st.code("x^2 - 5*x + 6 = 0")
    st.code("3*x + 7 = 22")
    st.code("x^3 - 6*x^2 + 11*x - 6 = 0")

with examples_col2:
    st.markdown("**Calculus**")
    st.code("x^3 + 2*x^2 - x")
    st.code("sin(x) * cos(x)")
    st.code("exp(-x^2)")

with examples_col3:
    st.markdown("**Systems**")
    st.code("2x + 3y = 7, x - y = 1")
    st.code("x + y + z = 6")