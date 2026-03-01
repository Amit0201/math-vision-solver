# src/math_engine/step_generator.py

from typing import List, Dict
import json


class StepByStepGenerator:
    """
    Formats step-by-step solutions into
    human-readable explanations.
    """

    MATH_RULES = {
        'power_rule': {
            'name': 'Power Rule',
            'formula': 'd/dx[xⁿ] = n·xⁿ⁻¹',
            'description': 'Bring down the exponent, reduce power by 1'
        },
        'chain_rule': {
            'name': 'Chain Rule',
            'formula': 'd/dx[f(g(x))] = f\'(g(x))·g\'(x)',
            'description': 'Derivative of outer × derivative of inner'
        },
        'quadratic_formula': {
            'name': 'Quadratic Formula',
            'formula': 'x = (-b ± √(b²-4ac)) / 2a',
            'description': 'Standard formula for ax² + bx + c = 0'
        },
        'integration_by_parts': {
            'name': 'Integration by Parts',
            'formula': '∫u dv = uv - ∫v du',
            'description': 'Used when integrand is a product'
        },
    }

    def format_solution(self, result: Dict) -> str:
        """Format a solution dict into readable text."""
        output = []
        output.append("=" * 50)
        output.append(f"📝 Problem: {result.get('equation', result.get('original', ''))}")
        output.append("=" * 50)

        for step in result.get('steps', []):
            output.append(
                f"\n  Step {step['step']}: {step['description']}"
            )
            output.append(f"    → {step['expression']}")

        output.append("\n" + "─" * 50)

        if 'solutions' in result:
            output.append(f"✅ Answer: {result['solutions']}")
        elif 'result' in result:
            output.append(f"✅ Answer: {result['result']}")

        output.append("=" * 50)
        return '\n'.join(output)

    def to_html(self, result: Dict) -> str:
        """Convert solution to HTML with MathJax rendering."""
        html = f"""
        <div class="solution-container">
            <h3>Problem</h3>
            <p class="math">$${result.get('equation', '')}$$</p>

            <h3>Step-by-Step Solution</h3>
            <ol>
        """
        for step in result.get('steps', []):
            html += f"""
                <li>
                    <strong>{step['description']}</strong><br>
                    <span class="math">$${step['expression']}$$</span>
                </li>
            """
        html += """
            </ol>
            <div class="answer">
                <h3>✅ Final Answer</h3>
        """

        if 'solutions_latex' in result:
            for sol in result['solutions_latex']:
                html += f'<p class="math">$${sol}$$</p>'

        html += "</div></div>"
        return html