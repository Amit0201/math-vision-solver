# src/math_engine/solver.py

import sympy as sp
from sympy import (
    symbols, solve, diff, integrate, simplify,
    factor, expand, limit, series, Matrix,
    sin, cos, tan, log, exp, sqrt, pi, oo,
    Eq, latex, Rational
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
from typing import List, Dict, Any, Optional


class MathSolver:
    """
    Comprehensive math solver supporting:
    - Arithmetic & Algebra
    - Equations (linear, quadratic, systems)
    - Calculus (derivatives, integrals, limits)
    - Matrix operations
    - Step-by-step solutions
    """

    TRANSFORMATIONS = (
            standard_transformations +
            (implicit_multiplication_application, convert_xor)
    )

    def __init__(self):
        self.x, self.y, self.z = symbols('x y z')
        self.t, self.n = symbols('t n')
        self.history: List[Dict] = []

    # ─── Expression Parsing ──────────────────

    def parse(self, expr_str: str) -> sp.Expr:
        """
        Parse string to SymPy expression.
        Handles: '2x^2 + 3x - 5', 'sqrt(x)', 'x/y + 1'
        """
        cleaned = expr_str.replace('^', '**')
        cleaned = cleaned.replace('÷', '/')
        cleaned = cleaned.replace('×', '*')

        return parse_expr(
            cleaned,
            transformations=self.TRANSFORMATIONS,
            local_dict={
                'x': self.x, 'y': self.y, 'z': self.z,
                'pi': pi, 'e': sp.E
            }
        )

    # ─── Equation Solving ────────────────────

    def solve_equation(self, equation_str: str) -> Dict[str, Any]:
        """
        Solve equations with step-by-step explanations.

        Examples:
            '2*x + 5 = 13'
            'x^2 - 4 = 0'
            'x^2 + 2*x - 8 = 0'
        """
        steps = []

        if '=' in equation_str:
            lhs_str, rhs_str = equation_str.split('=')
            lhs = self.parse(lhs_str.strip())
            rhs = self.parse(rhs_str.strip())
            equation = Eq(lhs, rhs)
            expr = lhs - rhs  # Move everything to one side
        else:
            expr = self.parse(equation_str)
            equation = Eq(expr, 0)

        steps.append({
            'step': 1,
            'description': 'Original equation',
            'expression': latex(equation),
        })

        # Simplify
        simplified = simplify(expr)
        steps.append({
            'step': 2,
            'description': 'Rearrange to standard form',
            'expression': f"{latex(simplified)} = 0",
        })

        # Detect equation type
        degree = sp.degree(simplified, self.x)
        var = self.x

        if degree == 1:
            steps = self._solve_linear_steps(simplified, steps)
        elif degree == 2:
            steps = self._solve_quadratic_steps(simplified, steps)

        # Final solution
        solutions = solve(expr, var)
        steps.append({
            'step': len(steps) + 1,
            'description': 'Solution',
            'expression': f"{var} = {', '.join(latex(s) for s in solutions)}",
        })

        result = {
            'equation': equation_str,
            'solutions': [str(s) for s in solutions],
            'solutions_latex': [latex(s) for s in solutions],
            'steps': steps,
            'equation_type': f"degree-{degree} polynomial",
        }

        self.history.append(result)
        return result

    def _solve_linear_steps(self, expr, steps):
        """Step-by-step for ax + b = 0."""
        a = expr.coeff(self.x, 1)
        b = expr.coeff(self.x, 0)

        steps.append({
            'step': len(steps) + 1,
            'description': 'Identify coefficients',
            'expression': f"a = {latex(a)}, \\quad b = {latex(b)}",
        })
        steps.append({
            'step': len(steps) + 1,
            'description': 'Apply formula: x = -b/a',
            'expression': f"x = \\frac{{-({latex(b)})}}{{{latex(a)}}} = {latex(-b / a)}",
        })
        return steps

    def _solve_quadratic_steps(self, expr, steps):
        """
        Step-by-step for ax² + bx + c = 0
        Uses the Quadratic Formula:
            x = (-b ± √(b²-4ac)) / 2a
        """
        a = expr.coeff(self.x, 2)
        b = expr.coeff(self.x, 1)
        c = expr.coeff(self.x, 0)

        discriminant = b ** 2 - 4 * a * c

        steps.append({
            'step': len(steps) + 1,
            'description': 'Identify coefficients: ax² + bx + c = 0',
            'expression': f"a = {latex(a)}, \\quad b = {latex(b)}, \\quad c = {latex(c)}",
        })
        steps.append({
            'step': len(steps) + 1,
            'description': 'Calculate discriminant: Δ = b² - 4ac',
            'expression': f"\\Delta = ({latex(b)})^2 - 4({latex(a)})({latex(c)}) = {latex(discriminant)}",
        })

        if discriminant > 0:
            steps.append({
                'step': len(steps) + 1,
                'description': 'Δ > 0 → Two real distinct roots',
                'expression': f"x = \\frac{{-b \\pm \\sqrt{{\\Delta}}}}{{2a}}",
            })
        elif discriminant == 0:
            steps.append({
                'step': len(steps) + 1,
                'description': 'Δ = 0 → One repeated real root',
                'expression': f"x = \\frac{{-b}}{{2a}} = {latex(-b / (2 * a))}",
            })
        else:
            steps.append({
                'step': len(steps) + 1,
                'description': 'Δ < 0 → Two complex conjugate roots',
                'expression': f"x = \\frac{{-b \\pm i\\sqrt{{|\\Delta|}}}}{{2a}}",
            })

        return steps

    # ─── Calculus ────────────────────────────

    def differentiate(self, expr_str: str,
                      var: str = 'x', order: int = 1) -> Dict:
        """
        Compute derivatives with steps.

        Math Rules Applied:
          - Power Rule: d/dx[x^n] = n·x^(n-1)
          - Chain Rule: d/dx[f(g(x))] = f'(g(x))·g'(x)
          - Product Rule: d/dx[f·g] = f'g + fg'
        """
        expr = self.parse(expr_str)
        variable = symbols(var)

        steps = [{
            'step': 1,
            'description': f'Differentiate with respect to {var}',
            'expression': f"\\frac{{d}}{{d{var}}}\\left[{latex(expr)}\\right]",
        }]

        # Step-by-step differentiation
        result = expr
        for i in range(order):
            result = diff(result, variable)
            simplified = simplify(result)
            steps.append({
                'step': len(steps) + 1,
                'description': f'Apply differentiation rules (order {i + 1})',
                'expression': latex(simplified),
            })

        return {
            'original': expr_str,
            'derivative': str(result),
            'derivative_latex': latex(simplify(result)),
            'steps': steps,
        }

    def integrate_expr(self, expr_str: str,
                       var: str = 'x',
                       bounds: Optional[tuple] = None) -> Dict:
        """
        Compute integrals (definite and indefinite).

        Math:
          ∫ f(x)dx (indefinite)
          ∫_a^b f(x)dx = F(b) - F(a) (definite)
        """
        expr = self.parse(expr_str)
        variable = symbols(var)

        steps = []

        if bounds:
            a, b = bounds
            steps.append({
                'step': 1,
                'description': f'Compute definite integral from {a} to {b}',
                'expression': f"\\int_{{{a}}}^{{{b}}} {latex(expr)} \\, d{var}",
            })

            antiderivative = integrate(expr, variable)
            steps.append({
                'step': 2,
                'description': 'Find antiderivative F(x)',
                'expression': f"F({var}) = {latex(antiderivative)}",
            })

            result = integrate(expr, (variable, a, b))
            steps.append({
                'step': 3,
                'description': 'Apply Fundamental Theorem: F(b) - F(a)',
                'expression': f"F({b}) - F({a}) = {latex(result)}",
            })
        else:
            steps.append({
                'step': 1,
                'description': 'Compute indefinite integral',
                'expression': f"\\int {latex(expr)} \\, d{var}",
            })

            result = integrate(expr, variable)
            steps.append({
                'step': 2,
                'description': 'Result (+ constant of integration)',
                'expression': f"{latex(result)} + C",
            })

        return {
            'original': expr_str,
            'result': str(result),
            'result_latex': latex(result),
            'steps': steps,
        }

    # ─── Linear Algebra ─────────────────────

    def solve_system(self, equations: List[str]) -> Dict:
        """
        Solve systems of linear equations.

        Example: ['2x + 3y = 7', 'x - y = 1']

        Uses Gaussian elimination / matrix methods:
            Ax = b  →  x = A⁻¹b
        """
        eq_objects = []
        all_vars = set()

        for eq_str in equations:
            lhs_str, rhs_str = eq_str.split('=')
            lhs = self.parse(lhs_str.strip())
            rhs = self.parse(rhs_str.strip())
            eq_objects.append(Eq(lhs, rhs))
            all_vars.update(lhs.free_symbols | rhs.free_symbols)

        all_vars = sorted(all_vars, key=str)
        solutions = solve(eq_objects, all_vars)

        steps = [{
            'step': 1,
            'description': 'System of equations',
            'expression': ' \\\\ '.join(latex(eq) for eq in eq_objects),
        }]

        # Build coefficient matrix for display
        if isinstance(solutions, dict):
            steps.append({
                'step': 2,
                'description': 'Solution by substitution / elimination',
                'expression': ', \\quad '.join(
                    f"{latex(var)} = {latex(val)}"
                    for var, val in solutions.items()
                ),
            })

        return {
            'equations': equations,
            'solutions': {str(k): str(v) for k, v in solutions.items()}
            if isinstance(solutions, dict) else str(solutions),
            'steps': steps,
        }

    # ─── Graph Data Generation ───────────────

    def generate_plot_data(self, expr_str: str,
                           x_range: tuple = (-10, 10),
                           num_points: int = 500) -> Dict:
        """Generate data points for plotting the function."""
        import numpy as np

        expr = self.parse(expr_str)
        func = sp.lambdify(self.x, expr, 'numpy')

        x_vals = np.linspace(x_range[0], x_range[1], num_points)

        try:
            y_vals = func(x_vals)
            # Handle infinities
            y_vals = np.where(np.abs(y_vals) > 1000, np.nan, y_vals)
        except Exception:
            y_vals = np.array([complex(expr.subs(self.x, xi)).real
                               for xi in x_vals])

        return {
            'x': x_vals.tolist(),
            'y': y_vals.tolist(),
            'expression': expr_str,
            'latex': latex(expr),
        }