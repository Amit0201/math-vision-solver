# src/utils/latex_converter.py

import re
from typing import List


class LatexConverter:
    """
    Converts between different math representations:
      - Raw symbol list  →  LaTeX string
      - Plain text       →  LaTeX string
      - LaTeX string     →  SymPy-parseable string
    """

    # Mapping: recognized symbol → LaTeX command
    SYMBOL_TO_LATEX = {
        '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
        '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
        '+': '+', '-': '-', '=': '=', '.': '.',
        '*': r'\times', '/': r'\div',
        '^': '^',
        '(': r'\left(', ')': r'\right)',
        'x': 'x', 'y': 'y', 'z': 'z',
        'sqrt': r'\sqrt',
        'pi': r'\pi',
        'frac': r'\frac',
    }

    # Mapping: LaTeX → SymPy-parseable
    LATEX_TO_SYMPY = {
        r'\times': '*',
        r'\div': '/',
        r'\pi': 'pi',
        r'\sqrt': 'sqrt',
        r'\left(': '(',
        r'\right)': ')',
        r'\frac': '/',  # Simplified — proper fractions need special handling
    }

    def symbols_to_latex(self, symbols: List[str]) -> str:
        """
        Convert a list of recognized symbols to a LaTeX string.

        Example:
            ['x', '^', '2', '+', '3', 'x', '-', '5', '=', '0']
            → 'x^{2} + 3x - 5 = 0'
        """
        latex_parts = []
        i = 0

        while i < len(symbols):
            sym = symbols[i]

            if sym == '^':
                # Look ahead for the exponent
                if i + 1 < len(symbols):
                    exponent = symbols[i + 1]
                    latex_parts.append(f'^{{{exponent}}}')
                    i += 2
                    continue

            elif sym == 'sqrt':
                # Look ahead for content inside sqrt
                if i + 1 < len(symbols):
                    # Collect until closing or next operator
                    content = self._collect_group(symbols, i + 1)
                    latex_parts.append(f'\\sqrt{{{content}}}')
                    i += 1 + len(content.replace(' ', ''))
                    continue

            elif sym == 'frac':
                # Simplified: next two groups are numerator/denominator
                if i + 2 < len(symbols):
                    num = symbols[i + 1]
                    den = symbols[i + 2]
                    latex_parts.append(f'\\frac{{{num}}}{{{den}}}')
                    i += 3
                    continue

            # Default: direct mapping
            latex_sym = self.SYMBOL_TO_LATEX.get(sym, sym)
            latex_parts.append(latex_sym)
            i += 1

        return ' '.join(latex_parts)

    def symbols_to_text(self, symbols: List[str]) -> str:
        """
        Convert recognized symbols to plain text equation.

        Example:
            ['x', '^', '2', '+', '3', 'x', '-', '5', '=', '0']
            → 'x^2 + 3*x - 5 = 0'
        """
        text_parts = []
        prev_sym = None

        for i, sym in enumerate(symbols):
            # Add implicit multiplication
            if prev_sym and self._needs_multiplication(prev_sym, sym):
                text_parts.append('*')

            if sym == '^':
                text_parts.append('**')
            elif sym == 'sqrt':
                text_parts.append('sqrt(')
                # Note: need to track when to close parenthesis
            elif sym == 'pi':
                text_parts.append('pi')
            else:
                text_parts.append(sym)

            prev_sym = sym

        return ''.join(text_parts)

    def text_to_latex(self, text: str) -> str:
        """
        Convert plain text equation to LaTeX.

        Example:
            'x**2 + 3*x - 5 = 0'  →  'x^{2} + 3x - 5 = 0'
        """
        latex = text

        # Replace ** with ^{}
        latex = re.sub(r'\*\*(\d+)', r'^{\1}', latex)
        latex = re.sub(r'\*\*\(([^)]+)\)', r'^{\1}', latex)

        # Replace * with implicit multiplication (remove it)
        latex = re.sub(r'(\d)\*([a-z])', r'\1\2', latex)
        # Keep * between variables as \cdot
        latex = re.sub(r'([a-z])\*([a-z])', r'\1 \cdot \2', latex)

        # Replace sqrt()
        latex = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', latex)

        # Replace pi
        latex = latex.replace('pi', r'\pi')

        return latex

    def latex_to_sympy(self, latex_str: str) -> str:
        """
        Convert LaTeX string to SymPy-parseable string.

        Example:
            'x^{2} + 3x - 5 = 0'  →  'x**2 + 3*x - 5'
        """
        sympy_str = latex_str

        # Remove LaTeX commands
        sympy_str = sympy_str.replace(r'\left', '')
        sympy_str = sympy_str.replace(r'\right', '')
        sympy_str = sympy_str.replace(r'\cdot', '*')
        sympy_str = sympy_str.replace(r'\times', '*')
        sympy_str = sympy_str.replace(r'\div', '/')
        sympy_str = sympy_str.replace(r'\pi', 'pi')

        # Handle powers: ^{n} → **n
        sympy_str = re.sub(r'\^{([^}]+)}', r'**(\1)', sympy_str)
        sympy_str = re.sub(r'\^(\d)', r'**\1', sympy_str)

        # Handle sqrt
        sympy_str = re.sub(r'\\sqrt{([^}]+)}', r'sqrt(\1)', sympy_str)

        # Handle fractions: \frac{a}{b} → (a)/(b)
        sympy_str = re.sub(
            r'\\frac{([^}]+)}{([^}]+)}',
            r'(\1)/(\2)',
            sympy_str
        )

        # Add implicit multiplication: 3x → 3*x
        sympy_str = re.sub(r'(\d)([a-z])', r'\1*\2', sympy_str)

        # Clean up spaces
        sympy_str = sympy_str.strip()

        return sympy_str

    def _needs_multiplication(self, prev: str, curr: str) -> bool:
        """Check if implicit multiplication is needed between two symbols."""
        # Cases: "2x" → "2*x", "x(" → "x*(", ")x" → ")*x"
        prev_is_num = prev.isdigit() or prev == '.'
        prev_is_var = prev in ('x', 'y', 'z', 'pi')
        prev_is_close = prev == ')'

        curr_is_var = curr in ('x', 'y', 'z', 'pi')
        curr_is_open = curr == '('
        curr_is_func = curr in ('sqrt',)

        if (prev_is_num or prev_is_var or prev_is_close) and \
           (curr_is_var or curr_is_open or curr_is_func):
            return True

        return False

    def _collect_group(self, symbols: List[str],
                       start: int) -> str:
        """Collect symbols until an operator or end."""
        operators = {'+', '-', '='}
        parts = []
        i = start
        while i < len(symbols) and symbols[i] not in operators:
            parts.append(symbols[i])
            i += 1
        return ''.join(parts)