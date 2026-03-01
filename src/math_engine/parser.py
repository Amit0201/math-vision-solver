# src/math_engine/parser.py

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Token:
    """Represents a single token in the equation."""
    type: str       # 'NUMBER', 'VARIABLE', 'OPERATOR', 'FUNCTION', 'PAREN'
    value: str      # The actual string value
    position: int   # Position in the original sequence


@dataclass
class ASTNode:
    """Abstract Syntax Tree node for math expressions."""
    type: str                              # 'binary_op', 'unary_op', 'number', 'variable', 'function'
    value: str                             # The operator, number, or variable
    children: List['ASTNode'] = field(default_factory=list)

    def __repr__(self):
        if self.children:
            children_str = ', '.join(str(c) for c in self.children)
            return f"{self.type}({self.value}, [{children_str}])"
        return f"{self.type}({self.value})"


class MathTokenizer:
    """
    Converts a list of recognized symbols (from the AI model)
    into structured tokens for parsing.
    """

    OPERATORS = {'+', '-', '*', '/', '^', '='}
    VARIABLES = {'x', 'y', 'z'}
    FUNCTIONS = {'sqrt', 'sin', 'cos', 'tan', 'log'}
    PARENS = {'(', ')'}
    CONSTANTS = {'pi'}

    def tokenize(self, symbols: List[str]) -> List[Token]:
        """
        Convert raw symbol list to tokens.
        Handles multi-digit numbers and implicit multiplication.

        Example:
            ['1', '2', '+', 'x'] → [Token(NUMBER, '12'), Token(OPERATOR, '+'), Token(VARIABLE, 'x')]
        """
        tokens = []
        i = 0

        while i < len(symbols):
            sym = symbols[i]

            # ── Multi-digit number ──
            if sym.isdigit() or sym == '.':
                num_str, i = self._collect_number(symbols, i)
                tokens.append(Token('NUMBER', num_str, len(tokens)))

            # ── Variable ──
            elif sym in self.VARIABLES:
                # Check for implicit multiplication: 3x → 3 * x
                if tokens and tokens[-1].type in ('NUMBER', 'VARIABLE', 'PAREN_CLOSE'):
                    tokens.append(Token('OPERATOR', '*', len(tokens)))
                tokens.append(Token('VARIABLE', sym, len(tokens)))

            # ── Operator ──
            elif sym in self.OPERATORS:
                tokens.append(Token('OPERATOR', sym, len(tokens)))

            # ── Function ──
            elif sym in self.FUNCTIONS:
                if tokens and tokens[-1].type in ('NUMBER', 'VARIABLE', 'PAREN_CLOSE'):
                    tokens.append(Token('OPERATOR', '*', len(tokens)))
                tokens.append(Token('FUNCTION', sym, len(tokens)))

            # ── Parentheses ──
            elif sym == '(':
                if tokens and tokens[-1].type in ('NUMBER', 'VARIABLE', 'PAREN_CLOSE'):
                    tokens.append(Token('OPERATOR', '*', len(tokens)))
                tokens.append(Token('PAREN_OPEN', '(', len(tokens)))

            elif sym == ')':
                tokens.append(Token('PAREN_CLOSE', ')', len(tokens)))

            # ── Constants ──
            elif sym in self.CONSTANTS:
                if tokens and tokens[-1].type in ('NUMBER', 'VARIABLE'):
                    tokens.append(Token('OPERATOR', '*', len(tokens)))
                tokens.append(Token('CONSTANT', sym, len(tokens)))

            # ── Power (^) ──
            elif sym == '^':
                tokens.append(Token('OPERATOR', '**', len(tokens)))

            i += 1

        return tokens

    def _collect_number(self, symbols: List[str],
                        start: int) -> Tuple[str, int]:
        """Collect consecutive digits and decimal point into one number."""
        num_str = ''
        i = start
        has_decimal = False

        while i < len(symbols):
            if symbols[i].isdigit():
                num_str += symbols[i]
            elif symbols[i] == '.' and not has_decimal:
                num_str += '.'
                has_decimal = True
            else:
                break
            i += 1

        return num_str, i - 1  # -1 because outer loop will increment


class MathParser:
    """
    Parses a list of tokens into a SymPy-compatible
    expression string.

    Connects the AI recognition output to the math solver.

    Flow:
        Recognized Symbols → Tokenizer → Parser → SymPy String → Solver
    """

    def __init__(self):
        self.tokenizer = MathTokenizer()

    def symbols_to_expression(self, symbols: List[str]) -> str:
        """
        Main entry point: Convert recognized symbols to
        a solvable expression string.

        Example:
            Input:  ['x', '^', '2', '+', '2', 'x', '-', '8', '=', '0']
            Output: 'x**2 + 2*x - 8 = 0'
        """
        tokens = self.tokenizer.tokenize(symbols)
        expression = self._tokens_to_string(tokens)
        expression = self._clean_expression(expression)
        return expression

    def symbols_to_equation_parts(self,
                                   symbols: List[str]) -> Tuple[str, Optional[str]]:
        """
        Split equation at '=' sign.

        Returns:
            (lhs_string, rhs_string) or (expression, None) if no '='
        """
        expression = self.symbols_to_expression(symbols)

        if '=' in expression:
            parts = expression.split('=')
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            return lhs, rhs
        else:
            return expression, None

    def detect_problem_type(self, symbols: List[str]) -> str:
        """
        Auto-detect what type of math problem this is.

        Returns one of:
            'linear_equation', 'quadratic_equation', 'polynomial',
            'expression', 'arithmetic'
        """
        has_variable = any(s in ('x', 'y', 'z') for s in symbols)
        has_equals = '=' in symbols
        has_power = '^' in symbols
        has_function = any(s in ('sqrt', 'sin', 'cos', 'log') for s in symbols)

        max_power = self._detect_max_power(symbols)

        if not has_variable:
            return 'arithmetic'
        elif has_equals:
            if max_power == 1:
                return 'linear_equation'
            elif max_power == 2:
                return 'quadratic_equation'
            else:
                return 'polynomial_equation'
        elif has_function:
            return 'function_expression'
        else:
            return 'expression'

    def _detect_max_power(self, symbols: List[str]) -> int:
        """Find the highest power in the expression."""
        max_power = 1
        for i, sym in enumerate(symbols):
            if sym == '^' and i + 1 < len(symbols):
                try:
                    power = int(symbols[i + 1])
                    max_power = max(max_power, power)
                except ValueError:
                    pass
        return max_power

    def _tokens_to_string(self, tokens: List[Token]) -> str:
        """Convert token list to string expression."""
        parts = []
        for token in tokens:
            if token.type == 'OPERATOR' and token.value == '**':
                parts.append('**')
            elif token.type == 'OPERATOR':
                parts.append(f' {token.value} ')
            elif token.type == 'FUNCTION':
                parts.append(f'{token.value}(')
            else:
                parts.append(token.value)
        return ''.join(parts)

    def _clean_expression(self, expr: str) -> str:
        """Clean up spacing and formatting."""
        expr = re.sub(r'\s+', ' ', expr)       # Multiple spaces → single
        expr = re.sub(r'\s*\*\*\s*', '**', expr)  # No spaces around **
        expr = expr.strip()
        return expr