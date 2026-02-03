---
name: calculator
description: Perform basic arithmetic calculations including addition, subtraction, multiplication, division, and handling of decimal numbers, parentheses, and order of operations. Use when users ask to calculate, compute, or solve mathematical expressions such as "What is 123 + 456?", "Calculate 15.5 * 3", "Compute (10 + 5) * 2 - 8", or any arithmetic operations.
---

# Calculator

This skill provides guidance for performing accurate arithmetic calculations.

## Supported Operations

- **Addition**: `a + b`
- **Subtraction**: `a - b`
- **Multiplication**: `a * b`
- **Division**: `a / b`

## Calculation Rules

1. **Order of operations**: Follow standard PEMDAS (Parentheses, Exponents, Multiplication/Division, Addition/Subtraction)
2. **Decimal precision**: Maintain appropriate precision for decimal calculations
3. **Division by zero**: Handle gracefully with an error message
4. **Negative numbers**: Support negative values in calculations

## Examples

```
123 + 456 = 579
15.5 * 3 = 46.5
(10 + 5) * 2 - 8 = 22
100 / 4 = 25
```

## Output Format

Present results clearly with:
- The original expression
- The calculated result
- Step-by-step breakdown for complex calculations (optional, when helpful)

Example:
```
Expression: (10 + 5) * 2 - 8
Step 1: 10 + 5 = 15
Step 2: 15 * 2 = 30
Step 3: 30 - 8 = 22
Result: 22
```
