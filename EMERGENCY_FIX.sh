#!/bin/bash
# Emergency fix for ai.py syntax errors
# This script removes f-string prefixes from all problematic lines

echo "🔧 Applying emergency syntax fixes to ai.py..."

# Backup original
cp ai.py ai.py.ORIGINAL_BACKUP

# Remove f prefix from all lines with {{ that don't have real variables
python3 << 'PYTHON_EOF'
import re

with open('ai.py', 'r') as f:
    content = f.read()

# Strategy: Remove f prefix from lines that have {{ but no real f-string variables
# Pattern: f"{{...}}" where ... doesn't contain {variable}

# Find and replace problematic f-strings
patterns_to_fix = [
    (r'f"(\{{[^}]*\}})*"', r'"\1"'),  # f"{{...}}" → "{{...}}"
    (r"f'(\{{'[^}']*\}}')*'", r"'\1'"),  # f'{{...}}' → '{{...}}'
]

# More aggressive: Just remove all f" that come before {{
lines = content.split('\n')
fixed_lines = []
for line in lines:
    # Only fix lines that have f" followed by {{ with no real variables
    if 'f"' in line and '{{' in line:
        # Check if there are real variables (curly braces without escaping)
        # Safe pattern: if line has {{{ then it's probably an escaped brace
        if not re.search(r'\{[^{]|\}[^}]', line.replace('{{', '').replace('}}', '')):
            # No real variables, safe to remove f prefix
            line = line.replace('f"{{', '"{{').replace("f'{{", "'{{")
    fixed_lines.append(line)

with open('ai.py', 'w') as f:
    f.write('\n'.join(fixed_lines))

print("✓ Fixed f-strings")
PYTHON_EOF

# Verify
echo ""
echo "🧪 Verifying syntax..."
if python3 -m py_compile ai.py 2>&1 | grep -q "SyntaxError: f-string"; then
    echo "⚠️  SyntaxError still present - trying more aggressive fix..."

    # Nuclear option: Remove ALL f-string prefixes from lines with {{
    python3 << 'NUKE_EOF'
with open('ai.py', 'r') as f:
    lines = f.readlines()

# Remove f prefix from every line with {{ - period
for i, line in enumerate(lines):
    if 'f"' in line and '{{' in line:
        lines[i] = line.replace('f"{{', '"{{')
    if "f'" in line and '{{' in line:
        lines[i] = lines[i].replace("f'{{", "'{{")

with open('ai.py', 'w') as f:
    f.writelines(lines)
print("✓ Applied nuclear option")
NUKE_EOF

fi

# Final verification
echo ""
if python3 -m py_compile ai.py 2>&1 | grep -q "SyntaxError"; then
    echo "❌ Still has SyntaxError - file needs manual review"
    python3 -m py_compile ai.py 2>&1 | head -10
else
    echo "✅ SUCCESS! ai.py now compiles without SyntaxError"
    echo ""
    echo "📝 Next steps:"
    echo "   python3 ai.py"
    echo ""
fi
