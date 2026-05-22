"""Patches tvm_ffi _optional_torch_c_dlpack.py to skip the broken torch extension."""
import re

f = '/usr/local/lib/python3.11/dist-packages/tvm_ffi/_optional_torch_c_dlpack.py'
txt = open(f).read()

# Remove all previous patch attempts (multiple try: blocks etc.)
# Find the line that assigns _LIB and replace everything from the function definition
# to the _LIB assignment with a safe version

# Strategy: find the load_torch_c_dlpack_extension function and the _LIB assignment
# Replace the _LIB assignment line(s) with a safe try/except

# First, restore any double-patched state by removing stray try: blocks
# Then do a clean single replacement

lines = txt.split('\n')
new_lines = []
skip_next = False
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    # Remove stray 'try:' lines that have no body (followed by another try: or _LIB)
    if stripped == 'try:' and i + 1 < len(lines):
        next_stripped = lines[i+1].strip()
        if next_stripped == 'try:' or next_stripped.startswith('_LIB = load_torch'):
            # This is a broken/duplicate try: — skip it
            i += 1
            continue

    new_lines.append(line)
    i += 1

txt = '\n'.join(new_lines)

# Now do the clean replacement
old = '_LIB = load_torch_c_dlpack_extension()'
if old in txt:
    # Find indentation of this line
    for line in txt.split('\n'):
        if old in line:
            indent = len(line) - len(line.lstrip())
            pad = ' ' * indent
            break

    new_block = f"""{pad}try:
{pad}    _LIB = load_torch_c_dlpack_extension()
{pad}except Exception as _e:
{pad}    _LIB = None"""

    txt = txt.replace(pad + old, new_block)
    print(f'Applied clean patch (indent={indent})')
else:
    print('Pattern not found - checking current state')
    for i, line in enumerate(txt.split('\n')):
        if 'LIB' in line or 'try' in line.lower():
            print(f'  Line {i+1}: {repr(line)}')

open(f, 'w').write(txt)
print('Done - verifying:')
import ast
try:
    ast.parse(open(f).read())
    print('  File is valid Python')
except SyntaxError as e:
    print(f'  Syntax error: {e}')
