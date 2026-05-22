"""Patches tvm_ffi to handle missing torch_c_dlpack_ext gracefully."""
import re

f = '/usr/local/lib/python3.11/dist-packages/tvm_ffi/_optional_torch_c_dlpack.py'
lines = open(f).readlines()

out = []
i = 0
while i < len(lines):
    line = lines[i]
    # Find the assignment line
    if '_LIB = load_torch_c_dlpack_extension()' in line and 'try' not in line:
        indent = len(line) - len(line.lstrip())
        pad = ' ' * indent
        out.append(pad + 'try:\n')
        out.append(pad + '    _LIB = load_torch_c_dlpack_extension()\n')
        out.append(pad + 'except Exception:\n')
        out.append(pad + '    _LIB = None\n')
        print(f'Patched line {i+1}')
    else:
        out.append(line)
    i += 1

open(f, 'w').write(''.join(out))
print('Done')
