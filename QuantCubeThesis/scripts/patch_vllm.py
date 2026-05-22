"""
Replaces tvm_ffi/_optional_torch_c_dlpack.py with a stub that skips
loading the incompatible torch_c_dlpack_ext library.
"""

stub = '''# Stub: torch_c_dlpack_ext skipped (incompatible with installed torch version)
_LIB = None

def load_torch_c_dlpack_extension():
    return None
'''

f = '/usr/local/lib/python3.11/dist-packages/tvm_ffi/_optional_torch_c_dlpack.py'
open(f, 'w').write(stub)
print('Replaced with stub')

import ast
try:
    ast.parse(open(f).read())
    print('Syntax OK')
except SyntaxError as e:
    print(f'Syntax error: {e}')
