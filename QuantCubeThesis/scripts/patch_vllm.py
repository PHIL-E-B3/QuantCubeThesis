"""Patches tvm_ffi to handle missing torch_c_dlpack_ext gracefully."""
f = '/usr/local/lib/python3.11/dist-packages/tvm_ffi/_optional_torch_c_dlpack.py'
txt = open(f).read()
old = '_LIB = load_torch_c_dlpack_extension()'
new = 'try:\n    _LIB = load_torch_c_dlpack_extension()\nexcept Exception:\n    _LIB = None'
if old in txt:
    open(f, 'w').write(txt.replace(old, new))
    print('Patched successfully')
else:
    print('Pattern not found - already patched or file changed')
