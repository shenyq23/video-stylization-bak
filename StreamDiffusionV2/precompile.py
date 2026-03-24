import torch
import time
import sys
sys.path.append("../")

print("="*60)
print("Starting pre-compilation of the 'sige3d' CUDA extension...")
print("This process can take several minutes. Please be patient and do not interrupt.")
print("You will see compiler warnings, which are normal.")
print("="*60)

start_time = time.time()

try:
    # This is the function call that triggers the JIT compilation.
    # We are importing it and calling it in a controlled way.
    from deps.sige3d.torch_kernels._sige_cuda import get_sige3d_cuda_ext
    
    # Calling the function ensures the library is loaded/built.
    ext = get_sige3d_cuda_ext()
    
    end_time = time.time()
    
    print("\n" + "="*60)
    print("✅ COMPILE SUCCESSFUL!")
    print(f"✅ Compilation took {end_time - start_time:.2f} seconds.")
    print(f"✅ CUDA extension loaded: {ext}")
    print(f"✅ The compiled file is located at: {ext.__file__}")
    print("="*60)
    print("\nYou can now run your main inference script.")

except Exception as e:
    end_time = time.time()
    print("\n" + "!"*60)
    print(f"❌ COMPILATION FAILED after {end_time - start_time:.2f} seconds.")
    print("❌ An error occurred during compilation:")
    print(e)
    print("!"*60)
    print("\nPlease check the error messages above. You might need to downgrade Python to 3.10.")