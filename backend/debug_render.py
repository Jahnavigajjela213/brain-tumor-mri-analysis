import os
import sys

def debug_paths():
    print("--- RENDER RUNTIME DIAGNOSTICS ---")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print("\n--- Python Search Paths (sys.path) ---")
    for path in sys.path:
        print(f"  - {path}")
    
    print("\n--- Directory Contents (Recursive) ---")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}[{os.path.basename(root) or '.'}]")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            if not f.endswith(('.pyc', '.pth')): # Clean output
                print(f"{sub_indent}{f}")
        if level > 2: # Don't go too deep
            break

if __name__ == "__main__":
    debug_paths()
    # Keep process alive long enough to read logs if needed
    import time
    time.sleep(5)
