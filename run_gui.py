import sys
import os

# Ensure the root directory is in the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        import PyQt5
    except ImportError:
        print("Error: PyQt5 is not installed in the virtual environment.")
        print("Please run: .venv\\Scripts\\pip install PyQt5")
        sys.exit(1)
        
    from gui.app import run_app
    run_app()
