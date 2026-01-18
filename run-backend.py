"""
FastAPI Backend Runner for Diet Planner
This script configures the Python path and starts the FastAPI server
"""
import sys
import os
from pathlib import Path

# Get paths - Backend is INSIDE DietPlanner
DIETPLANNER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DIETPLANNER_DIR  # For this structure, they're the same
BACKEND_SRC = DIETPLANNER_DIR / "Backend" / "src"

print("=" * 50)
print("Python Path Configuration")
print(f"  DietPlanner Dir: {DIETPLANNER_DIR}")
print(f"  Backend Src: {BACKEND_SRC}")
print("=" * 50)

# Verify Backend exists
if not BACKEND_SRC.exists():
    print(f"ERROR: Backend source directory not found: {BACKEND_SRC}")
    print(f"\nChecking if Backend folder exists: {(DIETPLANNER_DIR / 'Backend').exists()}")
    if (DIETPLANNER_DIR / "Backend").exists():
        print(f"Backend folder contents: {list((DIETPLANNER_DIR / 'Backend').iterdir())}")
    sys.exit(1)

# Add paths to Python path
sys.path.insert(0, str(DIETPLANNER_DIR))
sys.path.insert(0, str(BACKEND_SRC))

print(f"[OK] Backend source found!")
print("=" * 50)

# Change to DietPlanner directory for relative imports
os.chdir(DIETPLANNER_DIR)

# Now import and run uvicorn
if __name__ == "__main__":
    try:
        import uvicorn
        
        print("\n" + "=" * 50)
        print("Starting FastAPI Server...")
        print("=" * 50 + "\n")
        
        # Run the FastAPI app
        uvicorn.run(
            "Backend.src.DietPlanRAG.DietGuide:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            reload_dirs=[str(BACKEND_SRC)]
        )
        
    except ImportError as e:
        print(f"\nERROR: Failed to import required modules: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install fastapi uvicorn python-multipart")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)