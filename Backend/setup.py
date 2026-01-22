# test_imports.py
import sys
import os

print("=" * 60)
print("DEBUG: Testing imports")
print("=" * 60)

# Add paths
sys.path.insert(0, '/app/Backend/src')
print(f"Python path: {sys.path}")

# Check if directories exist
print(f"\nChecking directories:")
print(f"/app exists: {os.path.exists('/app')}")
print(f"/app/Backend exists: {os.path.exists('/app/Backend')}")
print(f"/app/Backend/src exists: {os.path.exists('/app/Backend/src')}")

# List contents
print(f"\nContents of /app/Backend/src:")
try:
    for item in os.listdir('/app/Backend/src'):
        print(f"  - {item}")
except Exception as e:
    print(f"  Error: {e}")

# Try to import
print(f"\nTrying imports...")
try:
    # Try to import the module directly
    import importlib.util
    
    # Check if DietPlanRAG exists
    diet_rag_path = '/app/Backend/src/DietPlanRAG'
    if os.path.exists(diet_rag_path):
        print(f"✅ DietPlanRAG directory exists")
        if os.path.exists(f'{diet_rag_path}/DietGuide.py'):
            print(f"✅ DietGuide.py exists")
            
            # Try to import
            spec = importlib.util.spec_from_file_location("DietGuide", f'{diet_rag_path}/DietGuide.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ DietGuide imported successfully")
        else:
            print(f"❌ DietGuide.py not found in {diet_rag_path}/")
    else:
        print(f"❌ DietPlanRAG directory not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()