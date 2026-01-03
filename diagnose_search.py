
import sys
import os
import sqlite3
import pickle
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.core.config import ConfigLoader
from src.core.registry import ComponentRegistry

def diagnose():
    print("--- Diagnostic Start ---")
    
    # 1. Check sqlite-vec
    try:
        import sqlite_vec
        print(f"✓ sqlite-vec module found: {sqlite_vec.loadable_path()}")
    except ImportError:
        print("✗ sqlite-vec module NOT found")

    # 2. Check DB Connection and Extension
    try:
        conn = sqlite3.connect('data/recall.db')
        conn.enable_load_extension(True)
        try:
            import sqlite_vec
            conn.load_extension(sqlite_vec.loadable_path())
            print("✓ sqlite-vec extension loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load sqlite-vec extension: {e}")
            
        # 3. Check Vectors
        cursor = conn.execute("SELECT COUNT(*) as count FROM messages WHERE vector IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"Messages with vectors: {count}")
        
        if count == 0:
            print("⚠ NO VECTORS FOUND! Run 'crec.py rebuild --vectors-only --yes'")
            return

        # 4. Test Search (Manual)
        print("\n--- Test Search 'data science' (Manual Vector) ---")
        # Load vectorizer to get query vector
        cfg = ConfigLoader.load()
        registry = ComponentRegistry()
        vectorizer_cfg = cfg["components"]["vectorizer"]
        vectorizer = registry.create_instance(
            name="vectorizer",
            class_path=vectorizer_cfg["class"],
            config=vectorizer_cfg.get("config", {})
        )
        vectorizer.load()
        
        vec = vectorizer.vectorize("data science")
        
        # Search using raw SQL
        # Fallback logic simulation
        cursor = conn.execute("SELECT id, content, vector FROM messages WHERE vector IS NOT NULL")
        results = []
        for row in cursor:
            stored_vec = pickle.loads(row[2])
            score = float(np.dot(vec, stored_vec))
            if score > 0.3:
                results.append((score, row[1][:50]))
        
        results.sort(key=lambda x: x[0], reverse=True)
        print(f"Found {len(results)} matches > 0.3")
        for score, content in results[:3]:
            print(f"  {score:.3f}: {content}...")

    except Exception as e:
        print(f"Diagnostic Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    diagnose()
