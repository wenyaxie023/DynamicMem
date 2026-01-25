import sys
import os

print("Sys Path:", sys.path)
try:
    from hipporag import HippoRAG
    print("Import Successful!")
except ImportError as e:
    print("Import Failed:", e)
