import os
import sys

# Make sure the project root is importable (agent/, harness/, tools/) regardless
# of which directory pytest is invoked from.
sys.path.insert(0, os.path.dirname(__file__))
