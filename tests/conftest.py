import os
import sys

# the package is a flat folder of scripts (pipeline/*.py) that import each other by module name
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
