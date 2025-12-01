#!/usr/bin/env python3
"""
Run training as a module
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from scripts.train import main

if __name__ == "__main__":
    main()