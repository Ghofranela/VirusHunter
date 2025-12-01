#!/usr/bin/env python3
"""
Run preprocessing as a module
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from scripts.preprocess import main

if __name__ == "__main__":
    main()