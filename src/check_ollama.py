#!/usr/bin/env python3
"""
Check Ollama installation and status
"""
import requests
import subprocess
import sys


def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama is installed")
            print(f"  Version: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        print("✗ Ollama is NOT installed")
        print("\nInstall with:")
        print("  Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh")
        print("  Windows: Download from https://ollama.com/download")
        return False


def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            print("✓ Ollama service is running")
            return True
    except:
        print("✗ Ollama service is NOT running")
        print("\nStart with:")
        print("  ollama serve")
        return False


def check_models():
    """Check available models"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            models = result.stdout.strip()
            if models:
                print("✓ Available models:")
                for line in models.split('\n')[1:]:  # Skip header
                    if line.strip():
                        print(f"  • {line.split()[0]}")
                return True
            else:
                print("✗ No models installed")
                print("\nInstall a model:")
                print("  ollama pull llama2")
                return False
    except:
        return False


def test_generation():
    """Test LLM generation"""
    try:
        print("\nTesting LLM generation...")
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama2',
                'prompt': 'Say "Hello" in one word.',
                'stream': False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()['response']
            print(f"✓ Generation test successful")
            print(f"  Response: {result[:50]}...")
            return True
        else:
            print(f"✗ Generation failed: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Generation test failed: {e}")
        return False


def main():
    print("="*60)
    print("OLLAMA STATUS CHECK")
    print("="*60)
    
    # Check installation
    print("\n1. Checking installation...")
    installed = check_ollama_installed()
    
    if not installed:
        print("\n" + "="*60)
        print("RESULT: Ollama not installed")
        print("="*60)
        return
    
    # Check service
    print("\n2. Checking service...")
    running = check_ollama_running()
    
    if not running:
        print("\n" + "="*60)
        print("RESULT: Ollama installed but not running")
        print("="*60)
        return
    
    # Check models
    print("\n3. Checking models...")
    has_models = check_models()
    
    # Test generation
    if has_models:
        test_generation()
    
    # Summary
    print("\n" + "="*60)
    if installed and running and has_models:
        print("RESULT: ✓ Ollama is ready to use!")
    elif installed and running:
        print("RESULT: ⚠ Ollama is running but needs models")
        print("        Run: ollama pull llama2")
    else:
        print("RESULT: ⚠ Ollama needs configuration")
    print("="*60)


if __name__ == "__main__":
    main()
