#!/usr/bin/env python3
"""
Enhanced Ollama Configuration Checker with Llama3:8b Support
Tests Ollama service, models, and generation speed
"""
import sys
import requests
import time
from datetime import datetime


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_ollama_service():
    """Check if Ollama service is running"""
    print_header("Checking Ollama Service")
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
        else:
            print(f"❌ Ollama returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("\n💡 Start Ollama:")
        print("   ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def list_available_models():
    """List all available Ollama models"""
    print_header("Available Models")
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            if not models:
                print("⚠️  No models installed")
                print("\n💡 Install recommended model:")
                print("   ollama pull llama3:8b")
                return []
            
            print(f"Found {len(models)} model(s):\n")
            
            model_list = []
            for model in models:
                name = model.get('name', 'unknown')
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                modified = model.get('modified_at', 'unknown')
                
                print(f"  📦 {name}")
                print(f"     Size: {size:.2f} GB")
                print(f"     Modified: {modified}")
                
                model_list.append(name)
            
            return model_list
        else:
            print(f"❌ Failed to list models: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")
        return []


def test_model_generation(model_name, prompt="What is malware?", timeout=60):
    """Test model generation speed and quality"""
    print_header(f"Testing {model_name} Generation")
    print(f"Prompt: '{prompt}'")
    print(f"⏳ Timeout: {timeout}s\n")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model_name,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'num_predict': 100,  # Limit for testing
                }
            },
            timeout=timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            
            print(f"✅ Generation successful!")
            print(f"⏱️  Time: {duration:.2f}s")
            print(f"📝 Response length: {len(generated_text)} characters")
            print(f"\n--- Response Preview ---")
            print(generated_text[:200] + ("..." if len(generated_text) > 200 else ""))
            print(f"--- End Preview ---\n")
            
            # Performance rating
            if duration < 5:
                print("⚡ Performance: EXCELLENT (< 5s)")
            elif duration < 15:
                print("✅ Performance: GOOD (< 15s)")
            elif duration < 30:
                print("⚠️  Performance: ACCEPTABLE (< 30s)")
            else:
                print("🐌 Performance: SLOW (> 30s)")
            
            return True
        else:
            print(f"❌ Generation failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Generation timed out after {timeout}s")
        print("\n💡 Solutions:")
        print("   1. Use a faster model: ollama pull llama3:8b")
        print("   2. Increase system resources")
        print("   3. Reduce prompt length")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def recommend_model():
    """Recommend best model based on use case"""
    print_header("Recommended Models for VirusHunter")
    
    models = [
        {
            'name': 'llama3:8b',
            'params': '3.8B',
            'speed': '3-8s',
            'quality': '⭐⭐⭐⭐',
            'note': 'RECOMMENDED - Fast & efficient',
            'install': 'ollama pull llama3:8b'
        },
        {
            'name': 'llama3:8b',
            'params': '7B',
            'speed': '10-30s',
            'quality': '⭐⭐⭐⭐⭐',
            'note': 'Higher quality, slower',
            'install': 'ollama pull llama3:8b'
        },
        {
            'name': 'llama3:8b',
            'params': '8B',
            'speed': '15-40s',
            'quality': '⭐⭐⭐⭐⭐',
            'note': 'Best quality, slowest',
            'install': 'ollama pull llama3:8b'
        }
    ]
    
    for model in models:
        print(f"\n📦 {model['name']}")
        print(f"   Size:    {model['params']} parameters")
        print(f"   Speed:   {model['speed']} per response")
        print(f"   Quality: {model['quality']}")
        print(f"   Note:    {model['note']}")
        print(f"   Install: {model['install']}")


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("  🛡️ VirusHunter - Ollama Configuration Check")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Check service
    if not check_ollama_service():
        print("\n" + "=" * 60)
        print("⚠️  Setup Required")
        print("=" * 60)
        print("\n1. Start Ollama service:")
        print("   ollama serve")
        print("\n2. Install a model (in another terminal):")
        print("   ollama pull llama3:8b")
        print("\n3. Run this script again")
        sys.exit(1)
    
    # Step 2: List models
    models = list_available_models()
    
    if not models:
        print("\n" + "=" * 60)
        print("⚠️  No Models Found")
        print("=" * 60)
        recommend_model()
        sys.exit(1)
    
    # Step 3: Test models
    print("\n" + "=" * 60)
    print("  Testing Model Performance")
    print("=" * 60)
    
    # Prioritize testing order: llama3:8b > llama3:8b > llama3:8b > others
    test_order = []
    for preferred in ['llama3:8b', 'llama3:8b', 'llama3:8b']:
        matching = [m for m in models if preferred in m.lower()]
        test_order.extend(matching)
    
    # Add remaining models
    for model in models:
        if model not in test_order:
            test_order.append(model)
    
    # Test first available model (or up to 3)
    success = False
    for model in test_order[:3]:
        if test_model_generation(model):
            success = True
            break
        print()
    
    # Step 4: Recommendations
    if not success:
        print("\n" + "=" * 60)
        print("⚠️  Generation Tests Failed")
        print("=" * 60)
        recommend_model()
    else:
        print("\n" + "=" * 60)
        print("✅ Configuration Check Complete")
        print("=" * 60)
        print("\n✅ Ollama is ready for VirusHunter!")
        print("   You can now run: ./run.sh")
    
    # Always show recommendations
    recommend_model()
    
    print("\n" + "=" * 60)
    print("  Configuration Check Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()