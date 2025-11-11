# 🤖 OLLAMA SETUP & CONFIGURATION GUIDE

Complete guide to setup and configure Ollama for VirusHunter's LLM-powered threat intelligence.

## 📋 Table of Contents

1. [What is Ollama?](#what-is-ollama)
2. [Installation Methods](#installation-methods)
3. [Model Selection](#model-selection)
4. [Configuration](#configuration)
5. [Integration with VirusHunter](#integration-with-virushunter)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 What is Ollama?

Ollama is a local LLM (Large Language Model) runtime that allows you to run models like:
- **Llama 2** (Meta)
- **Mistral** (Mistral AI)
- **CodeLlama** (Meta)
- **Phi** (Microsoft)

**Benefits:**
- ✅ Runs completely offline (no API keys needed)
- ✅ Privacy-focused (data stays local)
- ✅ Free and open-source
- ✅ Easy to use API

---

## 💻 Installation Methods

### Method 1: Native Installation (Recommended)

#### Linux
```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# Start Ollama service
ollama serve
```

#### macOS
```bash
# Download from https://ollama.com/download
# Or use Homebrew:
brew install ollama

# Start service
ollama serve
```

#### Windows
```bash
# Download installer from: https://ollama.com/download
# Run the installer
# Ollama will start automatically
```

---

### Method 2: Docker (Easiest with VirusHunter)

#### Using Docker Compose (Recommended)
```bash
# Start both Ollama and VirusHunter
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs ollama
```

#### Standalone Ollama Docker
```bash
# CPU only
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama

# With GPU
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama
```

---

## 🎭 Model Selection

### Pull a Model

After Ollama is running, download a model:

#### Llama 2 (7B) - Best Balance
```bash
# Standard version (4GB RAM)
ollama pull llama2

# Specific version
ollama pull llama2:7b
ollama pull llama2:13b  # Needs 16GB RAM
ollama pull llama2:70b  # Needs 64GB RAM
```

#### Mistral (7B) - Fast & Efficient
```bash
ollama pull mistral

# Or specific version
ollama pull mistral:7b-instruct
```

#### CodeLlama (7B) - For Technical Analysis
```bash
ollama pull codellama

# Specific versions
ollama pull codellama:7b-instruct
ollama pull codellama:13b
```

#### Phi (2.7B) - Lightweight
```bash
# Smallest, fastest (good for testing)
ollama pull phi
```

### Recommended Models by Use Case

| Use Case | Model | Size | RAM Needed |
|----------|-------|------|------------|
| **Quick Testing** | phi | 1.6GB | 4GB |
| **General Use** | llama2:7b | 3.8GB | 8GB |
| **Best Quality** | mistral:7b | 4.1GB | 8GB |
| **Technical Analysis** | codellama:7b | 3.8GB | 8GB |
| **Maximum Performance** | llama2:13b | 7.3GB | 16GB |

### Check Installed Models
```bash
ollama list
```

---

## ⚙️ Configuration

### 1. Verify Ollama is Running

```bash
# Check service status
curl http://localhost:11434/api/tags

# Should return JSON with available models
```

### 2. Test Model
```bash
# Interactive test
ollama run llama2

# Type a message and press Enter
# Type /bye to exit
```

### 3. Test API
```bash
# Test generation endpoint
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Explain what malware is in one sentence.",
  "stream": false
}'
```

---

## 🔗 Integration with VirusHunter

### Configuration in Code

The VirusHunter code already includes Ollama integration. No changes needed!

**Location**: `app/streamlit_app.py`

```python
def call_ollama(prompt, model="llama2"):
    """Call Ollama LLM"""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return "LLM service unavailable. Start with: ollama serve"
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"
```

### Environment Variables (Optional)

Create `.env` file:
```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_TIMEOUT=30
```

### Docker Integration

If using Docker Compose, Ollama is automatically configured:
```yaml
environment:
  - OLLAMA_API_URL=http://ollama:11434
```

---

## 🧪 Testing

### Test 1: Verify Ollama Service
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Expected output: JSON with model list
```

### Test 2: Simple Generation
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "What is malware?",
  "stream": false
}'
```

### Test 3: From VirusHunter
```python
# Run this in Python
import requests

def test_ollama():
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'llama2',
            'prompt': 'Explain ransomware in one sentence.',
            'stream': False
        }
    )
    print(response.json()['response'])

test_ollama()
```

### Test 4: In Streamlit App
1. Start Ollama: `ollama serve`
2. Start VirusHunter: `streamlit run app/streamlit_app.py`
3. Go to "Intelligence" tab
4. Ask a question about malware
5. Check if LLM responds

---

## 🎯 Usage in VirusHunter

### 1. Malware Analysis
When you analyze a file, Ollama provides:
- **Technical analysis** of detection results
- **Top 3 recommendations** for handling the threat
- **Potential attack vectors** based on features

### 2. Threat Intelligence Chat
Go to "Intelligence" tab to:
- Ask questions about malware types
- Get explanations of attack techniques
- Learn about MITRE ATT&CK tactics
- Understand security concepts

### Example Prompts:
```
- "What is a polymorphic virus?"
- "Explain the difference between malware and ransomware"
- "What are common indicators of malware infection?"
- "How does MITRE ATT&CK help in threat detection?"
```

---

## 🚀 Performance Optimization

### 1. Use Smaller Models for Speed
```bash
# Fastest (good for testing)
ollama pull phi

# Update in streamlit_app.py
call_ollama(prompt, model="phi")
```

### 2. Adjust Context Length
```bash
# In API call, add num_ctx parameter
{
  "model": "llama2",
  "prompt": "...",
  "num_ctx": 2048  # Default: 2048, Max: 8192
}
```

### 3. Use GPU Acceleration
```bash
# Ollama automatically uses GPU if available
# Check GPU usage:
nvidia-smi

# Force CPU only:
OLLAMA_NUM_GPU=0 ollama serve
```

---

## 🐛 Troubleshooting

### Issue 1: "Connection refused" Error

**Problem**: Can't connect to Ollama

**Solutions**:
```bash
# 1. Check if Ollama is running
ps aux | grep ollama

# 2. Start Ollama
ollama serve

# 3. Check port
netstat -an | grep 11434

# 4. Test connection
curl http://localhost:11434/api/tags
```

---

### Issue 2: "Model not found" Error

**Problem**: Requested model not installed

**Solution**:
```bash
# List available models
ollama list

# Pull the model
ollama pull llama2

# Verify
ollama list
```

---

### Issue 3: Slow Response Times

**Problem**: LLM takes too long to respond

**Solutions**:
```bash
# 1. Use smaller model
ollama pull phi  # Faster, smaller

# 2. Reduce prompt length
# Edit prompts in streamlit_app.py to be shorter

# 3. Enable GPU (if available)
# Ollama uses GPU automatically if detected

# 4. Increase timeout
# In call_ollama() function:
timeout=60  # Increase from 30
```

---

### Issue 4: Out of Memory

**Problem**: System runs out of RAM

**Solutions**:
```bash
# 1. Use smaller model
ollama pull phi  # Only 1.6GB

# 2. Close other applications

# 3. Check memory usage
free -h  # Linux
top      # Linux/Mac

# 4. Limit Ollama memory
# Set in Ollama config or use smaller model
```

---

### Issue 5: Docker Container Issues

**Problem**: Ollama container not starting

**Solutions**:
```bash
# 1. Check Docker logs
docker-compose logs ollama

# 2. Restart container
docker-compose restart ollama

# 3. Rebuild
docker-compose up -d --build

# 4. Check if port is available
lsof -i :11434
```

---

## 📊 Model Comparison

### Response Quality vs Speed

| Model | Quality | Speed | RAM | Use Case |
|-------|---------|-------|-----|----------|
| phi | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 4GB | Testing, quick answers |
| llama2:7b | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8GB | **Recommended** general use |
| mistral:7b | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8GB | Best quality |
| codellama:7b | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8GB | Technical analysis |
| llama2:13b | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 16GB | Maximum quality |

---

## 🔄 Complete Setup Workflow

### Quick Setup (5 minutes)
```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start service
ollama serve &

# 3. Pull model
ollama pull llama2

# 4. Test
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Hello!",
  "stream": false
}'

# 5. Start VirusHunter
streamlit run app/streamlit_app.py
```

### Docker Setup (3 minutes)
```bash
# 1. Start everything
docker-compose up -d

# 2. Wait for Ollama to download model
docker-compose logs -f ollama

# 3. Access VirusHunter
# Open: http://localhost:8501
```

---

## 💡 Best Practices

### 1. Model Selection
- **Development**: Use `phi` (fastest)
- **Production**: Use `llama2:7b` (best balance)
- **High Quality**: Use `mistral:7b` (best quality)

### 2. Prompt Engineering
```python
# Good prompt (specific, concise)
prompt = f"""You are a cybersecurity expert. Analyze this malware detection:
Probability: {prob:.1%}
Top features: {features}
Provide: 1) Analysis, 2) Recommendations, 3) Attack vectors
Keep response under 200 words."""

# Bad prompt (vague, long)
prompt = "Tell me everything about this file..."
```

### 3. Error Handling
```python
def call_ollama_safe(prompt, model="llama2"):
    try:
        response = call_ollama(prompt, model)
        return response
    except Exception as e:
        return f"LLM unavailable. Analysis results only."
```

---

## 📚 Additional Resources

- **Ollama Website**: https://ollama.com
- **Ollama GitHub**: https://github.com/ollama/ollama
- **Model Library**: https://ollama.com/library
- **API Documentation**: https://github.com/ollama/ollama/blob/main/docs/api.md

---

## ✅ Verification Checklist

Before using VirusHunter with Ollama:

- [ ] Ollama installed
- [ ] Ollama service running
- [ ] Model downloaded (`ollama list`)
- [ ] API responding (`curl http://localhost:11434/api/tags`)
- [ ] VirusHunter can connect
- [ ] Test question works in Intelligence tab

---

## 🎉 You're Ready!

Ollama is now configured for VirusHunter! The LLM will provide:
- Intelligent malware analysis
- Natural language explanations
- Threat intelligence
- Mitigation recommendations

**Start using**: Go to the "Intelligence" tab in VirusHunter and ask questions!