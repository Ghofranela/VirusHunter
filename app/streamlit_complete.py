#!/usr/bin/env python3
"""
VirusHunter - Modern Malware Detection Interface
Deep Learning + LLM Analysis in One File
"""
import streamlit as st
import numpy as np
import torch
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.model import MalwareDetector
    import joblib
except ImportError:
    pass

import requests

# ====================================================================
# PAGE CONFIGURATION
# ====================================================================

st.set_page_config(
    page_title="VirusHunter",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================================================================
# CUSTOM STYLING
# ====================================================================

st.markdown("""
<style>
:root {
    --primary: #0066cc;
    --secondary: #1a1a2e;
    --surface: #16213e;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --text-primary: #ffffff;
    --text-secondary: #d1d5db;
}

* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f0f1e;
    color: #ffffff;
}

[data-testid="stHeader"] {
    background-color: transparent;
}

[data-testid="stSidebar"] {
    background-color: #1a1a2e;
}

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: #0066cc;
    box-shadow: 0 0 20px rgba(0, 102, 204, 0.1);
}

.status-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.status-safe {
    background-color: rgba(16, 185, 129, 0.1);
    color: #10b981;
    border: 1px solid #10b981;
}

.status-warning {
    background-color: rgba(245, 158, 11, 0.1);
    color: #f59e0b;
    border: 1px solid #f59e0b;
}

.status-danger {
    background-color: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    border: 1px solid #ef4444;
}

.feature-list {
    background-color: #16213e;
    border-left: 3px solid #0066cc;
    padding: 15px;
    border-radius: 4px;
    margin: 10px 0;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2d3561, transparent);
    margin: 30px 0;
}

.button-primary {
    background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.3s ease;
}

.button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(0, 102, 204, 0.3);
}

</style>
""", unsafe_allow_html=True)

# ====================================================================
# CACHE & UTILITIES
# ====================================================================

@st.cache_resource
def load_model():
    """Load Deep Learning model"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MalwareDetector()
        
        checkpoint = torch.load('models/best_model.pth', map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        scaler = joblib.load('models/preprocessor.pkl')
        return model, scaler, device
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None, None

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
            timeout=90
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return "⚠️ LLM service unavailable. Start with: `ollama serve`"
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to Ollama. Ensure it's running: `ollama serve`"
    except requests.exceptions.Timeout:
        return "⚠️ LLM request timed out. Try again or use a smaller model."
    except Exception as e:
        return f"⚠️ Error connecting to LLM: {str(e)}"

def get_risk_level(prob):
    """Get risk classification"""
    if prob > 0.9:
        return "CRITICAL", "danger", prob
    elif prob > 0.7:
        return "HIGH", "warning", prob
    elif prob > 0.4:
        return "MEDIUM", "warning", prob
    else:
        return "LOW", "safe", prob

def check_system_status():
    """Check system components"""
    status = {
        'ollama': False,
        'model': False,
        'device': 'cpu'
    }
    
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=2)
        status['ollama'] = r.status_code == 200
    except:
        pass
    
    if Path('models/best_model.pth').exists():
        status['model'] = True
    
    status['device'] = 'GPU' if torch.cuda.is_available() else 'CPU'
    
    return status

def extract_features_from_file(file_obj, file_type):
    """Extract features from different file types"""
    try:
        file_obj.seek(0)
        file_bytes = file_obj.read()
        file_size = len(file_bytes)
        
        # Initialize feature vector
        features = np.zeros(2381)
        
        # Basic file metadata features (first 50 features)
        features[0] = file_size
        features[1] = len(set(file_bytes))  # Unique bytes
        features[2] = file_bytes.count(0x00) / max(file_size, 1)  # Null byte ratio
        features[3] = sum(file_bytes) / max(file_size, 1)  # Average byte value
        
        # Entropy calculation
        if file_size > 0:
            byte_counts = np.bincount(file_bytes, minlength=256)
            probabilities = byte_counts / file_size
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            features[4] = entropy
        
        # Byte frequency features (features 50-305: 256 byte frequencies)
        if file_size > 0:
            byte_freq = np.bincount(file_bytes, minlength=256) / file_size
            features[50:306] = byte_freq
        
        # N-gram features (bigrams, features 306-1000)
        if file_size > 1:
            bigrams = {}
            for i in range(min(file_size - 1, 10000)):  # Limit for performance
                bigram = (file_bytes[i], file_bytes[i+1])
                bigrams[bigram] = bigrams.get(bigram, 0) + 1
            
            # Top bigrams
            top_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[:694]
            for idx, (bigram, count) in enumerate(top_bigrams):
                if 306 + idx < 1000:
                    features[306 + idx] = count / max(file_size, 1)
        
        # PE header features (if executable)
        if file_type in ['exe', 'dll', 'bin', 'so']:
            # Check for PE signature
            if len(file_bytes) > 64 and file_bytes[0:2] == b'MZ':
                features[1000] = 1  # PE file indicator
                
                # Extract PE header info if available
                try:
                    pe_offset = int.from_bytes(file_bytes[60:64], 'little')
                    if pe_offset < len(file_bytes) - 4:
                        if file_bytes[pe_offset:pe_offset+2] == b'PE':
                            features[1001] = 1  # Valid PE signature
                            
                            # Number of sections
                            if pe_offset + 6 < len(file_bytes):
                                num_sections = int.from_bytes(file_bytes[pe_offset+6:pe_offset+8], 'little')
                                features[1002] = min(num_sections, 100)
                except:
                    pass
            
            # Check for ELF signature (Linux)
            elif len(file_bytes) > 4 and file_bytes[0:4] == b'\x7fELF':
                features[1003] = 1  # ELF file indicator
        
        # Document-specific features
        elif file_type in ['pdf', 'docx', 'doc', 'rtf']:
            # PDF features
            if b'%PDF' in file_bytes[:10]:
                features[1100] = 1  # PDF indicator
                features[1101] = file_bytes.count(b'/JavaScript')
                features[1102] = file_bytes.count(b'/OpenAction')
                features[1103] = file_bytes.count(b'/Launch')
            
            # Office document features
            elif b'PK\x03\x04' in file_bytes[:10]:  # ZIP-based (docx, xlsx)
                features[1104] = 1  # Office document indicator
                features[1105] = file_bytes.count(b'macro')
                features[1106] = file_bytes.count(b'VBA')
        
        # Archive features
        elif file_type in ['zip', 'rar', '7z', 'tar', 'gz']:
            # ZIP signature
            if file_bytes[:4] == b'PK\x03\x04':
                features[1200] = 1
            # RAR signature
            elif file_bytes[:7] == b'Rar!\x1a\x07\x00':
                features[1201] = 1
            # 7z signature
            elif file_bytes[:6] == b"7z\xbc\xaf'\x1c":
                features[1202] = 1
        
        # Script features
        elif file_type in ['py', 'js', 'ps1', 'sh', 'bat', 'vbs']:
            try:
                text_content = file_bytes.decode('utf-8', errors='ignore')
                
                # Suspicious keywords
                suspicious_keywords = [
                    'eval', 'exec', 'system', 'shell', 'cmd', 'powershell',
                    'download', 'invoke', 'base64', 'decode', 'encrypt',
                    'shellcode', 'payload', 'exploit'
                ]
                
                for idx, keyword in enumerate(suspicious_keywords):
                    if 1300 + idx < 2381:
                        features[1300 + idx] = text_content.lower().count(keyword)
                
                # Obfuscation indicators
                features[1350] = text_content.count('\\x')  # Hex encoding
                features[1351] = len([c for c in text_content if ord(c) > 127])  # Non-ASCII
                features[1352] = text_content.count('chr(')  # Character encoding
            except:
                pass
        
        # Statistical features (last 500 features)
        if file_size > 0:
            # Byte value statistics
            byte_array = np.array(list(file_bytes))
            features[1881] = np.mean(byte_array)
            features[1882] = np.std(byte_array)
            features[1883] = np.median(byte_array)
            features[1884] = np.min(byte_array)
            features[1885] = np.max(byte_array)
            
            # Runs of repeated bytes (potential packing/encryption)
            max_run = 1
            current_run = 1
            for i in range(1, min(file_size, 10000)):
                if file_bytes[i] == file_bytes[i-1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            features[1886] = max_run
        
        return features.reshape(1, -1)
        
    except Exception as e:
        st.error(f"Feature extraction error: {str(e)}")
        return None
    """Save analysis to history"""
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    
    st.session_state['analysis_history'].insert(0, {
        'filename': filename,
        'probability': prob,
        'risk': risk_level,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Keep only last 50 entries
    if len(st.session_state['analysis_history']) > 50:
        st.session_state['analysis_history'] = st.session_state['analysis_history'][:50]

# ====================================================================
# SIDEBAR NAVIGATION
# ====================================================================

with st.sidebar:
    st.markdown("### 🛡️ VirusHunter")
    st.markdown("Intelligent malware detection system")
    
    page = st.radio(
        "Navigation",
        ["Overview", "Analyze", "Intelligence", "History"],
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    status = check_system_status()
    st.markdown("### System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if status['ollama']:
            st.success("🟢 LLM Online")
        else:
            st.warning("🟡 LLM Offline")
    
    with col2:
        if status['model']:
            st.success("🟢 Model Ready")
        else:
            st.error("🔴 Model Missing")
    
    st.caption(f"🖥️ Device: {status['device']}")

# ====================================================================
# PAGE: OVERVIEW
# ====================================================================

if page == "Overview":
    st.title("🛡️ VirusHunter")
    st.markdown("### Advanced Malware Detection Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Detection Accuracy", "98.5%", "Deep Learning")
    
    with col2:
        st.metric("Features Analyzed", "2,381", "Per Sample")
    
    with col3:
        st.metric("Processing Time", "< 100ms", "GPU Accelerated")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 How It Works")
        st.markdown("""
        **VirusHunter** uses advanced Deep Learning combined with Large Language Models 
        to provide intelligent malware detection and analysis.
        
        1. **Upload** binary or feature files (.npy format)
        2. **Analyze** with neural networks
        3. **Get** risk assessment and probability score
        4. **Understand** with AI-powered insights
        """)
    
    with col2:
        st.markdown("### ⚡ Key Features")
        st.markdown("""
        - ✅ Real-time threat detection
        - 🧠 Deep neural network analysis
        - 💬 Natural language explanations
        - 🌐 Threat intelligence integration
        - 📊 Historical tracking
        - 🎯 MITRE ATT&CK correlation
        """)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### 📈 Recent Analysis")
    if 'analysis_history' in st.session_state and len(st.session_state['analysis_history']) > 0:
        for analysis in st.session_state['analysis_history'][:5]:
            col1, col2, col3 = st.columns([3, 1, 2])
            with col1:
                st.markdown(f"**{analysis['filename']}**")
            with col2:
                prob = analysis['probability']
                if prob > 0.5:
                    st.error(f"{prob:.1%}")
                else:
                    st.success(f"{prob:.1%}")
            with col3:
                st.caption(analysis['timestamp'])
    else:
        st.info("📭 No analysis history yet. Head to Analyze to get started.")

# ====================================================================
# PAGE: ANALYZE
# ====================================================================

elif page == "Analyze":
    st.title("🔬 File Analysis")
    
    tab1, tab2 = st.tabs(["📤 Upload", "🧪 Generate Sample"])
    
    with tab1:
        st.markdown("### Upload File for Analysis")
        
        file_type = st.radio(
            "Select file type:",
            ["Feature Vector (.npy)", "Executable (.exe, .dll, .bin)", "Document (.pdf, .docx)", "Archive (.zip, .rar)", "Script (.py, .js, .ps1)"],
            horizontal=True
        )
        
        # Determine file types based on selection
        if "Feature Vector" in file_type:
            accepted_types = ['npy']
            help_text = "Preprocessed feature vectors in numpy format"
        elif "Executable" in file_type:
            accepted_types = ['exe', 'dll', 'bin', 'so']
            help_text = "Binary executable files for analysis"
        elif "Document" in file_type:
            accepted_types = ['pdf', 'docx', 'doc', 'rtf']
            help_text = "Documents that may contain malicious macros or exploits"
        elif "Archive" in file_type:
            accepted_types = ['zip', 'rar', '7z', 'tar', 'gz']
            help_text = "Compressed archives that may contain malware"
        else:  # Script
            accepted_types = ['py', 'js', 'ps1', 'sh', 'bat', 'vbs']
            help_text = "Script files for behavioral analysis"
        
        uploaded_file = st.file_uploader(
            f"Select file ({', '.join(['.' + t for t in accepted_types])})",
            type=accepted_types,
            help=help_text
        )
        
        if uploaded_file:
            try:
                from io import BytesIO
                
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                # Read file bytes
                bytes_data = uploaded_file.read()
                
                # Try multiple loading methods
                features = None
                load_method = None
                
                # Method 1: Try without allow_pickle first (safest)
                try:
                    features = np.load(BytesIO(bytes_data), allow_pickle=False)
                    load_method = "standard numpy"
                except:
                    pass
                
                # Method 2: Try with allow_pickle
                if features is None:
                    try:
                        features = np.load(BytesIO(bytes_data), allow_pickle=True)
                        load_method = "numpy with pickle"
                    except:
                        pass
                
                # Method 3: Try loading as raw bytes and converting
                if features is None:
                    try:
                        features = np.frombuffer(bytes_data, dtype=np.float32)
                        load_method = "raw float32"
                    except:
                        pass
                
                # Method 4: Try as float64
                if features is None:
                    try:
                        features = np.frombuffer(bytes_data, dtype=np.float64)
                        load_method = "raw float64"
                    except:
                        pass
                
                if features is None:
                    raise ValueError("Could not load file with any known method")
                
                # Handle different numpy array formats
                if isinstance(features, np.ndarray):
                    if len(features.shape) == 1:
                        features = features.reshape(1, -1)
                elif hasattr(features, 'item'):
                    # If it's a numpy scalar or 0-d array
                    features = np.array(features).reshape(1, -1)
                else:
                    # If it's a pickled object, try to extract array
                    features = np.array(features)
                    if len(features.shape) == 1:
                        features = features.reshape(1, -1)
                
                # Validate feature dimensions
                if features.shape[1] != 2381:
                    st.warning(f"⚠️ Feature count mismatch: {features.shape[1]} found, 2381 expected.")
                    
                    # Try to pad or truncate
                    if st.checkbox("🔧 Attempt auto-correction (pad/truncate)?"):
                        if features.shape[1] < 2381:
                            # Pad with zeros
                            padding = np.zeros((features.shape[0], 2381 - features.shape[1]))
                            features = np.concatenate([features, padding], axis=1)
                            st.info(f"✅ Padded to 2381 features with zeros")
                        else:
                            # Truncate
                            features = features[:, :2381]
                            st.info(f"✅ Truncated to 2381 features")
                
                if features.shape[1] == 2381:
                    st.session_state['features'] = features
                    st.session_state['filename'] = uploaded_file.name
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                    st.info(f"📊 Shape: {features.shape} | Method: {load_method}")
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.info("💡 Ensure file is a valid .npy feature vector with 2381 features.")
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.write(f"**File name:** {uploaded_file.name}")
                    st.write(f"**File size:** {uploaded_file.size} bytes")
                    st.write(f"**File type:** {uploaded_file.type}")
                    st.write(f"**Error type:** {type(e).__name__}")
                    st.write(f"**Error message:** {str(e)}")
                    
                    # Show first few bytes
                    uploaded_file.seek(0)
                    first_bytes = uploaded_file.read(16)
                    st.write(f"**First 16 bytes:** {first_bytes.hex()}")
                    
                    st.markdown("---")
                    st.markdown("**Possible solutions:**")
                    st.markdown("1. Verify the file is a valid numpy array saved with `np.save()`")
                    st.markdown("2. Check that features were extracted correctly")
                    st.markdown("3. Try regenerating the feature file")
                    st.markdown("4. Use the 'Generate Sample' tab to test the system")
    
    with tab2:
        st.markdown("### Generate Test Sample")
        st.caption("Create synthetic samples for testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🦠 Malware Sample", use_container_width=True):
                model, scaler, device = load_model()
                if scaler:
                    features = np.random.randn(1, 2381) + 2.5
                    features = scaler.transform(features)
                    st.session_state['features'] = features
                    st.session_state['filename'] = "sample_malware.bin"
                    st.success("✅ Malware-like sample generated")
                else:
                    st.error("❌ Scaler not available")
        
        with col2:
            if st.button("✅ Benign Sample", use_container_width=True):
                model, scaler, device = load_model()
                if scaler:
                    features = np.random.randn(1, 2381) - 1.5
                    features = scaler.transform(features)
                    st.session_state['features'] = features
                    st.session_state['filename'] = "sample_benign.bin"
                    st.success("✅ Benign-like sample generated")
                else:
                    st.error("❌ Scaler not available")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if st.button("🚀 ANALYZE", type="primary", use_container_width=True):
        if 'features' not in st.session_state:
            st.error("❌ No file loaded. Upload or generate a sample first.")
        else:
            with st.spinner("🔍 Analyzing with Deep Learning..."):
                model, scaler, device = load_model()
                if model:
                    features = st.session_state['features']
                    filename = st.session_state.get('filename', 'unknown')
                    
                    try:
                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(features).to(device)
                            output = model(X_tensor)
                            prob = torch.sigmoid(output).item()
                        
                        risk_level, risk_class, _ = get_risk_level(prob)
                        
                        # Save to history
                        save_to_history(filename, prob, risk_level)
                        
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("📄 File", filename[:30])
                        
                        with col2:
                            st.metric("🎯 Detection", f"{prob:.1%}")
                        
                        with col3:
                            st.metric("⚠️ Risk Level", risk_level)
                        
                        progress_val = min(prob, 1.0)
                        st.progress(progress_val)
                        
                        if prob > 0.5:
                            st.error("🚨 THREAT DETECTED - Malware pattern identified")
                        else:
                            st.success("✅ ANALYSIS COMPLETE - No malware detected")
                        
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        
                        st.markdown("### 🔍 Suspicious Features")
                        top_indices = np.argsort(np.abs(features[0]))[-10:][::-1]
                        
                        feature_data = []
                        for i, idx in enumerate(top_indices):
                            feature_data.append({
                                "Rank": i + 1,
                                "Feature ID": f"F-{idx:04d}",
                                "Value": f"{features[0][idx]:+.4f}",
                                "Abs Value": f"{abs(features[0][idx]):.4f}"
                            })
                        
                        st.dataframe(
                            feature_data,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        
                        st.markdown("### 🤖 AI Analysis")
                        with st.spinner("🧠 Consulting AI..."):
                            prompt = f"""You are a cybersecurity expert. Analyze this malware detection result:

File: {filename}
Malware Probability: {prob:.1%}
Risk Level: {risk_level}

Top suspicious features:
{chr(10).join([f"- Feature {idx}: {features[0][idx]:.4f}" for idx in top_indices[:5]])}

Provide:
1. Brief technical analysis (2-3 sentences)
2. Top 3 security recommendations
3. Potential attack vectors or benign explanation

Keep response concise and technical. Use bullet points."""
                            
                            ai_response = call_ollama(prompt)
                            st.markdown(ai_response)
                        
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        
                        st.markdown("### 📋 Recommendations")
                        if prob > 0.5:
                            st.markdown("""
                            - 🔴 **Isolate** affected system immediately from network
                            - 🔍 **Scan** with comprehensive antivirus tools
                            - 📊 **Check** system logs for suspicious activity
                            - 💾 **Consider** system restoration or rebuild
                            - 🚨 **Alert** security team if in corporate environment
                            - 🔐 **Change** all passwords from a clean system
                            """)
                        else:
                            st.markdown("""
                            - ✅ **Status**: File appears safe based on analysis
                            - 🔄 **Verify**: Perform secondary validation with alternate tools
                            - 👁️ **Monitor**: Watch system for unusual behavior
                            - 🔄 **Update**: Keep security tools and definitions current
                            - 📝 **Document**: Log analysis results for future reference
                            """)
                    
                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
                else:
                    st.error("❌ Model not loaded. Check system status in sidebar.")

# ====================================================================
# PAGE: INTELLIGENCE
# ====================================================================

elif page == "Intelligence":
    st.title("🧠 Threat Intelligence Chat")
    st.markdown("Ask questions about malware, threats, and security")
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Display chat history
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.chat_message("user").write(msg['content'])
        else:
            st.chat_message("assistant").write(msg['content'])
    
    # Chat input
    user_input = st.chat_input("Ask about malware, threats, security tactics...")
    
    if user_input:
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
        st.chat_message("user").write(user_input)
        
        with st.spinner("🔍 Consulting threat intelligence..."):
            prompt = f"""You are a cybersecurity expert specializing in malware analysis and threat intelligence. Answer this question:

{user_input}

Provide technical, actionable insights. Reference MITRE ATT&CK tactics/techniques when relevant. Use bullet points for clarity. Keep response focused and practical."""
            
            response = call_ollama(prompt)
            st.session_state['chat_history'].append({'role': 'assistant', 'content': response})
            st.chat_message("assistant").write(response)
    
    # Clear chat button
    if len(st.session_state['chat_history']) > 0:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat History"):
            st.session_state['chat_history'] = []
            st.rerun()

# ====================================================================
# PAGE: HISTORY
# ====================================================================

else:  # History
    st.title("📚 Analysis History")
    
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    
    if len(st.session_state['analysis_history']) == 0:
        st.info("📭 No analysis history yet. Your scans will appear here.")
    else:
        st.markdown(f"### Previous Scans ({len(st.session_state['analysis_history'])} total)")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        threats = sum(1 for a in st.session_state['analysis_history'] if a['probability'] > 0.5)
        safe = len(st.session_state['analysis_history']) - threats
        avg_prob = np.mean([a['probability'] for a in st.session_state['analysis_history']])
        
        with col1:
            st.metric("🦠 Threats Detected", threats)
        with col2:
            st.metric("✅ Safe Files", safe)
        with col3:
            st.metric("📊 Avg Detection", f"{avg_prob:.1%}")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Analysis table
        for i, analysis in enumerate(st.session_state['analysis_history']):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
                
                with col1:
                    icon = "🦠" if analysis['probability'] > 0.5 else "✅"
                    st.markdown(f"{icon} **{analysis['filename']}**")
                
                with col2:
                    prob = analysis['probability']
                    if prob > 0.5:
                        st.markdown(f"🔴 {prob:.1%}")
                    else:
                        st.markdown(f"🟢 {prob:.1%}")
                
                with col3:
                    st.caption(f"Risk: {analysis['risk']}")
                
                with col4:
                    st.caption(f"🕒 {analysis['timestamp']}")
                
                if i < len(st.session_state['analysis_history']) - 1:
                    st.divider()
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Clear history button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear History"):
                st.session_state['analysis_history'] = []
                st.rerun()

# ====================================================================
# FOOTER
# ====================================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 20px; color: #999;">
    <p>🛡️ VirusHunter | Ghofrane LABIDI Chokri KHEMIRA Meriem FREJ</p>
    <small>2025 | Intelligent Security Analysis</small>
</div>
""", unsafe_allow_html=True)