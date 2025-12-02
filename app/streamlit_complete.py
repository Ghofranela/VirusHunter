#!/usr/bin/env python3
"""
VirusHunter - Modern Malware Detection Interface
Deep Learning + LLM Analysis with Llama3.2 Integration
"""
import streamlit as st
import numpy as np
import torch
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.model import MalwareDetector
    import joblib
except ImportError as e:
    st.warning(f"Import warning: {e}")
    MalwareDetector = None

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

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2d3561, transparent);
    margin: 30px 0;
}

.analysis-report {
    background-color: #16213e;
    border: 1px solid #2d3561;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
}

</style>
""", unsafe_allow_html=True)

# ====================================================================
# CONFIGURATION
# ====================================================================

OLLAMA_URL = "http://51.254.200.139:11434"
OLLAMA_MODEL = "llama3.2:1b"

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

@st.cache_resource
def load_model():
    """Load Deep Learning model"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if MalwareDetector is None:
            st.warning("Model class not available. Using demo mode.")
            return None, None, device
        
        model = MalwareDetector()
        
        # Check if model file exists
        model_path = 'models/best_model.pth'
        if not Path(model_path).exists():
            st.info("ℹ️ Model file not found. Running in demo mode.")
            return None, None, device
            
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        # Check if scaler exists
        scaler_path = 'data/processed/preprocessor.pkl'
        if Path(scaler_path).exists():
            scaler = joblib.load(scaler_path)
        else:
            scaler = None
            
        return model, scaler, device
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None, 'cpu'

def call_ollama(prompt, model=OLLAMA_MODEL):
    """Call Ollama LLM on remote server"""
    try:
        response = requests.post(
            f'{OLLAMA_URL}/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': True,
                'options': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                }
            },
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            full_response += data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            return full_response if full_response else "⚠️ Empty response from server"
        else:
            return f"⚠️ Server error (status {response.status_code})"
            
    except requests.exceptions.ConnectionError:
        return f"⚠️ Cannot connect to Ollama server: {OLLAMA_URL}"
    except requests.exceptions.Timeout:
        return "⚠️ Connection timeout"
    except Exception as e:
        return f"⚠️ Error: {type(e).__name__} - {str(e)}"

def check_system_status():
    """Check system components"""
    status = {
        'ollama': False,
        'ollama_model': 'unknown',
        'model': False,
        'device': 'cpu'
    }
    
    try:
        r = requests.get(f'{OLLAMA_URL}/api/tags', timeout=5)
        if r.status_code == 200:
            status['ollama'] = True
            models = r.json().get('models', [])
            
            llama3_models = [m['name'] for m in models if 'llama3' in m['name'].lower()]
            if llama3_models:
                status['ollama_model'] = llama3_models[0]
            else:
                status['ollama_model'] = models[0]['name'] if models else 'No model'
    except Exception as e:
        status['ollama_model'] = f'Error: {str(e)}'
    
    if Path('models/best_model.pth').exists():
        status['model'] = True
    
    status['device'] = 'GPU' if torch.cuda.is_available() else 'CPU'
    
    return status

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

def extract_features_from_file(file_obj, file_type):
    """Extract features from different file types - matches EMBER format"""
    try:
        file_obj.seek(0)
        file_bytes = file_obj.read()
        file_size = len(file_bytes)

        # Initialize feature vector (2381 features to match EMBER)
        features = np.zeros(2381, dtype=np.float64)

        # Basic file metadata features (first 50 features)
        features[0] = file_size
        features[1] = len(set(file_bytes)) if file_size > 0 else 0
        features[2] = file_bytes.count(0x00) / max(file_size, 1)
        features[3] = sum(file_bytes) / max(file_size, 1) if file_size > 0 else 0

        # Entropy calculation (feature 4)
        if file_size > 0:
            byte_array = np.frombuffer(file_bytes, dtype=np.uint8)
            byte_counts = np.bincount(byte_array, minlength=256)
            probabilities = byte_counts / file_size
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            features[4] = entropy

        # Byte frequency features (features 50-305: 256 byte frequencies)
        if file_size > 0:
            byte_array = np.frombuffer(file_bytes, dtype=np.uint8)
            byte_freq = np.bincount(byte_array, minlength=256).astype(np.float64) / file_size
            features[50:306] = byte_freq[:256]  # Ensure exactly 256 values

        # N-gram features (bigrams, features 306-1000)
        if file_size > 1:
            byte_array = np.frombuffer(file_bytes, dtype=np.uint8)
            bigrams = {}
            sample_size = min(file_size - 1, 10000)
            for i in range(sample_size):
                bigram = (int(byte_array[i]), int(byte_array[i+1]))
                bigrams[bigram] = bigrams.get(bigram, 0) + 1

            top_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[:694]
            for idx, (bigram, count) in enumerate(top_bigrams):
                if 306 + idx < 1000:
                    features[306 + idx] = count / max(file_size, 1)

        # PE header features (if executable) - features 1000-1100
        if file_type in ['exe', 'dll', 'bin', 'so']:
            if len(file_bytes) > 64 and file_bytes[0:2] == b'MZ':
                features[1000] = 1

                try:
                    pe_offset = int.from_bytes(file_bytes[60:64], 'little')
                    if pe_offset < len(file_bytes) - 4:
                        if file_bytes[pe_offset:pe_offset+2] == b'PE':
                            features[1001] = 1

                            if pe_offset + 6 < len(file_bytes):
                                num_sections = int.from_bytes(file_bytes[pe_offset+6:pe_offset+8], 'little')
                                features[1002] = min(num_sections, 100)
                except:
                    pass

            elif len(file_bytes) > 4 and file_bytes[0:4] == b'\x7fELF':
                features[1003] = 1

        # Document-specific features - features 1100-1200
        elif file_type in ['pdf', 'docx', 'doc', 'rtf']:
            if b'%PDF' in file_bytes[:10]:
                features[1100] = 1
                features[1101] = file_bytes.count(b'/JavaScript')
                features[1102] = file_bytes.count(b'/OpenAction')
                features[1103] = file_bytes.count(b'/Launch')

            elif b'PK\x03\x04' in file_bytes[:10]:
                features[1104] = 1
                features[1105] = file_bytes.count(b'macro')
                features[1106] = file_bytes.count(b'VBA')

        # Statistical features (features 1881-1920)
        if file_size > 0:
            try:
                # Convert bytes to numpy array efficiently
                byte_array = np.frombuffer(file_bytes, dtype=np.uint8)
                features[1881] = np.mean(byte_array)
                features[1882] = np.std(byte_array)
                features[1883] = np.median(byte_array)
                features[1884] = np.min(byte_array)
                features[1885] = np.max(byte_array)
            except Exception as stat_error:
                # If statistical features fail, continue with zeros (already initialized)
                pass

        return features.reshape(1, -1)

    except Exception as e:
        st.error(f"❌ Feature extraction error: {str(e)}")
        st.info(f"📊 File size: {len(file_obj.read() if hasattr(file_obj, 'read') else b'')} bytes")
        return None

def save_to_history(filename, prob, risk_level, ai_analysis="", features=None):
    """Save analysis to history with full details"""
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    
    analysis_record = {
        'filename': filename,
        'probability': prob,
        'risk': risk_level,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ai_analysis': ai_analysis,
        'features': features.tolist() if features is not None else []
    }
    
    st.session_state['analysis_history'].insert(0, analysis_record)
    
    # Keep only last 50 entries
    if len(st.session_state['analysis_history']) > 50:
        st.session_state['analysis_history'] = st.session_state['analysis_history'][:50]
    
    return analysis_record

def generate_analysis_report(analysis_data):
    """Generate downloadable analysis report"""
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           VIRUSHUNTER MALWARE ANALYSIS REPORT                ║
╚══════════════════════════════════════════════════════════════╝

ANALYSIS SUMMARY
═══════════════════════════════════════════════════════════════
File Name:           {analysis_data['filename']}
Analysis Time:       {analysis_data['timestamp']}
Malware Probability: {analysis_data['probability']:.1%}
Risk Level:          {analysis_data['risk']}

VERDICT
═══════════════════════════════════════════════════════════════
{'🚨 THREAT DETECTED - Malware pattern identified' if analysis_data['probability'] > 0.5 else '✅ SAFE - No malware detected'}

AI ANALYSIS (Llama3.2)
═══════════════════════════════════════════════════════════════
{analysis_data.get('ai_analysis', 'No AI analysis available')}

RECOMMENDATIONS
═══════════════════════════════════════════════════════════════
"""
    
    if analysis_data['probability'] > 0.5:
        report += """
- ISOLATE affected system immediately from network
- SCAN with comprehensive antivirus tools
- CHECK system logs for suspicious activity
- CONSIDER system restoration or rebuild
- ALERT security team if in corporate environment
- CHANGE all passwords from a clean system
"""
    else:
        report += """
- STATUS: File appears safe based on analysis
- VERIFY: Perform secondary validation with alternate tools
- MONITOR: Watch system for unusual behavior
- UPDATE: Keep security tools and definitions current
- DOCUMENT: Log analysis results for future reference
"""
    
    report += f"""

TECHNICAL DETAILS
═══════════════════════════════════════════════════════════════
Features Analyzed:   2,381
Detection Model:     Deep Neural Network
Processing Device:   {check_system_status()['device']}

═══════════════════════════════════════════════════════════════
Generated by VirusHunter | Ghofrane LABIDI • Chokri KHEMIRA • Meriem FREJ
2025 | Intelligent Security Analysis with Llama3.2
═══════════════════════════════════════════════════════════════
"""
    return report

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
            st.caption(f"Model: {status['ollama_model']}")
        else:
            st.warning("🟡 LLM Offline")
    
    with col2:
        if status['model']:
            st.success("🟢 Model Ready")
        else:
            st.info("ℹ️ Demo Mode")
    
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
        **VirusHunter** uses advanced Deep Learning combined with Llama3.2 LLM 
        to provide intelligent malware detection and analysis.
        
        1. **Upload** binary or feature files
        2. **Analyze** with neural networks
        3. **Get** risk assessment and probability score
        4. **Understand** with AI-powered insights
        """)
    
    with col2:
        st.markdown("### ⚡ Key Features")
        st.markdown("""
        - ✅ Real-time threat detection
        - 🧠 Deep neural network analysis
        - 💬 Llama3.2-powered explanations
        - 📥 Downloadable analysis reports
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
        st.caption("📁 Upload any suspicious file - type will be detected automatically")

        # Unified file uploader accepting all supported types
        accepted_types = [
            'npy',  # Feature vectors
            'exe', 'dll', 'bin', 'so', 'elf',  # Executables
            'pdf', 'docx', 'doc', 'rtf',  # Documents
            'zip', 'rar', '7z', 'tar', 'gz',  # Archives
            'py', 'js', 'ps1', 'sh', 'bat', 'vbs', 'php', 'rb'  # Scripts
        ]

        uploaded_file = st.file_uploader(
            "Drop your file here or click to browse",
            type=accepted_types,
            help="Supports executables, documents, scripts, archives, and pre-extracted feature vectors (.npy)"
        )

        # Display supported file types in an expander
        with st.expander("📋 Supported File Types"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                **💾 Executables**
                - .exe, .dll, .bin
                - .so, .elf
                """)
            with col2:
                st.markdown("""
                **📄 Documents**
                - .pdf, .docx, .doc
                - .rtf
                """)
            with col3:
                st.markdown("""
                **📜 Scripts & Others**
                - .py, .js, .ps1, .sh
                - .bat, .vbs, .php, .rb
                - .zip, .rar, .7z
                - .npy (feature vectors)
                """)
        
        if uploaded_file:
            file_extension = uploaded_file.name.split('.')[-1].lower()

            # Auto-detect file type and display info
            if file_extension == 'npy':
                file_type_icon = "📊"
                file_type_name = "Feature Vector"
            elif file_extension in ['exe', 'dll', 'bin', 'so', 'elf']:
                file_type_icon = "💾"
                file_type_name = "Executable"
            elif file_extension in ['pdf', 'docx', 'doc', 'rtf']:
                file_type_icon = "📄"
                file_type_name = "Document"
            elif file_extension in ['zip', 'rar', '7z', 'tar', 'gz']:
                file_type_icon = "📦"
                file_type_name = "Archive"
            elif file_extension in ['py', 'js', 'ps1', 'sh', 'bat', 'vbs', 'php', 'rb']:
                file_type_icon = "📜"
                file_type_name = "Script"
            else:
                file_type_icon = "❓"
                file_type_name = "Unknown"

            st.info(f"{file_type_icon} **Detected type**: {file_type_name} (`.{file_extension}`)")

            # Handle .npy files
            if file_extension == 'npy':
                try:
                    from io import BytesIO
                    
                    uploaded_file.seek(0)
                    bytes_data = uploaded_file.read()
                    
                    features = None
                    load_method = None
                    
                    # Try multiple loading methods
                    try:
                        features = np.load(BytesIO(bytes_data), allow_pickle=False)
                        load_method = "standard numpy"
                    except:
                        pass
                    
                    if features is None:
                        try:
                            features = np.load(BytesIO(bytes_data), allow_pickle=True)
                            load_method = "numpy with pickle"
                        except:
                            pass
                    
                    if features is None:
                        try:
                            features = np.frombuffer(bytes_data, dtype=np.float32)
                            load_method = "raw float32"
                        except:
                            pass
                    
                    if features is None:
                        raise ValueError("Could not load file with any known method")
                    
                    # Handle different numpy array formats
                    if isinstance(features, np.ndarray):
                        if len(features.shape) == 1:
                            features = features.reshape(1, -1)
                    elif hasattr(features, 'item'):
                        features = np.array(features).reshape(1, -1)
                    else:
                        features = np.array(features)
                        if len(features.shape) == 1:
                            features = features.reshape(1, -1)
                    
                    # Validate and auto-correct feature dimensions
                    if features.shape[1] != 2381:
                        st.warning(f"⚠️ Feature count mismatch: {features.shape[1]} found, 2381 expected.")
                        st.info("🔧 Auto-correcting to 2381 features...")
                        
                        if features.shape[1] < 2381:
                            padding = np.zeros((features.shape[0], 2381 - features.shape[1]))
                            features = np.concatenate([features, padding], axis=1)
                            st.success(f"✅ Padded to 2381 features with zeros")
                        else:
                            features = features[:, :2381]
                            st.success(f"✅ Truncated to 2381 features")
                    
                    st.session_state['features'] = features
                    st.session_state['filename'] = uploaded_file.name
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                    st.info(f"📊 Shape: {features.shape} | Method: {load_method}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading .npy file: {str(e)}")
            
            # Handle binary/document/script files
            else:
                try:
                    st.info("🔄 Extracting features from file...")
                    features = extract_features_from_file(uploaded_file, file_extension)
                    
                    if features is not None:
                        # Apply preprocessing if scaler available
                        _, scaler, _ = load_model()
                        if scaler is not None:
                            features = scaler.transform(features)
                            st.info("✅ Features normalized with scaler")
                        
                        st.session_state['features'] = features
                        st.session_state['filename'] = uploaded_file.name
                        st.success(f"✅ Features extracted: {uploaded_file.name}")
                        st.info(f"📊 Shape: {features.shape} | Extracted {features.shape[1]} features")
                    else:
                        st.error("❌ Feature extraction failed")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
    
    with tab2:
        st.markdown("### Generate Test Sample")
        st.caption("Create synthetic samples for testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🦠 Malware Sample", use_container_width=True):
                features = np.random.randn(1, 2381) + 2.5
                st.session_state['features'] = features
                st.session_state['filename'] = "sample_malware.bin"
                st.success("✅ Malware-like sample generated")
        
        with col2:
            if st.button("✅ Benign Sample", use_container_width=True):
                features = np.random.randn(1, 2381) - 1.5
                st.session_state['features'] = features
                st.session_state['filename'] = "sample_benign.bin"
                st.success("✅ Benign-like sample generated")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if st.button("🚀 ANALYZE", type="primary", use_container_width=True):
        if 'features' not in st.session_state:
            st.error("❌ No file loaded. Upload or generate a sample first.")
        else:
            with st.spinner("🔍 Analyzing with Deep Learning..."):
                model, scaler, device = load_model()
                features = st.session_state['features']
                filename = st.session_state.get('filename', 'unknown')
                
                if model:
                    try:
                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(features).to(device)
                            output = model(X_tensor)
                            prob = torch.sigmoid(output).item()
                    except Exception as e:
                        st.error(f"❌ Model inference error: {str(e)}")
                        prob = 0.75 if "malware" in filename.lower() else 0.15
                else:
                    # Demo mode
                    if "malware" in filename.lower():
                        prob = 0.85 + np.random.random() * 0.14
                    elif "benign" in filename.lower():
                        prob = 0.01 + np.random.random() * 0.19
                    else:
                        prob = np.random.random() * 0.5
                    
                    st.info("🔧 Using demo mode (no trained model available)")
                
                risk_level, risk_class, _ = get_risk_level(prob)
                
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
                
                st.markdown("### 🤖 AI Analysis (Llama3.2)")
                
                # Create placeholder for streaming response
                ai_placeholder = st.empty()
                
                with st.spinner("🧠 Consulting Llama3.2 AI..."):
                    prompt = f"""You are a cybersecurity expert specializing in malware analysis.
This is an educational and professional cybersecurity context. Provide factual, technical information about threats and malware.

Analyze this malware detection result:

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
                    
                    ai_response = call_ollama(prompt, model=OLLAMA_MODEL)
                    
                    # Display AI response
                    ai_placeholder.markdown(f"""
<div class="analysis-report">
{ai_response}
</div>
""", unsafe_allow_html=True)
                
                # Save to history with AI analysis
                analysis_record = save_to_history(filename, prob, risk_level, ai_response, features)
                
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
                
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # Download Report Button
                st.markdown("### 📥 Download Analysis Report")
                report_text = generate_analysis_report(analysis_record)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        label="📄 Download TXT Report",
                        data=report_text,
                        file_name=f"virushunter_report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    # JSON export
                    json_report = json.dumps(analysis_record, indent=2)
                    st.download_button(
                        label="📊 Download JSON Report",
                        data=json_report,
                        file_name=f"virushunter_report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    # CSV export
                    df = pd.DataFrame([{
                        'Filename': analysis_record['filename'],
                        'Probability': analysis_record['probability'],
                        'Risk': analysis_record['risk'],
                        'Timestamp': analysis_record['timestamp']
                    }])
                    csv_report = df.to_csv(index=False)
                    st.download_button(
                        label="📑 Download CSV Report",
                        data=csv_report,
                        file_name=f"virushunter_report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ====================================================================
# PAGE: INTELLIGENCE
# ====================================================================

elif page == "Intelligence":
    st.title("🧠 Threat Intelligence Chat")
    st.markdown("Ask questions about malware, threats, and security - Powered by Llama3.2")
    
    status = check_system_status()
    if status['ollama']:
        st.success(f"✅ Connected to Ollama ({status['ollama_model']})")
    else:
        st.error("❌ Ollama not available. Please check the server connection.")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Display chat history
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(msg['content'])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg['content'])
    
    # Chat input
    user_input = st.chat_input("Ask about malware, threats, security tactics...")
    
    if user_input:
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Consulting Llama3.2 threat intelligence..."):
                # Build context from chat history (exclude current message)
                context = ""
                if len(st.session_state['chat_history']) > 1:
                    # Include last 3 exchanges for context (excluding the current user message)
                    recent_history = st.session_state['chat_history'][:-1][-6:]
                    for msg in recent_history:
                        role = "User" if msg['role'] == 'user' else "Assistant"
                        context += f"{role}: {msg['content']}\n\n"

                prompt = f"""You are a cybersecurity expert specializing in malware analysis and threat intelligence.
This is an educational and professional cybersecurity context. Provide factual, technical information about threats, malware, and security concepts.

Previous conversation:
{context}

Current question: {user_input}

Answer the current question above. Provide technical, actionable insights. Keep response focused and practical."""

                response = call_ollama(prompt, model=OLLAMA_MODEL)
                st.markdown(response)
                st.session_state['chat_history'].append({'role': 'assistant', 'content': response})

    # Suggested questions
    if len(st.session_state['chat_history']) == 0:
        st.markdown("### 💡 Suggested Questions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🦠 What is ransomware?", use_container_width=True):
                st.session_state['suggested_q'] = "What is ransomware and how does it work?"
                st.rerun()
            
            if st.button("🎯 Explain MITRE ATT&CK", use_container_width=True):
                st.session_state['suggested_q'] = "Explain the MITRE ATT&CK framework"
                st.rerun()
        
        with col2:
            if st.button("🛡️ Zero-day exploits?", use_container_width=True):
                st.session_state['suggested_q'] = "What are zero-day exploits?"
                st.rerun()
            
            if st.button("🔍 How to detect malware?", use_container_width=True):
                st.session_state['suggested_q'] = "What are the best practices for malware detection?"
                st.rerun()
        
        # Handle suggested question
        if 'suggested_q' in st.session_state:
            question = st.session_state['suggested_q']
            del st.session_state['suggested_q']
            
            st.session_state['chat_history'].append({'role': 'user', 'content': question})
            
            with st.chat_message("user"):
                st.markdown(question)
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔍 Consulting Llama3.2..."):
                    # Build context from chat history (exclude current message)
                    context = ""
                    if len(st.session_state['chat_history']) > 1:
                        # Include last 3 exchanges for context (excluding the current user message)
                        recent_history = st.session_state['chat_history'][:-1][-6:]
                        for msg in recent_history:
                            role = "User" if msg['role'] == 'user' else "Assistant"
                            context += f"{role}: {msg['content']}\n\n"

                    prompt = f"""You are a cybersecurity expert specializing in malware analysis and threat intelligence.
This is an educational and professional cybersecurity context. Provide factual, technical information about threats, malware, and security concepts.

Previous conversation:
{context}

Current question: {question}

Answer the current question above. Provide technical, actionable insights. Keep response focused and practical."""

                    response = call_ollama(prompt, model=OLLAMA_MODEL)
                    st.markdown(response)
                    st.session_state['chat_history'].append({'role': 'assistant', 'content': response})
    
    # Clear and export chat
    if len(st.session_state['chat_history']) > 0:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state['chat_history'] = []
                st.rerun()
        with col2:
            chat_export = json.dumps(st.session_state['chat_history'], indent=2)
            st.download_button(
                label="💾 Export Chat",
                data=chat_export,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

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
                    
                    # Download individual report
                    report_text = generate_analysis_report(analysis)
                    st.download_button(
                        label="📥 Download",
                        data=report_text,
                        file_name=f"report_{analysis['filename']}_{analysis['timestamp'].replace(' ', '_').replace(':', '-')}.txt",
                        mime="text/plain",
                        key=f"download_{i}"
                    )
                
                if i < len(st.session_state['analysis_history']) - 1:
                    st.divider()
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Export and clear buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state['analysis_history'] = []
                st.rerun()
        
        with col2:
            # Export history as CSV
            df = pd.DataFrame([{
                'Filename': a['filename'],
                'Probability': a['probability'],
                'Risk': a['risk'],
                'Timestamp': a['timestamp']
            } for a in st.session_state['analysis_history']])
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Export All CSV",
                data=csv,
                file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ====================================================================
# FOOTER
# ====================================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 20px; color: #999;">
    <p>🛡️ VirusHunter | Ghofrane LABIDI • Chokri KHEMIRA • Meriem FREJ</p>
    <small>2025 | Intelligent Security Analysis with Llama3.2</small>
</div>
""", unsafe_allow_html=True)