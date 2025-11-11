# ✅ VIRUSHUNTER - INSTALLATION & VERIFICATION CHECKLIST

## 📋 Pre-Installation Checklist

### System Requirements
- [ ] Python 3.8 or higher installed
- [ ] pip package manager installed
- [ ] 20GB+ free disk space
- [ ] 16GB+ RAM (32GB recommended)
- [ ] GPU with 8GB+ VRAM (optional but recommended)

### Check Python Version
```bash
python --version  # Should be 3.8+
```

---

## 🔧 Installation Steps

### Step 1: Extract Project
- [ ] Extract virushunter_complete folder
- [ ] Navigate to project directory
```bash
cd virushunter_complete
```

### Step 2: Create Virtual Environment
- [ ] Create venv
```bash
python -m venv venv
```

- [ ] Activate venv
```bash
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

- [ ] Verify activation (prompt should show `(venv)`)

### Step 3: Install Dependencies
- [ ] Update pip
```bash
pip install --upgrade pip
```

- [ ] Install requirements
```bash
pip install -r requirements.txt
```

**This will install (~15 minutes):**
- PyTorch 2.0+
- EMBER dataset tools
- SHAP, LIME
- Streamlit
- All other dependencies

### Step 4: Verify Installation
- [ ] Test imports
```bash
python -c "from src import *; print('✅ All modules imported successfully')"
```

- [ ] Test model creation
```bash
python -c "from src import create_model; m = create_model('dnn'); print('✅ Model created successfully')"
```

---

## 📂 Verify File Structure

Check that you have all these files:

### Core Source Files (src/)
- [ ] `src/__init__.py` - Module initialization
- [ ] `src/model.py` - Neural network architectures (550 lines)
- [ ] `src/preprocessing.py` - Data pipeline (450 lines)
- [ ] `src/training.py` - Training system (380 lines)
- [ ] `src/explainability.py` - XAI methods (520 lines)
- [ ] `src/evaluation.py` - Metrics & robustness (480 lines)
- [ ] `src/utils.py` - Utilities (320 lines)

### Application Files
- [ ] `main.py` - Complete pipeline (260 lines)
- [ ] `requirements.txt` - Dependencies

### Documentation
- [ ] `README.md` - Complete guide (400 lines)
- [ ] `PROJECT_SUMMARY.md` - Feature overview (350 lines)
- [ ] `QUICK_START.md` - Command reference (250 lines)
- [ ] `DELIVERY_SUMMARY.md` - Final delivery doc
- [ ] This file - Installation checklist

**Total: ~2,500+ lines of production code + 1,000+ lines of documentation**

---

## 🗂️ Create Directory Structure

Run this to create required directories:

```bash
mkdir -p data/ember data/processed models results/explanations logs
```

Verify structure:
- [ ] `data/ember/` exists
- [ ] `data/processed/` exists
- [ ] `models/` exists
- [ ] `results/` exists
- [ ] `logs/` exists

---

## 📥 Download EMBER Dataset

### Option 1: Automated (Recommended)
```bash
python main.py --mode download
```

### Option 2: Manual
1. Visit: https://github.com/elastic/ember
2. Download `ember2018_2.tar.bz2` (~13GB)
3. Extract to `data/ember/`

### Verify Download
- [ ] `data/ember/train_features_*.dat` exists
- [ ] `data/ember/test_features_*.dat` exists
- [ ] Total size ~15GB

---

## 🧪 Quick Tests

### Test 1: Import All Modules
```bash
python -c "
from src import (
    MalwareDetector, CNNMalwareDetector, LSTMMalwareDetector,
    EMBERPreprocessor, Trainer, IntegratedGradients,
    ModelEvaluator, setup_logging
)
print('✅ All imports successful')
"
```

### Test 2: Create Models
```bash
python -c "
from src import create_model
dnn = create_model('dnn', input_size=2381)
cnn = create_model('cnn', sequence_length=2381)
lstm = create_model('lstm', input_size=2381)
print('✅ All models created')
print(f'DNN parameters: {sum(p.numel() for p in dnn.parameters()):,}')
"
```

### Test 3: Check CUDA
```bash
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  CUDA not available, will use CPU')
"
```

---

## 🚀 First Run - Quick Test

### Preprocess Small Sample (5 minutes)
```bash
# This tests the preprocessing pipeline
python main.py --mode preprocess
```

**Expected output:**
```
Loading training data...
Loaded X samples with 2381 features
Preprocessing features...
After removing unlabeled: X samples
Fitting standard scaler...
Preprocessor saved to data/processed/preprocessor.pkl
✓ Preprocessing phase complete!
```

### Train for 5 Epochs (Testing - 10 minutes)
```bash
python main.py --mode train --model-type dnn --epochs 5
```

**Expected output:**
```
Using device: cuda/cpu
Model: dnn
Parameters: X,XXX,XXX
Training for 5 epochs
Epoch 1/5
Train Loss: 0.XXXX | Train Acc: 0.XXXX
Val Loss: 0.XXXX | Val Acc: 0.XXXX
...
✓ Training phase complete!
```

---

## ✅ Verification Checklist

### Installation Verified
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] All dependencies installed
- [ ] All source files present
- [ ] Directory structure created

### Functionality Verified
- [ ] Modules import successfully
- [ ] Models can be created
- [ ] CUDA detected (if GPU available)
- [ ] Preprocessing works
- [ ] Training works

### Ready for Full Run
- [ ] EMBER dataset downloaded
- [ ] At least 20GB free disk space
- [ ] System meets requirements
- [ ] No import errors
- [ ] Quick test passed

---

## 🎯 Next Steps After Verification

Once all items are checked:

### For Quick Results (2-3 hours)
```bash
python main.py --mode all --model-type dnn --epochs 20
```

### For Best Performance (4-6 hours)
```bash
python main.py --mode all --model-type dnn --adversarial --epochs 50
```

### For Maximum Accuracy (6-8 hours)
```bash
python main.py --mode all --model-type ensemble --adversarial --epochs 30
```

---

## 🐛 Troubleshooting

### Issue: Import Errors
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: CUDA Out of Memory
```bash
# Solution: Reduce batch size or use CPU
python main.py --mode train --batch-size 128
# or
python main.py --mode train --no-cuda
```

### Issue: EMBER Download Fails
```bash
# Solution: Manual download
# Visit: https://github.com/elastic/ember
# Download and extract to data/ember/
```

### Issue: Slow Training
```bash
# Solution 1: Verify GPU usage
python -c "import torch; print(torch.cuda.is_available())"

# Solution 2: Reduce epochs for testing
python main.py --mode train --epochs 5
```

---

## 📊 Expected Timeline

### With GPU (RTX 3080 or better)
- Download EMBER: 30 minutes
- Preprocessing: 1 hour
- Training (50 epochs): 2-3 hours
- Evaluation: 10 minutes
- Explanations: 5 minutes
- **Total**: 4-5 hours

### With CPU Only
- Download EMBER: 30 minutes
- Preprocessing: 2 hours
- Training (50 epochs): 8-12 hours
- Evaluation: 20 minutes
- Explanations: 10 minutes
- **Total**: 11-15 hours

---

## 🎉 Final Verification

Run this comprehensive test:

```bash
python -c "
print('='*60)
print('VIRUSHUNTER INSTALLATION VERIFICATION')
print('='*60)

# Test 1: Python version
import sys
print(f'\n✓ Python version: {sys.version.split()[0]}')

# Test 2: PyTorch
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')

# Test 3: Core modules
from src import *
print('✓ All core modules imported')

# Test 4: Model creation
model = create_model('dnn', input_size=2381)
params = sum(p.numel() for p in model.parameters())
print(f'✓ Model created with {params:,} parameters')

# Test 5: Key dependencies
import numpy, pandas, sklearn
print('✓ NumPy, Pandas, Scikit-learn available')

print('\n' + '='*60)
print('✅ INSTALLATION VERIFIED - READY TO USE')
print('='*60)
print('\nStart training with:')
print('  python main.py --mode all --model-type dnn --adversarial')
"
```

---

## ✅ Checklist Complete!

If all items are checked, you're ready to run the complete pipeline:

```bash
# Full pipeline with adversarial training
python main.py --mode all --model-type dnn --adversarial --epochs 50
```

**This will:**
1. ✅ Download EMBER dataset
2. ✅ Preprocess 1.1M samples
3. ✅ Train robust DNN model
4. ✅ Evaluate on test set
5. ✅ Generate explanations
6. ✅ Save all results

**Estimated time**: 4-6 hours with GPU, 12-15 hours with CPU

**Good luck! 🚀**