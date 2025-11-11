"""
VirusHunter - Advanced Malware Detection with Deep Learning
Detection of Adversarial Malware Using Deep Learning on Executable Files
"""

__version__ = "1.0.0"
__author__ = "VirusHunter Team"

# Import model components
try:
    from .model import (
        MalwareDetector,
        CNNMalwareDetector,
        LSTMMalwareDetector,
        EnsembleDetector,
        AdversarialDefense,
        create_model,
        load_model
    )
except ImportError as e:
    print(f"Warning: Could not import model components: {e}")

# Import preprocessing components
try:
    from .preprocessing import (
        EMBERPreprocessor,
        AdversarialAugmenter,
        FeatureAnalyzer,
        prepare_ember_dataset
    )
except ImportError as e:
    print(f"Warning: Could not import preprocessing components: {e}")

# Import training components
try:
    from .training import (
        MalwareDataset,
        Trainer,
        train_malware_detector
    )
except ImportError as e:
    print(f"Warning: Could not import training components: {e}")

# Import explainability components
try:
    from .explainability import (
        IntegratedGradients,
        SHAPExplainer,
        LIMEExplainer,
        ExplainabilityDashboard,
        generate_explanation_report
    )
except ImportError as e:
    print(f"Warning: Could not import explainability components: {e}")

# Import evaluation components
try:
    from .evaluation import (
        ModelEvaluator,
        RobustnessEvaluator,
        comprehensive_evaluation
    )
except ImportError as e:
    print(f"Warning: Could not import evaluation components: {e}")

# Import utilities
try:
    from .utils import (
        setup_logging,
        set_seed,
        get_device,
        save_checkpoint,
        load_checkpoint,
        save_json,
        load_json,
        count_parameters,
        print_model_summary,
        create_dir_structure,
        download_ember_dataset
    )
except ImportError as e:
    print(f"Warning: Could not import utility components: {e}")

__all__ = [
    # Models
    'MalwareDetector',
    'CNNMalwareDetector',
    'LSTMMalwareDetector',
    'EnsembleDetector',
    'AdversarialDefense',
    'create_model',
    'load_model',
    
    # Preprocessing
    'EMBERPreprocessor',
    'AdversarialAugmenter',
    'FeatureAnalyzer',
    'prepare_ember_dataset',
    
    # Training
    'MalwareDataset',
    'Trainer',
    'train_malware_detector',
    
    # Explainability
    'IntegratedGradients',
    'SHAPExplainer',
    'LIMEExplainer',
    'ExplainabilityDashboard',
    'generate_explanation_report',
    
    # Evaluation
    'ModelEvaluator',
    'RobustnessEvaluator',
    'comprehensive_evaluation',
    
    # Utils
    'setup_logging',
    'set_seed',
    'get_device',
    'save_checkpoint',
    'load_checkpoint',
    'save_json',
    'load_json',
    'count_parameters',
    'print_model_summary',
    'create_dir_structure',
    'download_ember_dataset',
]
