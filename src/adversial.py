"""
VirusHunter - Advanced Malware Detection System
Combining Deep Learning with LLM Analysis
"""

__version__ = "1.0.0"
__author__ = "Ghofrane LABIDI, Chokri KHEMIRA, Meriem FREJ"

# Import core modules
try:
    from .model import (
        MalwareDetector,
        DNNDetector,
        create_model,
        count_parameters
    )
    
    from .preprocessing import (
        EMBERPreprocessor,
        AdversarialAugmenter,
        FeatureAnalyzer,
        prepare_malware_dataset
    )
    
    from .training import (
        MalwareTrainer,
        train_malware_detector,
        evaluate_model
    )
    
    from .evaluation import (
        calculate_metrics,
        plot_confusion_matrix,
        plot_roc_curve,
        generate_classification_report
    )
    
    from .utils import (
        setup_logging,
        set_seed,
        get_device,
        save_checkpoint,
        load_checkpoint,
        EarlyStopping
    )
    
    # Optional imports
    try:
        from .data_generator import MalwareDataGenerator
    except ImportError:
        print("Warning: data_generator not available")
        MalwareDataGenerator = None
        
    try:
        from .adversarial import AdversarialAttack, AdversarialTrainer, AdversarialDefense
    except ImportError:
        print("Warning: adversarial module not available")
        AdversarialAttack = AdversarialTrainer = AdversarialDefense = None
        
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")

# Define __all__ for clean imports
__all__ = [
    # Models
    'MalwareDetector',
    'DNNDetector',
    'create_model',
    'count_parameters',
    
    # Preprocessing
    'EMBERPreprocessor',
    'AdversarialAugmenter',
    'FeatureAnalyzer',
    'prepare_malware_dataset',
    
    # Training
    'MalwareTrainer',
    'train_malware_detector',
    'evaluate_model',
    
    # Evaluation
    'calculate_metrics',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'generate_classification_report',
    
    # Utils
    'setup_logging',
    'set_seed',
    'get_device',
    'save_checkpoint',
    'load_checkpoint',
    'EarlyStopping',
    
    # Optional modules
    'MalwareDataGenerator',
    'AdversarialAttack', 
    'AdversarialTrainer',
    'AdversarialDefense'
]