"""
Explainable AI Module for Malware Detection
Implements SHAP, LIME, and Integrated Gradients
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class IntegratedGradients:
    """
    Integrated Gradients for feature attribution
    """
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize Integrated Gradients
        
        Args:
            model: PyTorch model
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def compute_gradients(
        self,
        inputs: torch.Tensor,
        target_class: int = 1
    ) -> torch.Tensor:
        """Compute gradients with respect to inputs"""
        inputs.requires_grad = True
        
        outputs = self.model(inputs)
        
        # For binary classification
        if target_class == 1:
            score = outputs
        else:
            score = -outputs
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        return inputs.grad
    
    def integrated_gradients(
        self,
        inputs: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> np.ndarray:
        """
        Compute integrated gradients
        
        Args:
            inputs: Input features
            baseline: Baseline (default: zeros)
            steps: Number of integration steps
        
        Returns:
            Feature attributions
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        path_gradients = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (inputs - baseline)
            gradients = self.compute_gradients(interpolated)
            path_gradients.append(gradients)
        
        # Average gradients
        avg_gradients = torch.stack(path_gradients).mean(dim=0)
        
        # Integrated gradients
        integrated_grads = (inputs - baseline) * avg_gradients
        
        return integrated_grads.detach().cpu().numpy()
    
    def explain_prediction(
        self,
        features: np.ndarray,
        feature_names: List[str] = None,
        top_k: int = 20
    ) -> Dict:
        """
        Explain a single prediction
        
        Args:
            features: Input features
            feature_names: Names of features
            top_k: Number of top features to return
        
        Returns:
            Explanation dictionary
        """
        # Convert to tensor
        inputs = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Compute attribution
        attributions = self.integrated_gradients(inputs)[0]
        
        # Get prediction
        with torch.no_grad():
            output = self.model(inputs)
            prob = torch.sigmoid(output).item()
        
        # Sort by absolute attribution
        indices = np.argsort(np.abs(attributions))[::-1][:top_k]
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(features))]
        
        explanation = {
            'prediction': prob,
            'predicted_class': 'Malware' if prob > 0.5 else 'Benign',
            'top_features': [
                {
                    'name': feature_names[i] if i < len(feature_names) else f"Feature_{i}",
                    'value': float(features[i]),
                    'attribution': float(attributions[i]),
                    'importance': float(np.abs(attributions[i]))
                }
                for i in indices
            ]
        }
        
        return explanation


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model interpretation
    """
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize SHAP explainer
        
        Args:
            model: PyTorch model
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prediction function for SHAP"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        return probs
    
    def explain_with_shap(
        self,
        features: np.ndarray,
        background_data: np.ndarray,
        max_samples: int = 100
    ) -> Dict:
        """
        Explain using SHAP
        
        Args:
            features: Input to explain
            background_data: Background dataset
            max_samples: Maximum background samples
        
        Returns:
            SHAP values and explanation
        """
        try:
            import shap
            
            # Sample background data
            if len(background_data) > max_samples:
                indices = np.random.choice(
                    len(background_data),
                    max_samples,
                    replace=False
                )
                background_data = background_data[indices]
            
            # Create explainer
            explainer = shap.KernelExplainer(
                self.predict_proba,
                background_data
            )
            
            # Compute SHAP values
            shap_values = explainer.shap_values(features.reshape(1, -1))
            
            # Get prediction
            prediction = self.predict_proba(features.reshape(1, -1))[0]
            
            explanation = {
                'prediction': float(prediction),
                'predicted_class': 'Malware' if prediction > 0.5 else 'Benign',
                'shap_values': shap_values[0].tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                'base_value': float(explainer.expected_value)
            }
            
            return explanation
        
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
            return None


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations)
    """
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize LIME explainer
        
        Args:
            model: PyTorch model
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prediction function for LIME"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        # LIME expects [prob_class_0, prob_class_1]
        return np.column_stack([1 - probs, probs])
    
    def explain_with_lime(
        self,
        features: np.ndarray,
        num_features: int = 20,
        num_samples: int = 5000
    ) -> Dict:
        """
        Explain using LIME
        
        Args:
            features: Input to explain
            num_features: Number of top features
            num_samples: Number of perturbed samples
        
        Returns:
            LIME explanation
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer
            
            # Create explainer
            explainer = LimeTabularExplainer(
                training_data=features.reshape(1, -1),
                mode='classification',
                feature_names=[f"Feature_{i}" for i in range(len(features))]
            )
            
            # Explain instance
            exp = explainer.explain_instance(
                data_row=features,
                predict_fn=self.predict_proba,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Get prediction
            prediction = self.predict_proba(features.reshape(1, -1))[0, 1]
            
            explanation = {
                'prediction': float(prediction),
                'predicted_class': 'Malware' if prediction > 0.5 else 'Benign',
                'top_features': [
                    {
                        'name': name,
                        'weight': float(weight)
                    }
                    for name, weight in exp.as_list()
                ]
            }
            
            return explanation
        
        except ImportError:
            print("LIME not installed. Install with: pip install lime")
            return None


class ExplainabilityDashboard:
    """
    Comprehensive explainability dashboard
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        feature_names: List[str] = None
    ):
        """
        Initialize dashboard
        
        Args:
            model: PyTorch model
            device: Device to run on
            feature_names: Names of features
        """
        self.model = model
        self.device = device
        self.feature_names = feature_names
        
        # Initialize explainers
        self.ig_explainer = IntegratedGradients(model, device)
        self.shap_explainer = SHAPExplainer(model, device)
        self.lime_explainer = LIMEExplainer(model, device)
    
    def explain(
        self,
        features: np.ndarray,
        methods: List[str] = ['ig', 'shap', 'lime'],
        background_data: np.ndarray = None
    ) -> Dict:
        """
        Generate comprehensive explanation
        
        Args:
            features: Input features
            methods: List of explanation methods
            background_data: Background data for SHAP
        
        Returns:
            Complete explanation
        """
        explanations = {}
        
        # Integrated Gradients
        if 'ig' in methods:
            print("Computing Integrated Gradients...")
            explanations['integrated_gradients'] = self.ig_explainer.explain_prediction(
                features,
                feature_names=self.feature_names
            )
        
        # SHAP
        if 'shap' in methods and background_data is not None:
            print("Computing SHAP values...")
            explanations['shap'] = self.shap_explainer.explain_with_shap(
                features,
                background_data
            )
        
        # LIME
        if 'lime' in methods:
            print("Computing LIME explanation...")
            explanations['lime'] = self.lime_explainer.explain_with_lime(features)
        
        return explanations
    
    def visualize_explanation(
        self,
        explanation: Dict,
        save_path: str = None,
        method: str = 'ig'
    ):
        """
        Visualize explanation
        
        Args:
            explanation: Explanation dictionary
            save_path: Path to save plot
            method: Explanation method
        """
        if method == 'ig' and 'integrated_gradients' in explanation:
            self._plot_integrated_gradients(
                explanation['integrated_gradients'],
                save_path
            )
        
        elif method == 'lime' and 'lime' in explanation:
            self._plot_lime(explanation['lime'], save_path)
    
    def _plot_integrated_gradients(
        self,
        explanation: Dict,
        save_path: str = None
    ):
        """Plot Integrated Gradients explanation"""
        top_features = explanation['top_features'][:20]
        
        names = [f['name'] for f in top_features]
        attributions = [f['attribution'] for f in top_features]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['red' if a < 0 else 'green' for a in attributions]
        y_pos = np.arange(len(names))
        
        ax.barh(y_pos, attributions, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Attribution Score', fontsize=12)
        ax.set_title(
            f'Top 20 Features - Integrated Gradients\n'
            f'Prediction: {explanation["predicted_class"]} '
            f'({explanation["prediction"]:.2%})',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_lime(self, explanation: Dict, save_path: str = None):
        """Plot LIME explanation"""
        top_features = explanation['top_features'][:20]
        
        names = [f['name'] for f in top_features]
        weights = [f['weight'] for f in top_features]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['red' if w < 0 else 'green' for w in weights]
        y_pos = np.arange(len(names))
        
        ax.barh(y_pos, weights, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('LIME Weight', fontsize=12)
        ax.set_title(
            f'Top 20 Features - LIME\n'
            f'Prediction: {explanation["predicted_class"]} '
            f'({explanation["prediction"]:.2%})',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def generate_explanation_report(
    model: nn.Module,
    features: np.ndarray,
    background_data: np.ndarray = None,
    device: str = 'cuda',
    save_dir: str = "results/explanations"
) -> Dict:
    """
    Generate comprehensive explanation report
    
    Args:
        model: Trained model
        features: Input to explain
        background_data: Background data for SHAP
        device: Device to use
        save_dir: Directory to save results
    
    Returns:
        Complete explanation
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dashboard
    dashboard = ExplainabilityDashboard(model, device)
    
    # Generate explanations
    explanations = dashboard.explain(
        features,
        methods=['ig', 'lime'] if background_data is None else ['ig', 'shap', 'lime'],
        background_data=background_data
    )
    
    # Visualize
    if 'integrated_gradients' in explanations:
        dashboard.visualize_explanation(
            explanations,
            save_path=save_dir / "integrated_gradients.png",
            method='ig'
        )
    
    if 'lime' in explanations:
        dashboard.visualize_explanation(
            explanations,
            save_path=save_dir / "lime_explanation.png",
            method='lime'
        )
    
    # Save JSON
    import json
    with open(save_dir / "explanation.json", 'w') as f:
        json.dump(explanations, f, indent=2)
    
    print(f"\nExplanations saved to {save_dir}")
    
    return explanations


if __name__ == "__main__":
    # Test explainability
    from model import MalwareDetector
    
    print("Testing Explainability Module...")
    
    # Create dummy model and data
    model = MalwareDetector(input_size=2381)
    model.eval()
    
    features = np.random.randn(2381)
    
    # Test Integrated Gradients
    ig = IntegratedGradients(model, device='cpu')
    explanation = ig.explain_prediction(features, top_k=10)
    
    print(f"\nPrediction: {explanation['predicted_class']}")
    print(f"Probability: {explanation['prediction']:.4f}")
    print(f"\nTop 5 Features:")
    for feat in explanation['top_features'][:5]:
        print(f"  {feat['name']}: {feat['attribution']:.4f}")
    
    print("\nExplainability module tested successfully!")
