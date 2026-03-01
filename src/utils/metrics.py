# src/utils/metrics.py

import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluate recognition model performance."""

    def __init__(self, class_names: list):
        self.class_names = class_names

    def full_evaluation(self, y_true, y_pred) -> dict:
        """Run complete evaluation suite."""
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
        }

    def plot_confusion_matrix(self, y_true, y_pred,
                               save_path: str = None):
        """Generate and display confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)')

        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f',
                    cmap='RdYlGn',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()