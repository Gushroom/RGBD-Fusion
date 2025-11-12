import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


def calculate_metrics(y_true, y_pred, class_names=None):
    """Calculate comprehensive classification metrics"""
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
    }
    
    # Per-class breakdown
    if class_names:
        per_class_metrics = []
        for i, cls in enumerate(class_names):
            per_class_metrics.append({
                'class': cls,
                'precision': precision_per_class[i] * 100,
                'recall': recall_per_class[i] * 100,
                'f1': f1_per_class[i] * 100,
                'support': int(support_per_class[i])
            })
        metrics['per_class'] = per_class_metrics
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, figsize=(15, 12)):
    """Plot and save confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()
    
    return cm, cm_normalized


def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report"""
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0
    )
    
    print(report)
    print("="*80 + "\n")


def analyze_mistakes(y_true, y_pred, class_names, top_k=10):
    """Analyze most common classification mistakes"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Find most common mistakes (off-diagonal elements)
    mistakes = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                mistakes.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i, j],
                    'percentage': cm[i, j] / cm[i].sum() * 100
                })
    
    # Sort by count
    mistakes = sorted(mistakes, key=lambda x: x['count'], reverse=True)
    
    print("\n" + "="*80)
    print(f"TOP {top_k} CLASSIFICATION MISTAKES")
    print("="*80)
    print(f"{'True Label':<20} {'Predicted As':<20} {'Count':<10} {'%':<10}")
    print("-"*80)
    
    for mistake in mistakes[:top_k]:
        print(f"{mistake['true']:<20} {mistake['pred']:<20} "
              f"{mistake['count']:<10} {mistake['percentage']:.1f}%")
    
    print("="*80 + "\n")
    
    return mistakes


def plot_training_curves(history, save_path=None):
    """Plot training and validation curves"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()


def compare_modalities(results_dict, save_path=None):
    """Compare results across different modalities"""
    
    modalities = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] for m in modalities]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(modalities, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()


# Example usage
if __name__ == '__main__':
    # Dummy data for testing
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    
    class_names = [f'Class_{i}' for i in range(n_classes)]
    
    # Test metrics
    metrics = calculate_metrics(y_true, y_pred, class_names)
    print("Metrics:", metrics)
    
    # Test confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, 'test_cm.png')
    
    # Test mistake analysis
    analyze_mistakes(y_true, y_pred, class_names)