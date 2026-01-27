"""
Fine-tuning system for Track B with proper train/test split.
Optimized for CPU training with efficient batching and reduced overhead.
"""

import sys
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.util import cos_sim
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
import json
import random
import traceback
import matplotlib.pyplot as plt
from sentence_transformers import losses
from torch.optim.lr_scheduler import ReduceLROnPlateau


def format_e5_input(texts: List[str], prefix: str = "passage: ") -> List[str]:
    """Format texts for E5 models which require specific prefixes."""
    return [prefix + text for text in texts]


def preprocess_text(text: str) -> str:
    """Light preprocessing to improve embedding quality."""
    if text is None or pd.isna(text):
        return ""
    text = str(text)  # Ensure it's a string
    text = ' '.join(text.split())
    text = text.strip()
    return text


def prepare_training_data(synthetic_path: str, dev_track_a_path: str,
                                   dev_track_b_path: str) -> List[InputExample]:
    """
    Load and combine synthetic training data with development data.
    All data used for training - no split.
    """
    print("\nPreparing training data...")

    # Load data
    """
        Load and combine synthetic training data with development data.
        All data used for training - no split.
        """
    print(f"\n{'=' * 60}")
    print("LOADING TRAINING DATA")
    print(f"{'=' * 60}")

    all_train_examples = []

    # 1. Load synthetic data
    print(f"\n1. Loading synthetic data from {synthetic_path}...")
    df_synthetic = pd.read_json(synthetic_path, lines=True)
    print(f"   Loaded {len(df_synthetic)} synthetic triples")

    # Preprocess synthetic texts
    df_synthetic["anchor_text"] = df_synthetic["anchor_text"].apply(preprocess_text)
    df_synthetic["text_a"] = df_synthetic["text_a"].apply(preprocess_text)
    df_synthetic["text_b"] = df_synthetic["text_b"].apply(preprocess_text)

    # Create training examples from synthetic data
    for _, row in df_synthetic.iterrows():
        anchor = row["anchor_text"]

        if row["text_a_is_closer"]:
            positive = row["text_a"]
            negative = row["text_b"]
        else:
            positive = row["text_b"]
            negative = row["text_a"]

        all_train_examples.append(InputExample(texts=[anchor, positive, negative]))

    synthetic_count = len(all_train_examples)
    print(f"   ✓ Created {synthetic_count} examples from synthetic data")

    # 2. Load development data
    print(f"\n2. Loading development data:")
    print(f"   Track A: {dev_track_a_path}")
    print(f"   Track B: {dev_track_b_path}")

    df_dev_labels = pd.read_json(dev_track_a_path, lines=True)
    df_dev_texts = pd.read_json(dev_track_b_path, lines=True)

    print(f"   Loaded {len(df_dev_labels)} development triples")

    # Preprocess development texts
    df_dev_labels["anchor_text"] = df_dev_labels["anchor_text"].apply(preprocess_text)
    df_dev_labels["text_a"] = df_dev_labels["text_a"].apply(preprocess_text)
    df_dev_labels["text_b"] = df_dev_labels["text_b"].apply(preprocess_text)

    # Create training examples from development data
    dev_start = len(all_train_examples)
    for _, row in df_dev_labels.iterrows():
        anchor = row["anchor_text"]

        if row["text_a_is_closer"]:
            positive = row["text_a"]
            negative = row["text_b"]
        else:
            positive = row["text_b"]
            negative = row["text_a"]

        all_train_examples.append(InputExample(texts=[anchor, positive, negative]))

    dev_count = len(all_train_examples) - dev_start
    print(f"   ✓ Created {dev_count} examples from development data")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"TOTAL TRAINING DATA:")
    print(f"  Synthetic:    {synthetic_count} examples")
    print(f"  Development:  {dev_count} examples")
    print(f"  TOTAL:        {len(all_train_examples)} examples")
    print(f"{'=' * 60}")

    return all_train_examples


def load_test_data(test_track_a_path: str, test_track_b_path: str) -> Tuple[List[InputExample], pd.DataFrame]:
    """
    Load test data from Track A and Track B test files.
    NOTE: Test data may not have labels (text_a_is_closer field).
    """
    print(f"\n{'=' * 60}")
    print("LOADING TEST DATA")
    print(f"{'=' * 60}")

    df_test_labels = pd.read_json(test_track_a_path, lines=True)
    df_test_texts = pd.read_json(test_track_b_path, lines=True)

    print(f"Loaded {len(df_test_labels)} test triples")
    print(f"Loaded {len(df_test_texts)} test stories")

    # Check if test data has labels
    has_labels = 'text_a_is_closer' in df_test_labels.columns

    if not has_labels:
        print("⚠ Test data does NOT have labels (text_a_is_closer field)")
        print("  Cannot create test examples for evaluation during training")
        print("  Will skip evaluation on test set during training")
        return [], df_test_labels

    print("✓ Test data has labels - can evaluate during training")

    # Preprocess
    df_test_labels["anchor_text"] = df_test_labels["anchor_text"].apply(preprocess_text)
    df_test_labels["text_a"] = df_test_labels["text_a"].apply(preprocess_text)
    df_test_labels["text_b"] = df_test_labels["text_b"].apply(preprocess_text)

    # Create test examples
    test_examples = []
    for _, row in df_test_labels.iterrows():
        anchor = row["anchor_text"]

        if row["text_a_is_closer"]:
            positive = row["text_a"]
            negative = row["text_b"]
        else:
            positive = row["text_b"]
            negative = row["text_a"]

        test_examples.append(InputExample(texts=[anchor, positive, negative]))

    print(f"✓ Created {len(test_examples)} test examples")
    print(f"{'=' * 60}")

    return test_examples, df_test_labels

def compute_test_loss(model: SentenceTransformer, test_data: List[InputExample],
                      margin: float, batch_size: int = 32) -> float:
    """Compute loss on test set."""
    model.eval()
    test_losses = []

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]

            anchors = [ex.texts[0] for ex in batch]
            positives = [ex.texts[1] for ex in batch]
            negatives = [ex.texts[2] for ex in batch]

            anchor_embs = model.encode(anchors, convert_to_tensor=True,
                                       normalize_embeddings=True, show_progress_bar=False)
            pos_embs = model.encode(positives, convert_to_tensor=True,
                                    normalize_embeddings=True, show_progress_bar=False)
            neg_embs = model.encode(negatives, convert_to_tensor=True,
                                    normalize_embeddings=True, show_progress_bar=False)
            # Ensure embeddings are 2D
            if anchor_embs.dim() == 1:
                anchor_embs = anchor_embs.unsqueeze(0)
            if pos_embs.dim() == 1:
                pos_embs = pos_embs.unsqueeze(0)
            if neg_embs.dim() == 1:
                neg_embs = neg_embs.unsqueeze(0)
            distance_pos = 1 - torch.sum(anchor_embs * pos_embs, dim=1)
            distance_neg = 1 - torch.sum(anchor_embs * neg_embs, dim=1)

            losses_triplet = torch.nn.functional.relu(distance_pos - distance_neg + margin)
            test_losses.append(losses_triplet.mean().item())

    model.train()
    return np.mean(test_losses)


def plot_training_progress(history: Dict, output_path: str):
    """Create training progress plots."""
    output_dir = Path(output_path).parent

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

    # 1. Train vs Test Loss
    # 4. Loss Margin (Difference between Train and Test Loss)

    # --- Compute relative generalization gap ---
    #relative_gap = [
       # (test - train) / train
        #for test, train in zip(history['test_loss'], history['train_loss'])
    #]
    relative_gap = [
        (test - train)
        for test, train in zip(history['test_loss'], history['train_loss'])
        ]

    epochs = history['epoch']

    # --- Smooth the signal (moving average) ---
    window = 3
    relative_gap_smooth = np.convolve(
        relative_gap, np.ones(window) / window, mode='same'
    )

    # --- Define meaningful overfitting threshold (5%) ---
    threshold = 0.1# changing from 0.05 to 0.1

    # --- Plot ---
    axes[0, 0].plot(
        epochs,
        relative_gap_smooth,
        color='purple',
        marker='o',
        linewidth=2,
        markersize=6,
        label='Relative Generalization Gap'
    )

    # Zero reference
    axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)

    # Threshold reference
    axes[0, 0].axhline(
        threshold,
        color='red',
        linestyle=':',
        alpha=0.7,
        label='Overfitting Threshold (5%)'
    )

    # Fill areas
    axes[0, 0].fill_between(
        epochs,
        0,
        relative_gap_smooth,
        where=[gap > threshold for gap in relative_gap_smooth],
        alpha=0.3,
        color='red',
        label='Meaningful Overfitting'
    )

    axes[0, 0].fill_between(
        epochs,
        0,
        relative_gap_smooth,
        where=[gap <= threshold for gap in relative_gap_smooth],
        alpha=0.3,
        color='green',
        label='Healthy Generalization'
    )

    # Labels
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('(Test − Train) / Train')
    axes[0, 0].set_title('Generalization Gap (Relative & Smoothed)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    if len(history['epoch']) > 1:
        z = np.polyfit(history['epoch'], [a * 100 for a in history['test_accuracy']], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(history['epoch'], p(history['epoch']),
                        "r--", alpha=0.5, label=f'Trend: {z[0]:+.2f}%/epoch')

    # 2. Test Accuracy
    axes[0, 1].plot(history['epoch'], [a * 100 for a in history['test_accuracy']], 'g-o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Test Accuracy')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Batch Losses (with moving average)
    batch_losses = history['batch_losses']
    axes[1, 0].plot(batch_losses, 'b-', alpha=0.3, linewidth=0.5)
    window = 50
    if len(batch_losses) >= window:
        moving_avg = np.convolve(batch_losses, np.ones(window) / window, mode='valid')
        axes[1, 0].plot(range(window - 1, len(batch_losses)), moving_avg, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Batch Loss (with moving average)')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Loss Progress Throughout Epochs
    axes[1, 1].plot(history['epoch'], history['train_loss'], 'b-o',
                    label='Train Loss', linewidth=2, markersize=6)
    axes[1, 1].plot(history['epoch'], history['test_loss'], 'r-o',
                    label='Test Loss', linewidth=2, markersize=6)
    axes[1, 1].fill_between(history['epoch'], history['train_loss'],
                            alpha=0.3, color='blue')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Loss Improvement Over Epochs')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / "training_progress.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Training plots saved to {plot_file}")
    plt.close()

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Loss: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f} "
          f"({(1 - history['train_loss'][-1] / history['train_loss'][0]) * 100:.1f}% improvement)")
    print(f"Accuracy: {history['test_accuracy'][0] * 100:.1f}% → {history['test_accuracy'][-1] * 100:.1f}%")
    print(f"{'=' * 60}")


class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
def fine_tune_model(model_name: str, train_data: List[InputExample],
                    test_data: List[InputExample], epochs: int = 5,
                    batch_size: int = 8,
                    learning_rate: float = 2e-5,
                    output_path: str = "output/fine_tuned_model",
                    dropout_rate: float = 0.3,
                    margin: float = 0.5):
    """
    CPU-optimized fine-tuning using manual training loop WITHOUT DataLoader.
    Processes batches manually to avoid collate_fn issues.
    """
    print(f"\nFine-tuning {model_name}...")
    print(f"Using device: CPU")

    # Set number of threads for CPU optimization
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"Using {torch.get_num_threads()} CPU threads")
    # Detect and use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nFine-tuning {model_name}...")
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("⚠ No GPU detected, using CPU")
        torch.set_num_threads(os.cpu_count() or 4)
        print(f"Using {torch.get_num_threads()} CPU threads")

    model = SentenceTransformer(model_name, device=device)
    # Enable and configure dropout for regularization
    print(f"\nConfiguring dropout regularization (rate: {dropout_rate})...")
    dropout_layers_found = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate
            dropout_layers_found += 1
    print(f"  Configured {dropout_layers_found} dropout layers")
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Add scheduler

    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=1
    )
    # Calculate training steps
    num_batches = len(train_data) // batch_size
    total_steps = num_batches * epochs
    warmup_steps = int(total_steps * 0.1)

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {num_batches}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Margin: {margin}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    history = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'batch_losses': []
    }
    model.train()
    global_step = 0
    best_test_accuracy = 0.0

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        # Shuffle training data
        shuffled_data = train_data.copy()
        random.shuffle(shuffled_data)

        total_loss = 0
        progress_bar = tqdm(range(num_batches), desc=f"Training Epoch {epoch+1}")
        epoch_losses = []  # ADD THIS
        for batch_idx in progress_bar:
            # Get batch manually (no DataLoader needed)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = shuffled_data[start_idx:end_idx]

            # Extract texts from InputExample objects
            anchors = [example.texts[0] for example in batch]
            positives = [example.texts[1] for example in batch]
            negatives = [example.texts[2] for example in batch]



            # Tokenize texts
            anchor_inputs = model.tokenize(anchors)
            positive_inputs = model.tokenize(positives)
            negative_inputs = model.tokenize(negatives)

            # Move tokenized inputs to GPU
            anchor_inputs = {k: v.to(device) for k, v in anchor_inputs.items()}
            positive_inputs = {k: v.to(device) for k, v in positive_inputs.items()}
            negative_inputs = {k: v.to(device) for k, v in negative_inputs.items()}

            # Get embeddings through forward pass (maintains gradients)
            anchor_embeddings = model(anchor_inputs)['sentence_embedding']
            positive_embeddings = model(positive_inputs)['sentence_embedding']
            negative_embeddings = model(negative_inputs)['sentence_embedding']

            # Normalize embeddings
            anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)
            positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)
            negative_embeddings = torch.nn.functional.normalize(negative_embeddings, p=2, dim=1)

            # Compute cosine distance (1 - cosine_similarity)
            distance_pos = 1 - torch.sum(anchor_embeddings * positive_embeddings, dim=1)
            distance_neg = 1 - torch.sum(anchor_embeddings * negative_embeddings, dim=1)

            # Triplet loss with margin
            losses_triplet = torch.nn.functional.relu(distance_pos - distance_neg + margin)
            loss_value = losses_triplet.mean()

            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            # Batch Loss
            batch_loss = loss_value.item()
            epoch_losses.append(batch_loss)
            history['batch_losses'].append(batch_loss)


            # Learning rate warmup
            if global_step < warmup_steps:
                lr_scale = float(global_step + 1) / float(max(1, warmup_steps))
                for pg in optimizer.param_groups:
                    pg['lr'] = learning_rate * lr_scale

            total_loss += loss_value.item()
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss_value.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_loss = total_loss / num_batches
        print(f"Average loss: {avg_loss:.4f}")

        avg_train_loss = np.mean(epoch_losses)
        test_loss = compute_test_loss(model, test_data, margin, batch_size=32)


        print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f}")
        # Evaluate on test set after each epoch
        print(f"\nEvaluating on test set...")
        test_accuracy = evaluate_on_test_set(model, test_data, batch_size=32)
        print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")

        # Save best model
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            model.save(output_path)
            print(f"✓ New best model saved (accuracy: {test_accuracy:.4f})")
        else:
            print(f"  (Best so far: {best_test_accuracy:.4f})")
            # Save to history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)
        scheduler.step(test_accuracy)
        old_lr = optimizer.param_groups[0]['lr']
        # scheduler.step(test_accuracy)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != old_lr:
            print(f"📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        # best_test_accuracy = 0.0
        early_stopping = EarlyStopping(patience=2, min_delta=0.005)
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping triggered after epoch {epoch + 1}")
            break


    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best test accuracy: {best_test_accuracy:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"{'='*60}")
    output_png="data"
    plot_training_progress(history, output_png)

    return model


def evaluate_on_test_set(model: SentenceTransformer, test_data: List[InputExample],
                         batch_size: int = 32) -> float:
    """
    Efficient batch evaluation for CPU.
    """
    model.eval()
    anchors = [ex.texts[0] for ex in test_data]
    positives = [ex.texts[1] for ex in test_data]
    negatives = [ex.texts[2] for ex in test_data]

    with torch.no_grad():
        # Encode in batches
        anchor_embs = model.encode(
            anchors,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        pos_embs = model.encode(
            positives,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        neg_embs = model.encode(
            negatives,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        # Ensure embeddings are 2D (batch_size, embedding_dim)
        if anchor_embs.dim() == 1:
            anchor_embs = anchor_embs.unsqueeze(0)
        if pos_embs.dim() == 1:
            pos_embs = pos_embs.unsqueeze(0)
        if neg_embs.dim() == 1:
            neg_embs = neg_embs.unsqueeze(0)
        # Compute similarities
        sims_pos = torch.sum(anchor_embs * pos_embs, dim=1)
        sims_neg = torch.sum(anchor_embs * neg_embs, dim=1)

        correct = torch.sum(sims_pos > sims_neg).item()
        total = len(test_data)


    model.train()
    return correct / total


def evaluate_model(model, labeled_data_path: str, embedding_lookup: Dict,
                  margin: float = 0.0) -> Tuple[float, pd.DataFrame, Dict]:
    """
    Evaluate model accuracy on Track A judgments with comprehensive metrics.
    Returns accuracy, results dataframe, and detailed metrics dictionary.
    """
    df = pd.read_json(labeled_data_path, lines=True)

    # Preprocess
    df["anchor_text"] = df["anchor_text"].apply(preprocess_text)
    df["text_a"] = df["text_a"].apply(preprocess_text)
    df["text_b"] = df["text_b"].apply(preprocess_text)

    # Map texts to embeddings
    df["anchor_embedding"] = df["anchor_text"].map(embedding_lookup)
    df["a_embedding"] = df["text_a"].map(embedding_lookup)
    df["b_embedding"] = df["text_b"].map(embedding_lookup)

    # Check for missing embeddings
    missing = df[["anchor_embedding", "a_embedding", "b_embedding"]].isnull().any(axis=1).sum()
    if missing > 0:
        print(f"Warning: {missing} rows have missing embeddings")

    # Calculate similarities
    df["sim_a"] = df.apply(
        lambda row: cos_sim(row["anchor_embedding"], row["a_embedding"]).item(), axis=1
    )
    df["sim_b"] = df.apply(
        lambda row: cos_sim(row["anchor_embedding"], row["b_embedding"]).item(), axis=1
    )

    # Predict with optional margin
    df["similarity_diff"] = df["sim_a"] - df["sim_b"]

    # Calculate prediction confidence score (absolute difference)
    df["prediction_confidence"] = df["similarity_diff"].abs()

    if margin > 0:
        df["predicted_text_a_is_closer"] = df["similarity_diff"].apply(
            lambda x: True if x > margin else (False if x < -margin else None)
        )
        confident_predictions = df[df["predicted_text_a_is_closer"].notna()]
        accuracy = (confident_predictions["predicted_text_a_is_closer"] ==
                   confident_predictions["text_a_is_closer"]).mean()
        coverage = len(confident_predictions) / len(df)

        # Calculate metrics
        correct_predictions = (confident_predictions["predicted_text_a_is_closer"] ==
                              confident_predictions["text_a_is_closer"])

        metrics = {
            "accuracy": float(accuracy),
            "coverage": float(coverage),
            "total_samples": len(df),
            "confident_samples": len(confident_predictions),
            "correct_predictions": int(correct_predictions.sum()),
            "incorrect_predictions": int((~correct_predictions).sum()),
            "mean_confidence": float(confident_predictions["prediction_confidence"].mean()),
            "median_confidence": float(confident_predictions["prediction_confidence"].median()),
            "mean_sim_a": float(df["sim_a"].mean()),
            "mean_sim_b": float(df["sim_b"].mean()),
            "margin": float(margin)
        }
        print(f"Coverage with margin {margin}: {coverage:.2%}")
    else:
        df["predicted_text_a_is_closer"] = df["sim_a"] > df["sim_b"]
        correct_predictions = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"])
        accuracy = correct_predictions.mean()

        # Calculate comprehensive metrics
        correct_df = df[correct_predictions]
        incorrect_df = df[~correct_predictions]

        # Calculate precision, recall, F1 (treating "text_a is closer" as positive class)
        true_positives = ((df["predicted_text_a_is_closer"] == True) & (df["text_a_is_closer"] == True)).sum()
        false_positives = ((df["predicted_text_a_is_closer"] == True) & (df["text_a_is_closer"] == False)).sum()
        false_negatives = ((df["predicted_text_a_is_closer"] == False) & (df["text_a_is_closer"] == True)).sum()
        true_negatives = ((df["predicted_text_a_is_closer"] == False) & (df["text_a_is_closer"] == False)).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "coverage": 1.0,
            "total_samples": len(df),
            "confident_samples": len(df),
            "correct_predictions": int(correct_predictions.sum()),
            "incorrect_predictions": int((~correct_predictions).sum()),
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "true_negatives": int(true_negatives),
            "false_negatives": int(false_negatives),
            "mean_confidence": float(df["prediction_confidence"].mean()),
            "median_confidence": float(df["prediction_confidence"].median()),
            "std_confidence": float(df["prediction_confidence"].std()),
            "min_confidence": float(df["prediction_confidence"].min()),
            "max_confidence": float(df["prediction_confidence"].max()),
            "mean_sim_a": float(df["sim_a"].mean()),
            "mean_sim_b": float(df["sim_b"].mean()),
            "std_sim_a": float(df["sim_a"].std()),
            "std_sim_b": float(df["sim_b"].std()),
            "margin": float(margin)
        }

        # Confidence by correctness
        metrics["mean_confidence_correct"] = float(correct_df["prediction_confidence"].mean())
        metrics["mean_confidence_incorrect"] = float(incorrect_df["prediction_confidence"].mean())
        metrics["median_confidence_correct"] = float(correct_df["prediction_confidence"].median())
        metrics["median_confidence_incorrect"] = float(incorrect_df["prediction_confidence"].median())

        # Quartile analysis
        quartiles = df["prediction_confidence"].quantile([0.25, 0.5, 0.75]).to_dict()
        metrics["confidence_q1"] = float(quartiles[0.25])
        metrics["confidence_q2"] = float(quartiles[0.5])
        metrics["confidence_q3"] = float(quartiles[0.75])

    return accuracy, df, metrics


def main():
    # ============ CONFIGURATION ============
    MODE = "fine_tune"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Fine-tuning parameters
    EPOCHS = 6
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    MARGIN = 0.5
    DROPOUT_RATE = 0.2

    # Paths
    SYNTHETIC_TRAIN_PATH = "data/synthetic_data_for_classification.jsonl"
    DEV_TRACK_A_PATH = "data/dev_track_a.jsonl"  # Dev labels (for validation)
    DEV_TRACK_B_PATH = "data/dev_track_b.jsonl"  # Dev texts (for embeddings)
    TEST_TRACK_A_PATH = "data/test/test_track_a.jsonl"  # Test labels (if they exist)
    TEST_TRACK_B_PATH = "data/test/test_track_b.jsonl"  # Test texts (for embeddings)
    OUTPUT_MODEL_PATH = "output/fine_tuned_model"
    OUTPUT_DIR = "output"

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print(f"{'=' * 60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Training: Synthetic data")
    print(f"Validation: Development data")
    print(f"Test: Generate embeddings")
    print(f"{'=' * 60}")

    if MODE == "fine_tune":
        print(f"\n{'=' * 60}")
        print("STEP 1: LOAD TRAINING DATA (SYNTHETIC)")
        print(f"{'=' * 60}")


        # Create training examples
        train_data = []
        train_data = prepare_training_data(
            SYNTHETIC_TRAIN_PATH,
            DEV_TRACK_A_PATH,
            DEV_TRACK_B_PATH
        )
        # Load test data
        test_data, df_test_labels = load_test_data(TEST_TRACK_A_PATH, TEST_TRACK_B_PATH)


        print(f"\n{'=' * 60}")
        print(f"DATA SUMMARY:")
        print(f"  Training:   {len(train_data)} examples (synthetic)")
        print(f"  Validation: {len(test_data)} examples (development)")
        print(f"{'=' * 60}")

        # ========== STEP 3: FINE-TUNE MODEL ==========
        print(f"\n{'=' * 60}")
        print("STEP 3: FINE-TUNING MODEL")
        print(f"{'=' * 60}")

        model = fine_tune_model(
            MODEL_NAME,
            train_data,
            test_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            output_path=OUTPUT_MODEL_PATH,
            dropout_rate=DROPOUT_RATE,
            margin=MARGIN
        )

        # Load best model
        print(f"\nLoading best fine-tuned model from {OUTPUT_MODEL_PATH}...")
        model = SentenceTransformer(OUTPUT_MODEL_PATH, device="cpu")

        # ========== STEP 4: FINAL EVALUATION ON DEVELOPMENT SET ==========
        print("\n" + "=" * 60)
        print("STEP 4: FINAL EVALUATION ON DEVELOPMENT SET")
        print("=" * 60)

        # Load dev Track B for final embeddings
        data_b_dev = pd.read_json(DEV_TRACK_B_PATH, lines=True)
        data_b_dev["text"] = data_b_dev["text"].apply(preprocess_text)
        print(f"\nLoaded {len(data_b_dev)} test stories for embeddings")

        # Generate embeddings for development data
        print("\nGenerating embeddings for development data...")
        dev_texts = data_b_dev["text"].tolist()
        dev_embeddings = model.encode(
            dev_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Create embedding lookup for development data
        dev_embedding_lookup = dict(zip(data_b_dev["text"], dev_embeddings))

        # Evaluate on development set using DEV_TRACK_A_PATH
        dev_accuracy, dev_results_df, dev_metrics = evaluate_model(
            model, DEV_TRACK_A_PATH, dev_embedding_lookup, margin=0.0
        )

        # Print detailed results
        print(f"\nFINAL DEVELOPMENT SET RESULTS")
        print(f"{'=' * 60}")
        print(f"Accuracy:  {dev_accuracy:.4f} ({dev_accuracy * 100:.2f}%)")
        print(f"Precision: {dev_metrics['precision']:.4f}")
        print(f"Recall:    {dev_metrics['recall']:.4f}")
        print(f"F1-Score:  {dev_metrics['f1_score']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {dev_metrics['true_positives']}")
        print(f"  False Positives: {dev_metrics['false_positives']}")
        print(f"  True Negatives:  {dev_metrics['true_negatives']}")
        print(f"  False Negatives: {dev_metrics['false_negatives']}")
        print(f"\nPrediction Confidence:")
        print(f"  Mean:   {dev_metrics['mean_confidence']:.4f}")
        print(f"  Median: {dev_metrics['median_confidence']:.4f}")
        print(f"{'=' * 60}")

        # ========== STEP 5: GENERATE TEST EMBEDDINGS ==========
        print("\n" + "=" * 60)
        print("STEP 5: GENERATE TEST EMBEDDINGS (FOR SUBMISSION)")
        print("=" * 60)

        # Load test Track B for embeddings
        print(f"\nLoading test texts from {TEST_TRACK_B_PATH}...")
        data_b_test = pd.read_json(TEST_TRACK_B_PATH, lines=True)
        data_b_test["text"] = data_b_test["text"].apply(preprocess_text)
        print(f"Loaded {len(data_b_test)} test texts")

        # Generate embeddings for test data
        print("\nGenerating embeddings for test data...")
        test_texts = data_b_test["text"].tolist()
        test_embeddings = model.encode(
            test_texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        print(f"✓ Generated {len(test_embeddings)} test embeddings")

        # ========== STEP 6: SAVE EVERYTHING ==========
        print("\n" + "=" * 60)
        print("STEP 6: SAVING OUTPUTS")
        print("=" * 60)

        # Save test embeddings
        test_output_file = Path(OUTPUT_DIR) / "test_track_b_embeddings.npy"
        np.save(test_output_file, test_embeddings)
        print(f"✓ Test embeddings saved to {test_output_file}")
        print(f"  Shape: {test_embeddings.shape}")

        # Save development embeddings
        dev_output_file = Path(OUTPUT_DIR) / "dev_track_b_embeddings.npy"
        np.save(dev_output_file, dev_embeddings)
        print(f"✓ Development embeddings saved to {dev_output_file}")
        print(f"  Shape: {dev_embeddings.shape}")

        # Save metadata
        metadata = {
            "model": MODEL_NAME,
            "training_samples": len(train_data),
            "validation_samples": len(test_data),
            "dev_accuracy": float(dev_accuracy),
            "dev_metrics": dev_metrics,
            "test_embeddings_count": len(test_embeddings),
            "dev_embeddings_count": len(dev_embeddings),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "margin": MARGIN,
        }

        metadata_file = Path(OUTPUT_DIR) / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_file}")

        print(f"\n{'=' * 60}")
        print("PIPELINE COMPLETE")
        print(f"{'=' * 60}")
        print(f"Training Data:   {len(train_data)} examples (synthetic)")
        print(f"Validation Data: {len(test_data)} examples (development)")
        print(f"Dev Accuracy:    {dev_accuracy * 100:.2f}%")
        print(f"Test Embeddings: {len(test_embeddings)} generated")
        print(f"{'=' * 60}")

        return dev_accuracy

if __name__ == "__main__":
    try:
        import time
        start_time = time.time()

        accuracy = main()

        elapsed = time.time() - start_time
        print(f"\nTotal execution time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR OCCURRED:")
        print("="*60)
        traceback.print_exc()
        print("="*60)
        sys.exit(1)