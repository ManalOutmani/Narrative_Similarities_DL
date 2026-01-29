"""
Fine-tuning system with Separate Specialized Models architecture.
Three specialized models trained on different narrative aspects:
1. Theme Model - Abstract themes, ideas, and motifs
2. Action Model - Course of action, events, and sequences
3. Outcome Model - Story outcomes and resolutions

Each model is trained separately and their predictions are combined.
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
from torch.optim.lr_scheduler import ReduceLROnPlateau


def format_e5_input(texts: List[str], prefix: str = "passage: ") -> List[str]:
    """Format texts for E5 models which require specific prefixes."""
    return [prefix + text for text in texts]


def preprocess_text(text: str) -> str:
    """Light preprocessing to improve embedding quality."""
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    text = ' '.join(text.split())
    text = text.strip()
    return text


class SpecializedModelEnsemble:
    """
    Ensemble of three specialized models for narrative similarity.
    Each model focuses on a different aspect: theme, action, outcome.
    """

    def __init__(self, base_model_name: str, device: str = "cpu"):
        self.base_model_name = base_model_name
        self.device = device

        # Initialize three separate models
        print(f"\n{'=' * 60}")
        print("INITIALIZING SPECIALIZED MODEL ENSEMBLE")
        print(f"{'=' * 60}")

        print("Loading Theme Model (Abstract themes, ideas, motifs)...")
        self.theme_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # Better for semantic themes



        print("Loading Action Model (Course of action, events, sequences)...")
        #self.action_model = SentenceTransformer(base_model_name, device=device)
        self.action_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("Loading Outcome Model (Story outcomes, resolutions)...")
        self.outcome_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        #self.outcome_model = SentenceTransformer(base_model_name, device=device)

        # Weights for combining model predictions (can be tuned)
        self.weights = {
            'theme': 0.35,  # Theme importance
            'action': 0.40,  # Action sequence importance
            'outcome': 0.25  # Outcome importance
        }

        print(f"\n✓ All models initialized")
        print(f"Combination weights: Theme={self.weights['theme']:.2f}, "
              f"Action={self.weights['action']:.2f}, "
              f"Outcome={self.weights['outcome']:.2f}")
        print(f"{'=' * 60}")

    def get_model(self, aspect: str) -> SentenceTransformer:
        """Get the model for a specific aspect."""
        if aspect == 'theme':
            return self.theme_model
        elif aspect == 'action':
            return self.action_model
        elif aspect == 'outcome':
            return self.outcome_model
        else:
            raise ValueError(f"Unknown aspect: {aspect}")

    def encode(self, texts: List[str], aspect: str = 'combined',
               batch_size: int = 32, show_progress_bar: bool = True,
               normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode texts using specified aspect model or combined ensemble.

        Args:
            texts: List of texts to encode
            aspect: 'theme', 'action', 'outcome', or 'combined'
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress
            normalize_embeddings: Whether to normalize embeddings

        Returns:
            Embeddings as numpy array
        """
        if aspect in ['theme', 'action', 'outcome']:
            model = self.get_model(aspect)
            return model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True
            )
        elif aspect == 'combined':
            # Encode with all models and combine
            theme_embs = self.theme_model.encode(
                texts, batch_size=batch_size, show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings, convert_to_numpy=True
            )
            action_embs = self.action_model.encode(
                texts, batch_size=batch_size, show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings, convert_to_numpy=True
            )
            outcome_embs = self.outcome_model.encode(
                texts, batch_size=batch_size, show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings, convert_to_numpy=True
            )

            # Weighted combination
            combined = (
                    self.weights['theme'] * theme_embs +
                    self.weights['action'] * action_embs +
                    self.weights['outcome'] * outcome_embs
            )

            if normalize_embeddings:
                # Normalize combined embeddings
                norms = np.linalg.norm(combined, axis=1, keepdims=True)
                combined = combined / (norms + 1e-8)

            return combined
        else:
            raise ValueError(f"Unknown aspect: {aspect}")

    def compute_similarity(self, anchor: str, text_a: str, text_b: str,
                           aspect: str = 'combined') -> Tuple[float, float]:
        """
        Compute similarity scores for a triple using specified aspect.

        Returns:
            (similarity_to_a, similarity_to_b)
        """
        # Encode all texts
        embeddings = self.encode([anchor, text_a, text_b], aspect=aspect,
                                 show_progress_bar=False)

        anchor_emb = embeddings[0]
        a_emb = embeddings[1]
        b_emb = embeddings[2]

        # Compute cosine similarities
        sim_a = np.dot(anchor_emb, a_emb)
        sim_b = np.dot(anchor_emb, b_emb)

        return sim_a, sim_b

    def predict(self, anchor: str, text_a: str, text_b: str,
                aspect: str = 'combined') -> bool:
        """
        Predict which text is more similar to anchor.

        Returns:
            True if text_a is closer, False if text_b is closer
        """
        sim_a, sim_b = self.compute_similarity(anchor, text_a, text_b, aspect)
        return sim_a > sim_b

    def save(self, output_dir: str):
        """Save all models to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        theme_path = output_path / "theme_model"
        action_path = output_path / "action_model"
        outcome_path = output_path / "outcome_model"

        self.theme_model.save(str(theme_path))
        self.action_model.save(str(action_path))
        self.outcome_model.save(str(outcome_path))

        # Save weights
        weights_path = output_path / "ensemble_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(self.weights, f, indent=2)

        print(f"\n✓ Ensemble saved to {output_dir}")
        print(f"  - Theme model: {theme_path}")
        print(f"  - Action model: {action_path}")
        print(f"  - Outcome model: {outcome_path}")
        print(f"  - Weights: {weights_path}")

    @classmethod
    def load(cls, output_dir: str, device: str = "cpu"):
        """Load ensemble from saved directory."""
        output_path = Path(output_dir)

        theme_path = output_path / "theme_model"
        action_path = output_path / "action_model"
        outcome_path = output_path / "outcome_model"
        weights_path = output_path / "ensemble_weights.json"

        # Create instance with dummy base model
        ensemble = cls.__new__(cls)
        ensemble.device = device

        # Load models
        print(f"\nLoading ensemble from {output_dir}...")
        ensemble.theme_model = SentenceTransformer(str(theme_path), device=device)
        ensemble.action_model = SentenceTransformer(str(action_path), device=device)
        ensemble.outcome_model = SentenceTransformer(str(outcome_path), device=device)

        # Load weights
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                ensemble.weights = json.load(f)
        else:
            ensemble.weights = {'theme': 0.35, 'action': 0.40, 'outcome': 0.25}

        print(f"✓ Ensemble loaded successfully")
        return ensemble


def prepare_training_data(synthetic_path: str, dev_track_a_path: str) -> List[InputExample]:
    """
    Load and combine synthetic training data with development data.
    """
    print(f"\n{'=' * 60}")
    print("LOADING TRAINING DATA")
    print(f"{'=' * 60}")

    all_train_examples = []

    # Load synthetic data
    print(f"\n1. Loading synthetic data from {synthetic_path}...")
    df_synthetic = pd.read_json(synthetic_path, lines=True)
    print(f"   Loaded {len(df_synthetic)} synthetic triples")

    # Preprocess
    df_synthetic["anchor_text"] = df_synthetic["anchor_text"].apply(preprocess_text)
    df_synthetic["text_a"] = df_synthetic["text_a"].apply(preprocess_text)
    df_synthetic["text_b"] = df_synthetic["text_b"].apply(preprocess_text)

    # Create training examples
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

    # Load development data
    print(f"\n2. Loading development data from {dev_track_a_path}...")
    df_dev_labels = pd.read_json(dev_track_a_path, lines=True)
    print(f"   Loaded {len(df_dev_labels)} development triples")

    # Preprocess
    df_dev_labels["anchor_text"] = df_dev_labels["anchor_text"].apply(preprocess_text)
    df_dev_labels["text_a"] = df_dev_labels["text_a"].apply(preprocess_text)
    df_dev_labels["text_b"] = df_dev_labels["text_b"].apply(preprocess_text)

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

    print(f"\n{'=' * 60}")
    print(f"TOTAL TRAINING DATA:")
    print(f"  Synthetic:    {synthetic_count} examples")
    print(f"  Development:  {dev_count} examples")
    print(f"  TOTAL:        {len(all_train_examples)} examples")
    print(f"{'=' * 60}")

    return all_train_examples


def fine_tune_specialized_model(model: SentenceTransformer,
                                train_data: List[InputExample],
                                val_data: List[InputExample],
                                aspect_name: str,
                                epochs: int = 5,
                                batch_size: int = 8,
                                learning_rate: float = 2e-5,
                                margin: float = 0.5,
                                dropout_rate: float = 0.3,
                                device: str = "cpu") -> Dict:
    """
    Fine-tune a single specialized model on specific aspect.

    Args:
        model: SentenceTransformer model to train
        train_data: Training examples
        val_data: Validation examples
        aspect_name: Name of aspect (theme/action/outcome)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        margin: Triplet loss margin
        dropout_rate: Dropout rate
        device: Device to use

    Returns:
        Training history dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"FINE-TUNING {aspect_name.upper()} MODEL")
    print(f"{'=' * 60}")

    # Configure dropout
    dropout_layers_found = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate
            dropout_layers_found += 1
    print(f"Configured {dropout_layers_found} dropout layers (rate: {dropout_rate})")

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    # Training configuration
    num_batches = len(train_data) // batch_size
    total_steps = num_batches * epochs
    warmup_steps = int(total_steps * 0.1)

    print(f"\nTraining Configuration:")
    print(f"  Aspect: {aspect_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {num_batches}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Margin: {margin}")

    history = {
        'aspect': aspect_name,
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'batch_losses': []
    }

    model.train()
    global_step = 0
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        print(f"\n{'=' * 60}")
        print(f"[{aspect_name.upper()}] Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 60}")

        # Shuffle training data
        shuffled_data = train_data.copy()
        random.shuffle(shuffled_data)

        epoch_losses = []
        progress_bar = tqdm(range(num_batches), desc=f"Training {aspect_name}")

        for batch_idx in progress_bar:
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = shuffled_data[start_idx:end_idx]

            # Extract texts
            anchors = [example.texts[0] for example in batch]
            positives = [example.texts[1] for example in batch]
            negatives = [example.texts[2] for example in batch]

            # Tokenize
            anchor_inputs = model.tokenize(anchors)
            positive_inputs = model.tokenize(positives)
            negative_inputs = model.tokenize(negatives)

            # Move to device
            anchor_inputs = {k: v.to(device) for k, v in anchor_inputs.items()}
            positive_inputs = {k: v.to(device) for k, v in positive_inputs.items()}
            negative_inputs = {k: v.to(device) for k, v in negative_inputs.items()}

            # Get embeddings
            anchor_embeddings = model(anchor_inputs)['sentence_embedding']
            positive_embeddings = model(positive_inputs)['sentence_embedding']
            negative_embeddings = model(negative_inputs)['sentence_embedding']

            # Normalize
            anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)
            positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)
            negative_embeddings = torch.nn.functional.normalize(negative_embeddings, p=2, dim=1)

            # Compute distances
            distance_pos = 1 - torch.sum(anchor_embeddings * positive_embeddings, dim=1)
            distance_neg = 1 - torch.sum(anchor_embeddings * negative_embeddings, dim=1)

            # Triplet loss
            losses_triplet = torch.nn.functional.relu(distance_pos - distance_neg + margin)
            loss_value = losses_triplet.mean()

            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record loss
            batch_loss = loss_value.item()
            epoch_losses.append(batch_loss)
            history['batch_losses'].append(batch_loss)

            # Learning rate warmup
            if global_step < warmup_steps:
                lr_scale = float(global_step + 1) / float(max(1, warmup_steps))
                for pg in optimizer.param_groups:
                    pg['lr'] = learning_rate * lr_scale

            global_step += 1

            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # Epoch metrics
        avg_train_loss = np.mean(epoch_losses)
        val_loss = compute_test_loss(model, val_data, margin, batch_size, device)
        val_accuracy = evaluate_on_test_set(model, val_data, batch_size, device)

        print(f"\n[{aspect_name.upper()}] Epoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_accuracy:.4f} ({val_accuracy * 100:.1f}%)")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"  ✓ New best accuracy for {aspect_name}: {val_accuracy:.4f}")

        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        # Scheduler step
        scheduler.step(val_accuracy)

    print(f"\n[{aspect_name.upper()}] Training complete!")
    print(f"  Best validation accuracy: {best_val_accuracy:.4f}")

    return history


def compute_test_loss(model: SentenceTransformer, test_data: List[InputExample],
                      margin: float, batch_size: int, device: str) -> float:
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


def evaluate_on_test_set(model: SentenceTransformer, test_data: List[InputExample],
                         batch_size: int, device: str) -> float:
    """Evaluate model accuracy on test set."""
    model.eval()
    anchors = [ex.texts[0] for ex in test_data]
    positives = [ex.texts[1] for ex in test_data]
    negatives = [ex.texts[2] for ex in test_data]

    with torch.no_grad():
        anchor_embs = model.encode(anchors, convert_to_tensor=True, batch_size=batch_size,
                                   show_progress_bar=False, normalize_embeddings=True)
        pos_embs = model.encode(positives, convert_to_tensor=True, batch_size=batch_size,
                                show_progress_bar=False, normalize_embeddings=True)
        neg_embs = model.encode(negatives, convert_to_tensor=True, batch_size=batch_size,
                                show_progress_bar=False, normalize_embeddings=True)

        if anchor_embs.dim() == 1:
            anchor_embs = anchor_embs.unsqueeze(0)
        if pos_embs.dim() == 1:
            pos_embs = pos_embs.unsqueeze(0)
        if neg_embs.dim() == 1:
            neg_embs = neg_embs.unsqueeze(0)

        sims_pos = torch.sum(anchor_embs * pos_embs, dim=1)
        sims_neg = torch.sum(anchor_embs * neg_embs, dim=1)

        correct = torch.sum(sims_pos > sims_neg).item()

    model.train()
    return correct / len(test_data)


def plot_ensemble_training_progress(histories: Dict[str, Dict], output_path: str):
    """Create training progress plots for all three models."""
    output_dir = Path(output_path).parent

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Specialized Models Training Progress', fontsize=16, fontweight='bold')

    colors = {'theme': 'blue', 'action': 'green', 'outcome': 'red'}

    # Plot 1: Training Loss Comparison
    for aspect, history in histories.items():
        axes[0, 0].plot(history['epoch'], history['train_loss'],
                        label=f'{aspect.capitalize()}',
                        color=colors[aspect], marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Loss Comparison
    for aspect, history in histories.items():
        axes[0, 1].plot(history['epoch'], history['val_loss'],
                        label=f'{aspect.capitalize()}',
                        color=colors[aspect], marker='o', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Validation Accuracy Comparison
    for aspect, history in histories.items():
        axes[0, 2].plot(history['epoch'], [a * 100 for a in history['val_accuracy']],
                        label=f'{aspect.capitalize()}',
                        color=colors[aspect], marker='o', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Validation Accuracy Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4-6: Individual model batch losses
    for idx, (aspect, history) in enumerate(histories.items()):
        ax = axes[1, idx]
        batch_losses = history['batch_losses']
        ax.plot(batch_losses, alpha=0.3, linewidth=0.5, color=colors[aspect])

        # Moving average
        window = 50
        if len(batch_losses) >= window:
            moving_avg = np.convolve(batch_losses, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(batch_losses)), moving_avg,
                    linewidth=2, color=colors[aspect])

        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{aspect.capitalize()} Model - Batch Losses')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / "ensemble_training_progress.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training plots saved to {plot_file}")
    plt.close()


def main():
    # ============ CONFIGURATION ============
    MODE = "fine_tune"
    BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fine-tuning parameters
    EPOCHS = 6
    BATCH_SIZE = 32 if device == "cuda" else 8
    LEARNING_RATE = 5e-5
    MARGIN = 0.5
    DROPOUT_RATE = 0.2
    TRAIN_TEST_SPLIT = 0.15

    # Paths
    SYNTHETIC_TRAIN_PATH = "data/synthetic_data_for_classification.jsonl"
    DEV_TRACK_A_PATH = "data/dev_track_a.jsonl"
    DEV_TRACK_B_PATH = "data/dev_track_b.jsonl"
    TEST_TRACK_A_PATH = "data/test/test_track_a.jsonl"
    TEST_TRACK_B_PATH = "data/test/test_track_b.jsonl"
    OUTPUT_MODEL_PATH = "output/ensemble_models"
    OUTPUT_DIR = "output"

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print(f"{'=' * 60}")
    print(f"SPECIALIZED MODELS TRAINING CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Base Model: {BASE_MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Architecture: 3 Specialized Models")
    print(f"  1. Theme Model (Abstract themes, motifs)")
    print(f"  2. Action Model (Course of action, events)")
    print(f"  3. Outcome Model (Story outcomes, resolutions)")
    print(f"{'=' * 60}")

    if MODE == "fine_tune":
        # ========== STEP 1: LOAD DATA ==========
        all_training_data = prepare_training_data(SYNTHETIC_TRAIN_PATH, DEV_TRACK_A_PATH)

        # ========== STEP 2: SPLIT DATA ==========
        print(f"\n{'=' * 60}")
        print("SPLITTING DATA")
        print(f"{'=' * 60}")

        train_data, val_data = train_test_split(
            all_training_data, test_size=TRAIN_TEST_SPLIT, random_state=42
        )

        print(f"Training:   {len(train_data)} examples")
        print(f"Validation: {len(val_data)} examples")

        # ========== STEP 3: INITIALIZE ENSEMBLE ==========
        ensemble = SpecializedModelEnsemble(BASE_MODEL_NAME, device=device)

        # ========== STEP 4: TRAIN EACH MODEL ==========
        training_histories = {}

        for aspect in ['theme', 'action', 'outcome']:
            model = ensemble.get_model(aspect)
            history = fine_tune_specialized_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                aspect_name=aspect,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                margin=MARGIN,
                dropout_rate=DROPOUT_RATE,
                device=device
            )
            training_histories[aspect] = history

        # ========== STEP 5: SAVE ENSEMBLE ==========
        ensemble.save(OUTPUT_MODEL_PATH)

        # ========== STEP 6: PLOT TRAINING PROGRESS ==========
        plot_ensemble_training_progress(training_histories, OUTPUT_DIR)

        # ========== STEP 7: EVALUATE ENSEMBLE ==========
        print(f"\n{'=' * 60}")
        print("EVALUATING ENSEMBLE ON DEVELOPMENT SET")
        print(f"{'=' * 60}")

        # Load dev data
        data_b_dev = pd.read_json(DEV_TRACK_B_PATH, lines=True)
        data_b_dev["text"] = data_b_dev["text"].apply(preprocess_text)

        # Generate embeddings with combined ensemble
        print("\nGenerating combined embeddings...")
        dev_texts = data_b_dev["text"].tolist()
        dev_embeddings_combined = ensemble.encode(dev_texts, aspect='combined',
                                                  batch_size=32, show_progress_bar=True)

        # Also generate individual aspect embeddings for analysis
        print("\nGenerating theme embeddings...")
        dev_embeddings_theme = ensemble.encode(dev_texts, aspect='theme',
                                               batch_size=32, show_progress_bar=True)

        print("Generating action embeddings...")
        dev_embeddings_action = ensemble.encode(dev_texts, aspect='action',
                                                batch_size=32, show_progress_bar=True)

        print("Generating outcome embeddings...")
        dev_embeddings_outcome = ensemble.encode(dev_texts, aspect='outcome',
                                                 batch_size=32, show_progress_bar=True)

        # Evaluate each model individually
        print(f"\n{'=' * 60}")
        print("INDIVIDUAL MODEL PERFORMANCE")
        print(f"{'=' * 60}")

        for aspect, embeddings in [('theme', dev_embeddings_theme),
                                   ('action', dev_embeddings_action),
                                   ('outcome', dev_embeddings_outcome),
                                   ('combined', dev_embeddings_combined)]:
            embedding_lookup = dict(zip(data_b_dev["text"], embeddings))
            accuracy = evaluate_model_simple(DEV_TRACK_A_PATH, embedding_lookup)
            print(f"{aspect.capitalize():10s} Model Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # ========== STEP 8: GENERATE TEST PREDICTIONS ==========
        print(f"\n{'=' * 60}")
        print("GENERATING TEST PREDICTIONS")
        print(f"{'=' * 60}")

        # Load test data
        data_b_test = pd.read_json(TEST_TRACK_B_PATH, lines=True)
        data_b_test["text"] = data_b_test["text"].apply(preprocess_text)

        # Generate test embeddings (combined)
        print("\nGenerating test embeddings (combined ensemble)...")
        test_texts = data_b_test["text"].tolist()
        test_embeddings = ensemble.encode(test_texts, aspect='combined',
                                          batch_size=32, show_progress_bar=True)

        # Generate predictions
        df_test_a = pd.read_json(TEST_TRACK_A_PATH, lines=True)
        df_test_a["anchor_text"] = df_test_a["anchor_text"].apply(preprocess_text)
        df_test_a["text_a"] = df_test_a["text_a"].apply(preprocess_text)
        df_test_a["text_b"] = df_test_a["text_b"].apply(preprocess_text)

        test_embedding_lookup = dict(zip(data_b_test["text"], test_embeddings))

        predictions = []
        for idx, row in tqdm(df_test_a.iterrows(), total=len(df_test_a), desc="Predicting"):
            anchor_emb = test_embedding_lookup.get(row["anchor_text"])
            a_emb = test_embedding_lookup.get(row["text_a"])
            b_emb = test_embedding_lookup.get(row["text_b"])

            if anchor_emb is None or a_emb is None or b_emb is None:
                # Encode on the fly if not found
                if anchor_emb is None:
                    anchor_emb = ensemble.encode([row["anchor_text"]], aspect='combined',
                                                 show_progress_bar=False)[0]
                if a_emb is None:
                    a_emb = ensemble.encode([row["text_a"]], aspect='combined',
                                            show_progress_bar=False)[0]
                if b_emb is None:
                    b_emb = ensemble.encode([row["text_b"]], aspect='combined',
                                            show_progress_bar=False)[0]

            sim_a = np.dot(anchor_emb, a_emb)
            sim_b = np.dot(anchor_emb, b_emb)

            predictions.append({
                "anchor_text": row["anchor_text"],
                "text_a": row["text_a"],
                "text_b": row["text_b"],
                "text_a_is_closer": bool(sim_a > sim_b)
            })

        # ========== STEP 9: SAVE OUTPUTS ==========
        print(f"\n{'=' * 60}")
        print("SAVING OUTPUTS")
        print(f"{'=' * 60}")

        # Save embeddings
        np.save(Path(OUTPUT_DIR) / "test_embeddings_combined.npy", test_embeddings)
        np.save(Path(OUTPUT_DIR) / "dev_embeddings_combined.npy", dev_embeddings_combined)
        np.save(Path(OUTPUT_DIR) / "dev_embeddings_theme.npy", dev_embeddings_theme)
        np.save(Path(OUTPUT_DIR) / "dev_embeddings_action.npy", dev_embeddings_action)
        np.save(Path(OUTPUT_DIR) / "dev_embeddings_outcome.npy", dev_embeddings_outcome)

        # Save predictions
        predictions_file = Path(OUTPUT_DIR) / "test_track_a_predictions.jsonl"
        with open(predictions_file, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')

        # Save metadata
        metadata = {
            "architecture": "Separate Specialized Models",
            "base_model": BASE_MODEL_NAME,
            "device": device,
            "models": {
                "theme": "Abstract themes, ideas, motifs",
                "action": "Course of action, events, sequences",
                "outcome": "Story outcomes, resolutions"
            },
            "combination_weights": ensemble.weights,
            "training_config": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "margin": MARGIN,
                "dropout_rate": DROPOUT_RATE
            },
            "data_split": {
                "total": len(all_training_data),
                "train": len(train_data),
                "validation": len(val_data)
            },
            "training_histories": training_histories
        }

        with open(Path(OUTPUT_DIR) / "ensemble_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ All outputs saved to {OUTPUT_DIR}")
        print(f"✓ Ensemble saved to {OUTPUT_MODEL_PATH}")
        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 60}")


def evaluate_model_simple(labeled_data_path: str, embedding_lookup: Dict) -> float:
    """Simple accuracy evaluation."""
    df = pd.read_json(labeled_data_path, lines=True)
    df["anchor_text"] = df["anchor_text"].apply(preprocess_text)
    df["text_a"] = df["text_a"].apply(preprocess_text)
    df["text_b"] = df["text_b"].apply(preprocess_text)

    df["anchor_embedding"] = df["anchor_text"].map(embedding_lookup)
    df["a_embedding"] = df["text_a"].map(embedding_lookup)
    df["b_embedding"] = df["text_b"].map(embedding_lookup)

    df["sim_a"] = df.apply(lambda row: cos_sim(row["anchor_embedding"], row["a_embedding"]).item(), axis=1)
    df["sim_b"] = df.apply(lambda row: cos_sim(row["anchor_embedding"], row["b_embedding"]).item(), axis=1)

    df["predicted_text_a_is_closer"] = df["sim_a"] > df["sim_b"]
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()

    return accuracy


if __name__ == "__main__":
    try:
        import time

        start_time = time.time()

        main()

        elapsed = time.time() - start_time
        print(f"\nTotal execution time: {elapsed / 60:.1f} minutes ({elapsed:.0f} seconds)")
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR OCCURRED:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)