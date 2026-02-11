"""
Mental Health Prediction using RoBERTa + BiGRU + Multi-Head Attention
Dataset: Kaggle Mental Health Sentiment Analysis
Author: Your Name
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#==============================================================================
# 1. LINGUISTIC FEATURE EXTRACTOR
#==============================================================================

class LinguisticFeatureExtractor:
    """Extract psychology-informed linguistic features from text"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        # Psychology-informed word lists
        self.negative_words = set([
            'sad', 'depressed', 'anxious', 'worried', 'hopeless', 'worthless',
            'lonely', 'tired', 'exhausted', 'miserable', 'terrible', 'awful',
            'hate', 'hurt', 'pain', 'cry', 'fear', 'scared', 'angry', 'upset',
            'frustrated', 'helpless', 'empty', 'numb', 'broken', 'lost'
        ])
        
        self.positive_words = set([
            'happy', 'joy', 'excited', 'grateful', 'loved', 'confident',
            'hopeful', 'good', 'great', 'wonderful', 'amazing', 'fantastic',
            'peaceful', 'calm', 'relaxed', 'content', 'satisfied', 'proud'
        ])
        
        self.anxiety_words = set([
            'anxious', 'panic', 'worry', 'nervous', 'fear', 'scared',
            'terrified', 'stress', 'overwhelmed', 'tense', 'uneasy'
        ])
        
        self.depression_words = set([
            'depressed', 'sad', 'hopeless', 'worthless', 'empty', 'numb',
            'suicide', 'death', 'die', 'kill', 'end', 'give up'
        ])
        
        self.first_person = set(['i', 'me', 'my', 'mine', 'myself'])
    
    def extract_features(self, text):
        """Extract 15-dimensional feature vector"""
        if not isinstance(text, str) or len(text) == 0:
            return np.zeros(15)
        
        text_lower = text.lower()
        words = text_lower.split()
        
        if len(words) == 0:
            return np.zeros(15)
        
        features = []
        
        # 1-4: VADER Sentiment Scores
        sentiment = self.vader.polarity_scores(text)
        features.extend([
            sentiment['compound'],
            sentiment['pos'],
            sentiment['neg'],
            sentiment['neu']
        ])
        
        # 5-6: Emotion Lexicon Counts
        neg_count = sum(1 for w in words if w in self.negative_words)
        pos_count = sum(1 for w in words if w in self.positive_words)
        features.extend([
            neg_count / len(words),
            pos_count / len(words)
        ])
        
        # 7-8: Mental Health Specific Keywords
        anxiety_count = sum(1 for w in words if w in self.anxiety_words)
        depression_count = sum(1 for w in words if w in self.depression_words)
        features.extend([
            anxiety_count / len(words),
            depression_count / len(words)
        ])
        
        # 9: First-Person Pronoun Ratio
        pronoun_ratio = sum(1 for w in words if w in self.first_person) / len(words)
        features.append(pronoun_ratio)
        
        # 10-12: Text Statistics
        sentences = text.split('.')
        avg_sent_length = len(words) / max(len(sentences), 1)
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words) if len(words) > 0 else 0
        features.extend([
            len(sentences),
            avg_sent_length,
            lexical_diversity
        ])
        
        # 13-15: Punctuation Patterns
        text_len = len(text)
        features.extend([
            text.count('!') / text_len if text_len > 0 else 0,
            text.count('?') / text_len if text_len > 0 else 0,
            text.count('...') / text_len if text_len > 0 else 0
        ])
        
        return np.array(features, dtype=np.float32)

#==============================================================================
# 2. DATASET CLASS
#==============================================================================

class MentalHealthDataset(Dataset):
    """Custom Dataset for Mental Health Text Classification"""
    
    def __init__(self, texts, labels, tokenizer, feature_extractor, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize with RoBERTa
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Extract linguistic features
        ling_features = self.feature_extractor.extract_features(text)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'linguistic_features': torch.tensor(ling_features, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }

#==============================================================================
# 3. MODEL ARCHITECTURE
#==============================================================================

class RoBERTa_BiGRU_Attention_MentalHealth(nn.Module):
    """
    Hybrid Architecture:
    RoBERTa → BiGRU → Multi-Head Attention → Feature Fusion → Classification
    """
    
    def __init__(self, num_classes=7, hidden_dim=128, num_heads=4, dropout=0.3):
        super(RoBERTa_BiGRU_Attention_MentalHealth, self).__init__()
        
        # 1. RoBERTa Base Model
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        # Freeze first 6 layers (fine-tune last 6)
        for param in self.roberta.encoder.layer[:6].parameters():
            param.requires_grad = False
        
        # 2. Bidirectional GRU
        self.bigru = nn.GRU(
            input_size=768,  # RoBERTa hidden size
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 3. Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # BiGRU outputs 2*hidden_dim
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # 5. Feature Fusion for Linguistic Features
        self.feature_fc = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 6. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, linguistic_features):
        # 1. RoBERTa Embeddings
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = roberta_output.last_hidden_state  # (batch, seq, 768)
        
        # 2. BiGRU Processing
        gru_output, _ = self.bigru(sequence_output)  # (batch, seq, 256)
        
        # 3. Multi-Head Attention
        attn_output, attn_weights = self.attention(
            gru_output, gru_output, gru_output
        )  # (batch, seq, 256)
        
        # 4. Layer Normalization + Residual Connection
        attn_output = self.layer_norm(attn_output + gru_output)
        
        # 5. Global Average Pooling
        pooled_output = torch.mean(attn_output, dim=1)  # (batch, 256)
        
        # 6. Process Linguistic Features
        ling_features = self.feature_fc(linguistic_features)  # (batch, 64)
        
        # 7. Feature Fusion
        combined = torch.cat([pooled_output, ling_features], dim=1)  # (batch, 320)
        
        # 8. Classification
        logits = self.classifier(combined)
        
        return logits, attn_weights

#==============================================================================
# 4. DATA LOADING AND PREPROCESSING
#==============================================================================

def load_and_preprocess_data(file_path):
    """Load and preprocess the mental health dataset"""
    
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check for the correct column names
    # The dataset typically has 'statement' and 'status' columns
    if 'statement' in df.columns and 'status' in df.columns:
        text_column = 'statement'
        label_column = 'status'
    elif 'text' in df.columns and 'label' in df.columns:
        text_column = 'text'
        label_column = 'label'
    else:
        # Print available columns to help debug
        print("\nAvailable columns:", df.columns.tolist())
        raise ValueError("Could not find text and label columns")
    
    # Remove missing values
    df = df.dropna(subset=[text_column, label_column])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=[text_column])
    
    # Clean text
    df[text_column] = df[text_column].astype(str)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df[label_column])
    
    print(f"\nLabel distribution:")
    print(df[label_column].value_counts())
    
    print(f"\nEncoded labels:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{i}: {label}")
    
    return df[text_column].values, df['encoded_label'].values, label_encoder

#==============================================================================
# 5. TRAINING FUNCTION
#==============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        linguistic_features = batch['linguistic_features'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(input_ids, attention_mask, linguistic_features)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

#==============================================================================
# 6. EVALUATION FUNCTION
#==============================================================================

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            linguistic_features = batch['linguistic_features'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids, attention_mask, linguistic_features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels

#==============================================================================
# 7. VISUALIZATION FUNCTIONS
#==============================================================================

def plot_confusion_matrix(y_true, y_pred, label_encoder, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path='training_curves.png'):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy', marker='o')
    ax2.plot(val_accs, label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")

#==============================================================================
# 8. MAIN TRAINING LOOP
#==============================================================================

def main():
    """Main training pipeline"""
    
    # Hyperparameters
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    
    # Load data
    DATA_PATH = r'E:\Acedamics\Mental Health classification using the RBERT\data\Combined Data.csv'  # Update this path
    texts, labels, label_encoder = load_and_preprocess_data(DATA_PATH)
    
    num_classes = len(label_encoder.classes_)
    print(f"\nNumber of classes: {num_classes}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset splits:")
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")
    
    # Initialize tokenizer and feature extractor
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    feature_extractor = LinguisticFeatureExtractor()
    
    # Create datasets
    train_dataset = MentalHealthDataset(X_train, y_train, tokenizer, feature_extractor, MAX_LENGTH)
    val_dataset = MentalHealthDataset(X_val, y_val, tokenizer, feature_extractor, MAX_LENGTH)
    test_dataset = MentalHealthDataset(X_test, y_test, tokenizer, feature_extractor, MAX_LENGTH)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = RoBERTa_BiGRU_Attention_MentalHealth(
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS
    ).to(device)
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Training loop
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0
    
    print("\n" + "="*50)
    print("Starting Training...")
    print("="*50)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Best model saved with validation accuracy: {best_val_acc:.4f}")
    
    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Test evaluation
    print("\n" + "="*50)
    print("Testing on Test Set...")
    print("="*50)
    
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Calculate additional metrics
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    # Classification report
    print("\n" + "="*50)
    print("Classification Report:")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    
    # Plot results
    plot_confusion_matrix(y_true, y_pred, label_encoder)
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)

if __name__ == "__main__":
    main()