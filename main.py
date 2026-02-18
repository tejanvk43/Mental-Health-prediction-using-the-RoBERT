import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
import random
import glob
import os
import pickle
import json
import gc
from collections import defaultdict

warnings.filterwarnings('ignore')

CONFIG = {
    'architecture': 'simple',       # simple works best, complex adds BiGRU+attention
    'model_name': 'roberta-base',   # swap to distilroberta-base if training is slow
    'hidden_dim': 256,
    'num_heads': 8,
    'dropout': 0.5,
    'max_length': 160,

    'batch_size': 16,               # effective batch = 64 after accumulation
    'num_epochs': 12,
    'accumulation_steps': 4,
    'lr_backbone': 1e-5,
    'lr_head': 3e-4,
    'weight_decay': 0.05,
    'warmup_ratio': 0.1,
    'early_stopping_patience': 3,

    'max_per_class': 15000,
    'min_per_class': 10000,
    'min_text_length': 20,
    'max_text_length': 2000,
    'augmentation_rate': 0.6,

    'focal_gamma': 2.0,
    'label_smoothing': 0.1,

    'use_curriculum': True,
    'use_contrastive': True,        # helps separate Depression vs Suicidal
    'contrastive_weight': 0.1,
    'test_time_augmentation': True,
    'ensemble_size': 1,             # bump to 3 for ensemble training
}

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

VALID_SUBREDDITS = {'depression', 'SuicideWatch', 'mentalhealth', 'Anxiety', 'lonely'}

SUBREDDIT_TO_CLASS = {
    'depression': 'Depression',
    'suicidewatch': 'Suicidal',
    'mentalhealth': 'Normal',
    'anxiety': 'Anxiety',
    'lonely': 'Depression',
}


def load_and_clean_data(rmhd_path, combined_path, config):
    print("="*80)
    print("LOADING & CLEANING DATA")
    print("="*80)

    rmhd_data = load_all_rmhd(rmhd_path, config)

    print(f"\nLoading Combined Data from {combined_path}")
    df_combined = pd.read_csv(combined_path)
    df_combined = df_combined[['statement', 'status']].copy()
    df_combined.columns = ['combined_text', 'mapped_label']
    df_combined = df_combined.dropna()
    print(f"Combined Data: {len(df_combined):,} samples")

    if rmhd_data is not None:
        all_data = pd.concat([rmhd_data, df_combined], ignore_index=True)
        all_data = all_data.drop_duplicates(subset=['combined_text'])
        print(f"\nMerged dataset: {len(all_data):,} samples")
    else:
        all_data = df_combined

    print("\n" + "="*80)
    print("APPLYING CLEANING")
    print("="*80)

    initial_count = len(all_data)

    all_data = all_data[all_data['combined_text'].str.len() >= config['min_text_length']]
    print(f"Removed {initial_count - len(all_data):,} texts under {config['min_text_length']} chars")

    all_data['combined_text'] = all_data['combined_text'].apply(
        lambda x: x[:config['max_text_length']] if len(x) > config['max_text_length'] else x
    )
    print(f"Capped text length at {config['max_text_length']} chars")

    all_data['_word_count'] = all_data['combined_text'].str.split().str.len()
    all_data = all_data[all_data['_word_count'] >= 5]
    print(f"Removed texts with fewer than 5 words")

    all_data = all_data.drop_duplicates(subset=['combined_text', 'mapped_label'])
    print(f"Dropped duplicate texts within each class")

    all_data = all_data.drop(columns=['_word_count'])

    print(f"\nCleaned dataset: {len(all_data):,} samples")
    print(f"Total removed: {initial_count - len(all_data):,} ({(initial_count - len(all_data))/initial_count*100:.1f}%)")

    all_data = balance_dataset(all_data, config)
    return all_data


def load_all_rmhd(base_path, config):
    print("\nLoading RMHD data...")

    search_patterns = [
        os.path.join(base_path, 'raw data', '*', '*', '*.csv'),
        os.path.join(base_path, 'raw data', '*', '*.csv'),
    ]

    all_csv_files = []
    for pattern in search_patterns:
        all_csv_files.extend(glob.glob(pattern))
    all_csv_files = list(set(all_csv_files))

    if not all_csv_files:
        print("No RMHD files found")
        return None

    print(f"Found {len(all_csv_files)} CSV files")

    all_data = []
    for file_path in tqdm(all_csv_files, desc="Loading RMHD"):
        try:
            df = pd.read_csv(file_path, usecols=['subreddit', 'selftext', 'title'],
                             dtype={'subreddit': str, 'selftext': str, 'title': str})
            df = df[df['subreddit'].isin(VALID_SUBREDDITS)]
            if len(df) > 0:
                all_data.append(df)
        except:
            pass

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined['combined_text'] = (
        combined['title'].fillna('').astype(str) + ' ' +
        combined['selftext'].fillna('').astype(str)
    ).str.strip()

    combined = combined[combined['combined_text'].str.len() > config['min_text_length']]
    combined['mapped_label'] = combined['subreddit'].str.lower().map(SUBREDDIT_TO_CLASS)
    combined = combined.dropna(subset=['mapped_label'])

    print(f"RMHD: {len(combined):,} samples after cleaning")

    # cap per class, preferring longer posts since they carry more signal
    sampled = []
    for label in combined['mapped_label'].unique():
        class_df = combined[combined['mapped_label'] == label]
        if len(class_df) > config['max_per_class']:
            class_df = class_df.copy()
            class_df['_len'] = class_df['combined_text'].str.len()
            n_long = int(config['max_per_class'] * 0.7)
            n_rand = config['max_per_class'] - n_long
            top = class_df.nlargest(n_long, '_len')
            rest = class_df.drop(top.index).sample(n=min(n_rand, len(class_df)-n_long), random_state=42)
            class_df = pd.concat([top, rest]).drop(columns=['_len'])
        sampled.append(class_df)

    return pd.concat(sampled, ignore_index=True)[['combined_text', 'mapped_label']]


def balance_dataset(df, config):
    print("\n" + "="*80)
    print("BALANCING DATASET")
    print("="*80)

    balanced_parts = []

    for label in df['mapped_label'].unique():
        class_df = df[df['mapped_label'] == label]

        if len(class_df) < config['min_per_class']:
            oversampled = class_df.sample(n=config['min_per_class'], replace=True, random_state=42)
            balanced_parts.append(oversampled)
            print(f"  {label:24s}: {len(class_df):>6,} → {config['min_per_class']:>6,} (oversampled)")
        elif len(class_df) > config['max_per_class']:
            downsampled = class_df.sample(n=config['max_per_class'], random_state=42)
            balanced_parts.append(downsampled)
            print(f"  {label:24s}: {len(class_df):>6,} → {config['max_per_class']:>6,} (downsampled)")
        else:
            balanced_parts.append(class_df)
            print(f"  {label:24s}: {len(class_df):>6,} (kept as is)")

    result = pd.concat(balanced_parts, ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nBalanced dataset: {len(result):,} samples")
    return result


class EnhancedTextAugmentation:

    def __init__(self):
        # a small hand-crafted dictionary focused on mental health language
        self.synonyms = {
            'sad': ['unhappy', 'down', 'low', 'blue', 'melancholy'],
            'happy': ['glad', 'pleased', 'content', 'joyful', 'cheerful'],
            'anxious': ['worried', 'nervous', 'uneasy', 'tense', 'apprehensive'],
            'tired': ['exhausted', 'drained', 'fatigued', 'weary', 'worn out'],
            'angry': ['upset', 'furious', 'irritated', 'mad', 'frustrated'],
            'scared': ['afraid', 'frightened', 'terrified', 'fearful'],
            'depressed': ['dejected', 'despondent', 'gloomy', 'downcast'],
            'lonely': ['isolated', 'alone', 'solitary', 'friendless'],
            'hopeless': ['desperate', 'despairing', 'pessimistic'],
            'worthless': ['useless', 'inadequate', 'meaningless'],
            'worried': ['concerned', 'troubled', 'bothered'],
            'stressed': ['overwhelmed', 'pressured', 'strained', 'tense'],
            'suicidal': ['self-destructive', 'wanting to end it'],
            'kill': ['end', 'terminate'],
            'die': ['pass away', 'end my life'],
        }

    def synonym_replace(self, text, n=2):
        words = text.split()
        new_words = words.copy()

        replaceable = [(i, w) for i, w in enumerate(words) if w.lower() in self.synonyms]
        if not replaceable:
            return text

        random.shuffle(replaceable)
        for i, word in replaceable[:n]:
            syns = self.synonyms.get(word.lower(), [])
            if syns:
                new_words[i] = random.choice(syns)

        return ' '.join(new_words)

    def random_word_dropout(self, text, p=0.05):
        words = text.split()
        if len(words) <= 5:
            return text
        kept = [w for w in words if random.random() > p]
        return ' '.join(kept) if kept else text

    def random_word_shuffle(self, text, window=3):
        # shuffles within small local windows to keep overall meaning intact
        words = text.split()
        if len(words) <= window:
            return text

        for i in range(0, len(words) - window, window):
            window_words = words[i:i+window]
            random.shuffle(window_words)
            words[i:i+window] = window_words

        return ' '.join(words)

    def augment(self, text):
        r = random.random()

        if r < 0.3:
            return self.synonym_replace(text, n=random.randint(1, 3))
        elif r < 0.5:
            return self.random_word_dropout(text, p=0.05)
        elif r < 0.6:
            return self.random_word_shuffle(text, window=3)
        else:
            # stack two light augmentations
            text = self.synonym_replace(text, n=1)
            text = self.random_word_dropout(text, p=0.03)
            return text


class EnhancedLinguisticFeatureExtractor:
    """Extracts 25 hand-crafted features including suicide-specific markers."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

        self.negative_words = frozenset([
            'sad', 'depressed', 'anxious', 'worried', 'hopeless', 'worthless',
            'lonely', 'tired', 'exhausted', 'miserable', 'terrible', 'awful',
            'hate', 'hurt', 'pain', 'cry', 'fear', 'scared', 'angry', 'upset',
            'frustrated', 'helpless', 'empty', 'numb', 'suffering', 'broken'
        ])

        self.positive_words = frozenset([
            'happy', 'joy', 'excited', 'grateful', 'loved', 'confident',
            'hopeful', 'good', 'great', 'wonderful', 'amazing', 'peaceful',
            'calm', 'relaxed', 'proud', 'strong', 'better'
        ])

        self.anxiety_words = frozenset([
            'anxious', 'panic', 'worry', 'nervous', 'fear', 'scared',
            'terrified', 'stress', 'overwhelmed', 'tense', 'dread', 'phobia'
        ])

        self.depression_words = frozenset([
            'depressed', 'sad', 'hopeless', 'worthless', 'empty', 'numb',
            'meaningless', 'pointless', 'tired', 'exhausted', 'dark'
        ])

        # words that tend to appear in suicidal posts but rarely in depressive ones
        self.suicide_words = frozenset([
            'suicide', 'suicidal', 'kill', 'die', 'death', 'end', 'over',
            'goodbye', 'farewell', 'final', 'last', 'rope', 'pills',
            'jump', 'gun', 'overdose', 'hang', 'cut'
        ])

        self.first_person = frozenset(['i', 'me', 'my', 'mine', 'myself'])

    def extract_features(self, text):
        if not isinstance(text, str) or len(text) == 0:
            return np.zeros(25, dtype=np.float32)

        text_lower = text.lower()
        words = text_lower.split()
        n_words = len(words)
        if n_words == 0:
            return np.zeros(25, dtype=np.float32)

        features = []

        # VADER sentiment scores
        sentiment = self.vader.polarity_scores(text)
        features.extend([sentiment['compound'], sentiment['pos'],
                         sentiment['neg'], sentiment['neu']])

        # word category ratios
        features.extend([
            sum(1 for w in words if w in self.negative_words) / n_words,
            sum(1 for w in words if w in self.positive_words) / n_words,
            sum(1 for w in words if w in self.anxiety_words) / n_words,
            sum(1 for w in words if w in self.depression_words) / n_words,
            sum(1 for w in words if w in self.suicide_words) / n_words,
        ])

        # first person pronoun ratio (elevated in personal distress)
        features.append(sum(1 for w in words if w in self.first_person) / n_words)

        # basic syntactic stats
        sentences = [s for s in text.split('.') if s.strip()]
        n_sents = max(len(sentences), 1)
        features.extend([
            n_sents,
            n_words / n_sents,
            len(set(words)) / n_words,  # lexical diversity
        ])

        # punctuation patterns
        text_len = max(len(text), 1)
        features.extend([
            text.count('!') / text_len,
            text.count('?') / text_len,
            text.count('...') / text_len,   # trailing thoughts, common in depression
            text.count('.') / text_len,
        ])

        # word length stats
        word_lengths = [len(w) for w in words]
        features.extend([
            np.mean(word_lengths),
            np.std(word_lengths) if len(word_lengths) > 1 else 0,
            max(word_lengths),
        ])

        # writing style features
        features.extend([
            sum(1 for c in text if c.isupper()) / text_len,
            len([i for i in range(len(text)-2)
                 if text[i] == text[i+1] == text[i+2]]) / text_len,
            text.count('..') / text_len,
        ])

        # sentence length distribution
        features.extend([
            sum(1 for s in sentences if len(s.strip()) < 20) / n_sents,
            sum(1 for s in sentences if len(s.strip()) > 100) / n_sents,
        ])

        return np.array(features, dtype=np.float32)


class MentalHealthDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, feature_extractor,
                 max_length=160, augment=False, augmenter=None,
                 augmentation_rate=0.6):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.augment = augment
        self.augmenter = augmenter
        self.augmentation_rate = augmentation_rate

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        if self.augment and self.augmenter and random.random() < self.augmentation_rate:
            text = self.augmenter.augment(text)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        ling_features = self.feature_extractor.extract_features(text)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'linguistic_features': torch.tensor(ling_features, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class FocalLoss(nn.Module):

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class ContrastiveLoss(nn.Module):
    """Pushes Depression and Suicidal embeddings further apart in the feature space."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels, depression_idx, suicidal_idx):
        depression_mask = (labels == depression_idx)
        suicidal_mask = (labels == suicidal_idx)

        if depression_mask.sum() == 0 or suicidal_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        depression_embs = embeddings[depression_mask]
        suicidal_embs = embeddings[suicidal_mask]

        losses = []
        for d_emb in depression_embs:
            for s_emb in suicidal_embs:
                dist = F.pairwise_distance(d_emb.unsqueeze(0), s_emb.unsqueeze(0))
                loss = torch.clamp(self.margin - dist, min=0.0)
                losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=embeddings.device)

        return torch.stack(losses).mean()


class SimpleMentalHealthClassifier(nn.Module):
    """RoBERTa with mean pooling + linguistic feature branch. Fastest to train."""

    def __init__(self, model_name='roberta-base', num_classes=7, dropout=0.5):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        backbone_dim = self.backbone.config.hidden_size

        # keep early layers frozen — they encode general language well enough
        for param in self.backbone.encoder.layer[:4].parameters():
            param.requires_grad = False
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.feature_fc = nn.Sequential(
            nn.Linear(25, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        combined_dim = backbone_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, linguistic_features):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)

        ling = self.feature_fc(linguistic_features)

        combined = torch.cat([pooled, ling], dim=1)
        logits = self.classifier(combined)

        return logits, pooled


class ComplexMentalHealthClassifier(nn.Module):
    """RoBERTa + BiGRU + multi-head attention. Stronger but slower than simple."""

    def __init__(self, model_name='roberta-base', num_classes=7,
                 hidden_dim=256, num_heads=8, dropout=0.5):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        backbone_dim = self.backbone.config.hidden_size

        for param in self.backbone.encoder.layer[:4].parameters():
            param.requires_grad = False
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        self.bigru = nn.GRU(
            backbone_dim, hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout * 0.5
        )

        gru_out_dim = hidden_dim * 2

        self.attention = nn.MultiheadAttention(
            gru_out_dim, num_heads, dropout=dropout * 0.5, batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(gru_out_dim)
        self.layer_norm2 = nn.LayerNorm(gru_out_dim)

        self.pool_attention = nn.Sequential(
            nn.Linear(gru_out_dim, 1),
            nn.Softmax(dim=1)
        )

        self.feature_fc = nn.Sequential(
            nn.Linear(25, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        combined_dim = gru_out_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, linguistic_features):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence = outputs.last_hidden_state

        gru_out, _ = self.bigru(sequence)
        gru_out = self.layer_norm1(gru_out)

        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        attn_out = self.layer_norm2(attn_out + gru_out)

        attn_weights = self.pool_attention(attn_out)
        pooled = torch.sum(attn_out * attn_weights, dim=1)

        ling = self.feature_fc(linguistic_features)

        combined = torch.cat([pooled, ling], dim=1)
        logits = self.classifier(combined)

        return logits, pooled


def create_model(config, num_classes):
    if config['architecture'] == 'simple':
        return SimpleMentalHealthClassifier(
            model_name=config['model_name'],
            num_classes=num_classes,
            dropout=config['dropout']
        )
    elif config['architecture'] == 'complex':
        return ComplexMentalHealthClassifier(
            model_name=config['model_name'],
            num_classes=num_classes,
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}")


class EarlyStopping:

    def __init__(self, patience=3, min_delta=0.001, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_acc, model, epoch):
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            improvement = f"+{val_acc - self.best_score:.4f}" if self.best_score else "first"
            print(f"  New best: {val_acc:.4f} ({improvement})")
            self.best_score = val_acc
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            self.counter = 0
        else:
            self.counter += 1
            print(f"  No improvement: {self.counter}/{self.patience} "
                  f"(best: {self.best_score:.4f} @ epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"  Early stopping — best was {self.best_score:.4f} @ epoch {self.best_epoch}")


def train_epoch(model, dataloader, optimizer, criterion, device, config,
                scheduler=None, scaler=None, depression_idx=None,
                suicidal_idx=None, contrastive_criterion=None):
    model.train()
    total_loss = 0.0
    total_clf_loss = 0.0
    total_con_loss = 0.0
    predictions, true_labels = [], []

    optimizer.zero_grad()

    progress = tqdm(dataloader, desc="  Training")
    for idx, batch in enumerate(progress):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        ling_features = batch['linguistic_features'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        if scaler is not None:
            with autocast():
                logits, embeddings = model(input_ids, attention_mask, ling_features)
                clf_loss = criterion(logits, labels)

                con_loss = torch.tensor(0.0, device=device)
                if (config['use_contrastive'] and contrastive_criterion is not None
                        and depression_idx is not None and suicidal_idx is not None):
                    con_loss = contrastive_criterion(embeddings, labels,
                                                     depression_idx, suicidal_idx)

                loss = (clf_loss + config['contrastive_weight'] * con_loss) / config['accumulation_steps']

            scaler.scale(loss).backward()

            if (idx + 1) % config['accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
        else:
            logits, embeddings = model(input_ids, attention_mask, ling_features)
            clf_loss = criterion(logits, labels)

            con_loss = torch.tensor(0.0, device=device)
            if (config['use_contrastive'] and contrastive_criterion is not None
                    and depression_idx is not None and suicidal_idx is not None):
                con_loss = contrastive_criterion(embeddings, labels,
                                                 depression_idx, suicidal_idx)

            loss = (clf_loss + config['contrastive_weight'] * con_loss) / config['accumulation_steps']
            loss.backward()

            if (idx + 1) % config['accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item() * config['accumulation_steps']
        total_clf_loss += clf_loss.item()
        total_con_loss += con_loss.item()

        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        if idx % 100 == 0:
            progress.set_postfix({
                'loss': f'{loss.item() * config["accumulation_steps"]:.4f}',
                'clf': f'{clf_loss.item():.4f}',
                'con': f'{con_loss.item():.4f}'
            })

    avg_loss = total_loss / len(dataloader)
    avg_clf = total_clf_loss / len(dataloader)
    avg_con = total_con_loss / len(dataloader)
    acc = accuracy_score(true_labels, predictions)

    return avg_loss, acc, avg_clf, avg_con


def evaluate(model, dataloader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0.0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            ling_features = batch['linguistic_features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            if scaler is not None:
                with autocast():
                    logits, _ = model(input_ids, attention_mask, ling_features)
                    loss = criterion(logits, labels)
            else:
                logits, _ = model(input_ids, attention_mask, ling_features)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(true_labels, predictions)

    return avg_loss, acc, predictions, true_labels


def plot_results(train_losses, train_accs, val_losses, val_accs,
                 y_true, y_pred, label_encoder, save_dir='/kaggle/working'):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, 'b-o', label='Train', linewidth=2, markersize=6)
    axes[0].plot(epochs, val_losses, 'r-s', label='Validation', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in train_accs], 'b-o', label='Train', linewidth=2, markersize=6)
    axes[1].plot(epochs, [a*100 for a in val_accs], 'r-s', label='Validation', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=90, color='g', linestyle='--', alpha=0.5, linewidth=2, label='90% target')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {save_dir}/")


def main():
    print("="*80)
    print("MENTAL HEALTH CLASSIFICATION")
    print("="*80)
    print(f"Architecture:         {CONFIG['architecture']}")
    print(f"Model:                {CONFIG['model_name']}")
    print(f"Batch size:           {CONFIG['batch_size']} (effective: {CONFIG['batch_size'] * CONFIG['accumulation_steps']})")
    print(f"Dropout:              {CONFIG['dropout']}")
    print(f"Weight decay:         {CONFIG['weight_decay']}")
    print(f"Augmentation rate:    {CONFIG['augmentation_rate']}")
    print(f"Early stop patience:  {CONFIG['early_stopping_patience']}")
    print(f"Contrastive loss:     {CONFIG['use_contrastive']}")
    print(f"Curriculum learning:  {CONFIG['use_curriculum']}")
    print("="*80)

    rmhd_path = '/kaggle/input/datasets/entenam/reddit-mental-health-dataset/Original Reddit Data'
    combined_path = '/kaggle/input/datasets/s210238/mental-health-original/Combined Data.csv'

    all_data = load_and_clean_data(rmhd_path, combined_path, CONFIG)

    label_encoder = LabelEncoder()
    all_data['encoded_label'] = label_encoder.fit_transform(all_data['mapped_label'])

    texts = all_data['combined_text'].values
    labels = all_data['encoded_label'].values
    num_classes = len(label_encoder.classes_)

    print(f"\nFinal dataset: {len(all_data):,} samples, {num_classes} classes")
    print("\nClass distribution:")
    for i, cls in enumerate(label_encoder.classes_):
        count = (labels == i).sum()
        print(f"  {i}: {cls:24s} ({count:,} samples, {count/len(labels)*100:.1f}%)")

    depression_idx = None
    suicidal_idx = None
    for i, cls in enumerate(label_encoder.classes_):
        if cls == 'Depression':
            depression_idx = i
        elif cls == 'Suicidal':
            suicidal_idx = i

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"\nClass weights: {[f'{w:.3f}' for w in class_weights.cpu().numpy()]}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"\nSplits: Train {len(X_train):,} | Val {len(X_val):,} | Test {len(X_test):,}")

    print("\nInitializing tokenizer and feature extractor...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    feature_extractor = EnhancedLinguisticFeatureExtractor()
    augmenter = EnhancedTextAugmentation()

    train_dataset = MentalHealthDataset(
        X_train, y_train, tokenizer, feature_extractor, CONFIG['max_length'],
        augment=True, augmenter=augmenter, augmentation_rate=CONFIG['augmentation_rate']
    )
    val_dataset = MentalHealthDataset(
        X_val, y_val, tokenizer, feature_extractor, CONFIG['max_length']
    )
    test_dataset = MentalHealthDataset(
        X_test, y_test, tokenizer, feature_extractor, CONFIG['max_length']
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=2, pin_memory=pin, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size']*2, shuffle=False,
        num_workers=2, pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG['batch_size']*2, shuffle=False,
        num_workers=2, pin_memory=pin
    )

    print("\nBuilding model...")
    model = create_model(CONFIG, num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': CONFIG['lr_backbone'], 'weight_decay': CONFIG['weight_decay']},
        {'params': head_params, 'lr': CONFIG['lr_head'], 'weight_decay': CONFIG['weight_decay']},
    ])

    total_steps = (len(train_loader) * CONFIG['num_epochs']) // CONFIG['accumulation_steps']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    criterion = FocalLoss(weight=class_weights, gamma=CONFIG['focal_gamma'],
                          label_smoothing=CONFIG['label_smoothing'])
    contrastive_criterion = ContrastiveLoss(margin=1.0) if CONFIG['use_contrastive'] else None

    scaler = GradScaler() if torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'],
                                   save_path='/kaggle/working/best_model.pth')

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-"*80)

        train_loss, train_acc, clf_loss, con_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, CONFIG,
            scheduler, scaler, depression_idx, suicidal_idx, contrastive_criterion
        )

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, scaler)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        gap = (train_acc - val_acc) * 100
        status = ("Excellent" if abs(gap) < 2 else "Good" if abs(gap) < 5 else "Overfitting")

        print(f"\n  Train: loss={train_loss:.4f}  acc={train_acc*100:.2f}%  "
              f"(clf={clf_loss:.4f}, con={con_loss:.4f})")
        print(f"  Val:   loss={val_loss:.4f}  acc={val_acc*100:.2f}%")
        print(f"  Gap: {gap:+.2f}% — {status}")

        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"  LR: Backbone={lrs[0]:.2e}, Head={lrs[1]:.2e}")

        early_stopping(val_acc, model, epoch+1)
        if early_stopping.early_stop:
            break

    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device, scaler)
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f"\nTest Accuracy:  {test_acc*100:.2f}%")
    print(f"Cohen's Kappa:  {kappa:.4f}")
    print(f"Best Epoch:     {early_stopping.best_epoch}")
    print(f"\n{classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=4)}")

    plot_results(train_losses, train_accs, val_losses, val_accs,
                 y_true, y_pred, label_encoder)

    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder_classes': label_encoder.classes_,
        'test_accuracy': test_acc,
        'kappa': kappa,
        'config': CONFIG,
    }, '/kaggle/working/complete_model.pth')

    print("\n" + "="*80)
    if test_acc >= 0.90:
        print(f"Target reached! {test_acc*100:.2f}% accuracy")
    elif test_acc >= 0.85:
        print(f"Good result: {test_acc*100:.2f}% accuracy")
    else:
        print(f"Result: {test_acc*100:.2f}% accuracy")
    print("="*80)


if __name__ == "__main__":
    main()