import os
import time
import json
import shutil
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
import configuration as cfg
import preprocessing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Dynamic Device Allocation (MPS for Apple Silicon, CUDA for GPU, CPU fallback)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Compute Device Allocated: {device}")

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, dtype=torch.float).to(model.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

class IndoBERTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_optimal_threshold(y_true, y_probs):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1

def train_model():
    start_time = time.time()
    os.makedirs(cfg.MODEL_EXPORT_DIR, exist_ok=True)
    
    df_input, df_aug = preprocessing.get_data()
    
    if isinstance(df_input, dict):
        print("[WARNING] df_input detected as dictionary. Converting to DataFrame...")
        df_input = pd.DataFrame(df_input)
        
    df_input = df_input.reset_index(drop=True)
    
    print("\n[INFO] --- Initiating Strict Hold-out Test Split ---")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED)
    train_val_idx, test_idx = next(sss.split(df_input, df_input[cfg.LABEL_COLUMN_INPUT]))
    
    df_test = df_input.iloc[test_idx].copy()
    test_file_path = os.path.join(cfg.MODEL_EXPORT_DIR, "test_split.csv")
    df_test.to_csv(test_file_path, index=False)
    print(f"[INFO] Test Set saved strictly to {test_file_path}. Size: {len(df_test)}")
    
    df_cv = df_input.iloc[train_val_idx].copy().reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_CHECKPOINT)
    
    cv_folds = getattr(cfg, 'CROSS_VALIDATION_FOLDS', 5)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg.RANDOM_SEED)
    
    best_fold_index = -1
    best_fold_score = -1.0
    fold_metrics = {}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_cv, df_cv[cfg.LABEL_COLUMN_INPUT]), 1):
        print(f"\n{'='*20} FOLD {fold} {'='*20}")
        df_train = df_cv.iloc[train_idx].copy()
        df_val = df_cv.iloc[val_idx].copy()
        
        pd.DataFrame({'train_index': train_idx}).to_csv(
            os.path.join(cfg.MODEL_EXPORT_DIR, f"train_idx_fold_{fold}.csv"), index=False
        )
        pd.DataFrame({'val_index': val_idx}).to_csv(
            os.path.join(cfg.MODEL_EXPORT_DIR, f"val_idx_fold_{fold}.csv"), index=False
        )
        
        if getattr(cfg, 'USE_AUGMENTATION', False) and df_aug is not None:
            train_parent_ids = set(df_train[cfg.PARENT_ID_INPUT].dropna().unique())
            df_aug_filtered = df_aug[df_aug[cfg.PARENT_ID_AUGMENTATION].isin(train_parent_ids)].copy()
            
            df_aug_filtered = df_aug_filtered.rename(columns={
                cfg.TEXT_COLUMN_AUGMENTATION: cfg.TEXT_COLUMN_INPUT,
                cfg.LABEL_COLUMN_AUGMENTATION: cfg.LABEL_COLUMN_INPUT
            })
            
            df_train = pd.concat([df_train, df_aug_filtered], ignore_index=True)
            print(f"[INFO] Dynamic Augmentation: Injected {len(df_aug_filtered)} records into Train Set.")

        initial_train_shape = len(df_train)
        df_train = df_train.drop_duplicates(subset=[cfg.TEXT_COLUMN_INPUT]).reset_index(drop=True)
        dedup_train_shape = len(df_train)
        print(f"[INFO] Internal Train Deduplication: Removed {initial_train_shape - dedup_train_shape} post-augmentation duplicate records.")

        if getattr(cfg, 'USE_HIERARCHICAL', False):
            print("\n[INFO] --- Training Level 1 (Neutral vs Non-Neutral) ---")
            df_train_l1 = df_train.copy()
            df_val_l1 = df_val.copy()
            
            df_train_l1['L1_Label'] = np.where(df_train_l1[cfg.LABEL_COLUMN_INPUT] == cfg.BINARY_TARGET_LABEL, 'neutral', 'non_neutral')
            df_val_l1['L1_Label'] = np.where(df_val_l1[cfg.LABEL_COLUMN_INPUT] == cfg.BINARY_TARGET_LABEL, 'neutral', 'non_neutral')
            
            if getattr(cfg, 'USE_UNDERSAMPLING', False):
                df_non_neutral = df_train_l1[df_train_l1['L1_Label'] == 'non_neutral']
                df_neutral = df_train_l1[df_train_l1['L1_Label'] == 'neutral']
                
                sample_size = len(df_non_neutral)
                if len(df_neutral) > sample_size:
                    df_neutral = df_neutral.sample(n=sample_size, random_state=cfg.RANDOM_SEED)
                df_train_l1 = pd.concat([df_neutral, df_non_neutral]).sample(frac=1, random_state=cfg.RANDOM_SEED)
                
            labels_l1 = sorted(df_train_l1['L1_Label'].unique())
            label2id_l1 = {label: i for i, label in enumerate(labels_l1)}
            with open(os.path.join(cfg.MODEL_EXPORT_DIR, "label_mapping_lvl1.json"), 'w') as f:
                json.dump(label2id_l1, f)
                
            train_enc_l1 = tokenizer(df_train_l1[cfg.TEXT_COLUMN_INPUT].tolist(), truncation=True, padding=True, max_length=cfg.MAX_LEN)
            val_enc_l1 = tokenizer(df_val_l1[cfg.TEXT_COLUMN_INPUT].tolist(), truncation=True, padding=True, max_length=cfg.MAX_LEN)
            
            train_dataset_l1 = IndoBERTDataset(train_enc_l1, df_train_l1['L1_Label'].map(label2id_l1).tolist())
            val_dataset_l1 = IndoBERTDataset(val_enc_l1, df_val_l1['L1_Label'].map(label2id_l1).tolist())
            
            class_weights_l1 = None
            if not getattr(cfg, 'USE_UNDERSAMPLING', False):
                class_weights_l1 = compute_class_weight('balanced', classes=np.unique(df_train_l1['L1_Label']), y=df_train_l1['L1_Label'])
                
            model_l1 = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_CHECKPOINT, num_labels=len(labels_l1))
            model_l1.to(device)
            
            training_args = TrainingArguments(
                output_dir=os.path.join(cfg.MODEL_EXPORT_DIR, f"temp_fold_{fold}_l1"),
                num_train_epochs=cfg.EPOCHS,
                per_device_train_batch_size=cfg.BATCH_SIZE,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                learning_rate=cfg.LEARNING_RATE,
                seed=cfg.RANDOM_SEED,
                report_to="none"
            )
            
            trainer_l1 = CustomTrainer(
                model=model_l1,
                args=training_args,
                train_dataset=train_dataset_l1,
                eval_dataset=val_dataset_l1,
                class_weights=class_weights_l1,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            trainer_l1.train()
            
            preds_l1 = trainer_l1.predict(val_dataset_l1)
            probs_l1 = torch.nn.functional.softmax(torch.tensor(preds_l1.predictions), dim=-1).numpy()
            
            target_class_idx = label2id_l1.get('non_neutral', 1)
            best_thresh, l1_f1 = get_optimal_threshold(df_val_l1['L1_Label'].map(label2id_l1).tolist(), probs_l1[:, target_class_idx])
            
            print(f"[INFO] Level 1 Optimal Threshold: {best_thresh:.4f} | Macro F1: {l1_f1:.4f}")
            
            print("\n[INFO] --- Training Level 2 (Specific Categories) ---")
            df_train_l2 = df_train[df_train[cfg.LABEL_COLUMN_INPUT] != cfg.BINARY_TARGET_LABEL].copy()
            df_val_l2 = df_val[df_val[cfg.LABEL_COLUMN_INPUT] != cfg.BINARY_TARGET_LABEL].copy()
            
            labels_l2 = sorted(df_train_l2[cfg.LABEL_COLUMN_INPUT].unique())
            label2id_l2 = {label: i for i, label in enumerate(labels_l2)}
            with open(os.path.join(cfg.MODEL_EXPORT_DIR, "label_mapping_lvl2.json"), 'w') as f:
                json.dump(label2id_l2, f)
                
            train_enc_l2 = tokenizer(df_train_l2[cfg.TEXT_COLUMN_INPUT].tolist(), truncation=True, padding=True, max_length=cfg.MAX_LEN)
            val_enc_l2 = tokenizer(df_val_l2[cfg.TEXT_COLUMN_INPUT].tolist(), truncation=True, padding=True, max_length=cfg.MAX_LEN)
            
            train_dataset_l2 = IndoBERTDataset(train_enc_l2, df_train_l2[cfg.LABEL_COLUMN_INPUT].map(label2id_l2).tolist())
            val_dataset_l2 = IndoBERTDataset(val_enc_l2, df_val_l2[cfg.LABEL_COLUMN_INPUT].map(label2id_l2).tolist())
            
            class_weights_l2 = compute_class_weight('balanced', classes=np.unique(df_train_l2[cfg.LABEL_COLUMN_INPUT]), y=df_train_l2[cfg.LABEL_COLUMN_INPUT])
            
            model_l2 = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_CHECKPOINT, num_labels=len(labels_l2))
            model_l2.to(device)
            
            training_args_l2 = TrainingArguments(
                output_dir=os.path.join(cfg.MODEL_EXPORT_DIR, f"temp_fold_{fold}_l2"),
                num_train_epochs=cfg.EPOCHS,
                per_device_train_batch_size=cfg.BATCH_SIZE,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                learning_rate=cfg.LEARNING_RATE,
                seed=cfg.RANDOM_SEED,
                report_to="none"
            )
            
            trainer_l2 = CustomTrainer(
                model=model_l2,
                args=training_args_l2,
                train_dataset=train_dataset_l2,
                eval_dataset=val_dataset_l2,
                class_weights=class_weights_l2
            )
            trainer_l2.train()
            
            preds_l2 = trainer_l2.predict(val_dataset_l2)
            y_pred_l2 = np.argmax(preds_l2.predictions, axis=1)
            l2_f1 = f1_score(df_val_l2[cfg.LABEL_COLUMN_INPUT].map(label2id_l2).tolist(), y_pred_l2, average='macro', zero_division=0)
            print(f"[INFO] Level 2 Macro F1: {l2_f1:.4f}")
            
            fold_macro_f1 = (l1_f1 + l2_f1) / 2
            
            trainer_l1.save_model(os.path.join(cfg.MODEL_EXPORT_DIR, f"model_hierarchical_lvl1_fold_{fold}"))
            trainer_l2.save_model(os.path.join(cfg.MODEL_EXPORT_DIR, f"model_hierarchical_lvl2_fold_{fold}"))
            
        else:
            print("\n[INFO] --- Training Flat Baseline ---")
            if getattr(cfg, 'USE_UNDERSAMPLING', False):
                df_minority = df_train[df_train[cfg.LABEL_COLUMN_INPUT] != cfg.BINARY_TARGET_LABEL]
                df_majority = df_train[df_train[cfg.LABEL_COLUMN_INPUT] == cfg.BINARY_TARGET_LABEL]
                sample_size = int(len(df_majority) * cfg.UNDERSAMPLING_PERCENTAGE)
                df_majority = df_majority.sample(n=sample_size, random_state=cfg.RANDOM_SEED)
                df_train = pd.concat([df_majority, df_minority]).sample(frac=1, random_state=cfg.RANDOM_SEED)
            
            labels = sorted(df_train[cfg.LABEL_COLUMN_INPUT].unique())
            label2id = {label: i for i, label in enumerate(labels)}
            with open(os.path.join(cfg.MODEL_EXPORT_DIR, "label_mapping_flat.json"), 'w') as f:
                json.dump(label2id, f)
                
            train_enc = tokenizer(df_train[cfg.TEXT_COLUMN_INPUT].tolist(), truncation=True, padding=True, max_length=cfg.MAX_LEN)
            val_enc = tokenizer(df_val[cfg.TEXT_COLUMN_INPUT].tolist(), truncation=True, padding=True, max_length=cfg.MAX_LEN)
            
            train_dataset = IndoBERTDataset(train_enc, df_train[cfg.LABEL_COLUMN_INPUT].map(label2id).tolist())
            val_dataset = IndoBERTDataset(val_enc, df_val[cfg.LABEL_COLUMN_INPUT].map(label2id).tolist())
            
            class_weights = None
            if not getattr(cfg, 'USE_UNDERSAMPLING', False):
                class_weights = compute_class_weight('balanced', classes=np.unique(df_train[cfg.LABEL_COLUMN_INPUT]), y=df_train[cfg.LABEL_COLUMN_INPUT])
            
            model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_CHECKPOINT, num_labels=len(labels))
            model.to(device)
            
            training_args = TrainingArguments(
                output_dir=os.path.join(cfg.MODEL_EXPORT_DIR, f"temp_fold_{fold}"),
                num_train_epochs=cfg.EPOCHS,
                per_device_train_batch_size=cfg.BATCH_SIZE,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                learning_rate=cfg.LEARNING_RATE,
                seed=cfg.RANDOM_SEED,
                report_to="none"
            )
            
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                class_weights=class_weights
            )
            trainer.train()
            
            preds = trainer.predict(val_dataset)
            y_pred = np.argmax(preds.predictions, axis=1)
            fold_macro_f1 = f1_score(df_val[cfg.LABEL_COLUMN_INPUT].map(label2id).tolist(), y_pred, average='macro', zero_division=0)
            
            trainer.save_model(os.path.join(cfg.MODEL_EXPORT_DIR, f"model_flat_fold_{fold}"))

        logger.info(f"Fold {fold} Overall Macro F1: {fold_macro_f1:.4f}")
        fold_metrics[f"fold_{fold}"] = fold_macro_f1
        
        if fold_macro_f1 > best_fold_score:
            best_fold_score = fold_macro_f1
            best_fold_index = fold
            if getattr(cfg, 'USE_HIERARCHICAL', False):
                with open(os.path.join(cfg.MODEL_EXPORT_DIR, cfg.THRESHOLD_FILE), 'w') as f:
                    json.dump({"best_threshold": best_thresh, "target_class_idx": target_class_idx}, f)
        
        for temp_dir in os.listdir(cfg.MODEL_EXPORT_DIR):
            if temp_dir.startswith("temp_fold_"):
                shutil.rmtree(os.path.join(cfg.MODEL_EXPORT_DIR, temp_dir))

    logger.info(f"Best Fold: {best_fold_index} (Macro F1: {best_fold_score:.4f})")
    for f in range(1, cv_folds + 1):
        if getattr(cfg, 'USE_HIERARCHICAL', False):
            lvl1_path = os.path.join(cfg.MODEL_EXPORT_DIR, f"model_hierarchical_lvl1_fold_{f}")
            lvl2_path = os.path.join(cfg.MODEL_EXPORT_DIR, f"model_hierarchical_lvl2_fold_{f}")
            if f == best_fold_index:
                os.rename(lvl1_path, os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_hierarchical_lvl1"))
                os.rename(lvl2_path, os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_hierarchical_lvl2"))
                tokenizer.save_pretrained(os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_hierarchical_lvl1"))
                tokenizer.save_pretrained(os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_hierarchical_lvl2"))
            else:
                if os.path.exists(lvl1_path): shutil.rmtree(lvl1_path)
                if os.path.exists(lvl2_path): shutil.rmtree(lvl2_path)
        else:
            flat_path = os.path.join(cfg.MODEL_EXPORT_DIR, f"model_flat_fold_{f}")
            if f == best_fold_index:
                os.rename(flat_path, os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_flat"))
                tokenizer.save_pretrained(os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_flat"))
            else:
                if os.path.exists(flat_path): shutil.rmtree(flat_path)

    end_time = time.time()
    execution_time = end_time - start_time
    metadata = {
        "start_time": datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
        "duration_seconds": execution_time,
        "best_fold_index": best_fold_index,
        "best_fold_macro_f1": best_fold_score,
        "fold_metrics": fold_metrics,
        "hyperparameters": {
            "max_len": cfg.MAX_LEN,
            "learning_rate": cfg.LEARNING_RATE,
            "batch_size": cfg.BATCH_SIZE,
            "epochs": cfg.EPOCHS,
            "seed": cfg.RANDOM_SEED
        },
        "schema": "Hierarchical" if getattr(cfg, 'USE_HIERARCHICAL', False) else "Flat"
    }
    
    with open(os.path.join(cfg.MODEL_EXPORT_DIR, cfg.METADATA_FILE), 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Training complete. Metadata written to {cfg.METADATA_FILE}")

if __name__ == "__main__":
    train_model()