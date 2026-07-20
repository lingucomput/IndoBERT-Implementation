import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import configuration as cfg

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Evaluation Compute Device: {device}")

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def predict_batch(texts, model, tokenizer, batch_size=32, desc="Predicting"):
    """Executes batch inference to prevent memory overload with progress tracking."""
    model.eval()
    all_logits = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=cfg.MAX_LEN, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu())
            
    return torch.cat(all_logits, dim=0)

def evaluate_model():
    print(f"\n[INFO] Loading Metadata from {cfg.METADATA_FILE}...")
    metadata_path = os.path.join(cfg.MODEL_EXPORT_DIR, cfg.METADATA_FILE)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Metadata not found. Please run train.py first.")
    
    metadata = load_json(metadata_path)
    print(f"[INFO] Best Fold identified as: {metadata['best_fold_index']}")
    
    test_file_path = os.path.join(cfg.MODEL_EXPORT_DIR, "test_split.csv")
    print(f"[INFO] Transparency Log: System is utilizing isolated pre-cached Test Set from {test_file_path}")
    df_test = pd.read_csv(test_file_path)
    
    df_test[cfg.TEXT_COLUMN_INPUT] = df_test[cfg.TEXT_COLUMN_INPUT].fillna("")
    texts = df_test[cfg.TEXT_COLUMN_INPUT].tolist()
    y_true = df_test[cfg.LABEL_COLUMN_INPUT].tolist()
    
    final_predictions = []

    if cfg.USE_HIERARCHICAL:
        print("\n[INFO] --- Initializing Hierarchical Inference ---")
        
        l1_dir = os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_hierarchical_lvl1")
        l2_dir = os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_hierarchical_lvl2")
        
        tokenizer_l1 = AutoTokenizer.from_pretrained(l1_dir)
        model_l1 = AutoModelForSequenceClassification.from_pretrained(l1_dir).to(device)
        
        tokenizer_l2 = AutoTokenizer.from_pretrained(l2_dir)
        model_l2 = AutoModelForSequenceClassification.from_pretrained(l2_dir).to(device)
        
        l1_map = load_json(os.path.join(cfg.MODEL_EXPORT_DIR, "label_mapping_lvl1.json"))
        l2_map = load_json(os.path.join(cfg.MODEL_EXPORT_DIR, "label_mapping_lvl2.json"))
        id2label_l1 = {v: k for k, v in l1_map.items()}
        id2label_l2 = {v: k for k, v in l2_map.items()}
        
        thresh_data = load_json(os.path.join(cfg.MODEL_EXPORT_DIR, cfg.THRESHOLD_FILE))
        best_thresh = thresh_data["best_threshold"]
        target_class_idx = thresh_data["target_class_idx"]
        print(f"[INFO] Enforcing optimized Level 1 Threshold: {best_thresh:.4f}")
        
        print("\n[INFO] Executing Level 1 (Binary) Predictions...")
        logits_l1 = predict_batch(texts, model_l1, tokenizer_l1, desc="Level 1 Inference")
        probs_l1 = torch.nn.functional.softmax(logits_l1, dim=-1).numpy()
        
        preds_l1_hard = (probs_l1[:, target_class_idx] >= best_thresh).astype(int)
        
        print("\n[INFO] Executing Level 2 (Multiclass) Predictions for non-neutral instances...")
        for i, l1_pred_idx in enumerate(tqdm(preds_l1_hard, desc="Level 2 Routing", unit="sample")):
            if l1_pred_idx == 1:
                predicted_l1_str = "non_neutral" 
            else:
                predicted_l1_str = "neutral"
                
            if predicted_l1_str == "neutral":
                final_predictions.append(cfg.BINARY_TARGET_LABEL)
            else:
                inputs = tokenizer_l2([texts[i]], truncation=True, padding=True, max_length=cfg.MAX_LEN, return_tensors="pt").to(device)
                with torch.no_grad():
                    l2_logits = model_l2(**inputs).logits.cpu()
                l2_pred_idx = torch.argmax(l2_logits, dim=-1).item()
                final_predictions.append(id2label_l2[l2_pred_idx])
                
    else:
        print("\n[INFO] --- Initializing Flat Baseline Inference ---")
        
        model_dir = os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_flat")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        
        flat_map = load_json(os.path.join(cfg.MODEL_EXPORT_DIR, "label_mapping_flat.json"))
        id2label = {v: k for k, v in flat_map.items()}
        
        print("\n[INFO] Executing Flat Predictions...")
        logits = predict_batch(texts, model, tokenizer, desc="Flat Inference")
        pred_indices = torch.argmax(logits, dim=-1).numpy()
        final_predictions = [id2label[idx] for idx in pred_indices]

    print("\n[INFO] Generating Evaluation Metrics...")
    
    unique_labels = sorted(list(set(y_true) | set(final_predictions)))
    
    report_str = classification_report(y_true, final_predictions, labels=unique_labels, digits=4)
    print("\n" + "="*60)
    print("GLOBAL CLASSIFICATION REPORT")
    print("="*60)
    print(report_str)
    
    cm = confusion_matrix(y_true, final_predictions, labels=unique_labels)
    cm_df = pd.DataFrame(
        cm, 
        index=[f"Act: {lbl}" for lbl in unique_labels], 
        columns=[f"Pred: {lbl}" for lbl in unique_labels]
    )
    cm_str = cm_df.to_string()
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX (TABULAR)")
    print("="*60)
    print(cm_str)
    
    with open(os.path.join(cfg.MODEL_EXPORT_DIR, "evaluation_results.txt"), "w") as f:
        f.write("GLOBAL CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(report_str)
        f.write("\n\n")
        f.write("CONFUSION MATRIX\n")
        f.write("="*60 + "\n")
        f.write(cm_str)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f"Confusion Matrix ({'Hierarchical' if cfg.USE_HIERARCHICAL else 'Flat'} Schema)")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.MODEL_EXPORT_DIR, "confusion_matrix.png"), dpi=300)
    print(f"\n[INFO] High-resolution Confusion Matrix saved to confusion_matrix.png")
    
    df_test['Actual_Label'] = y_true
    df_test['Predicted_Label'] = final_predictions
    
    conditions = [
        df_test['Actual_Label'] == df_test['Predicted_Label']
    ]
    choices = [
        "Correct"
    ]
    df_test['Prediction_Status'] = np.select(
        conditions, 
        choices, 
        default="Misclassified: Predicted " + df_test['Predicted_Label'] + " but Actual " + df_test['Actual_Label']
    )
    
    export_cols = [cfg.TEXT_COLUMN_INPUT, 'Actual_Label', 'Predicted_Label', 'Prediction_Status']
    error_analysis_df = df_test[export_cols]
    
    error_path = os.path.join(cfg.MODEL_EXPORT_DIR, "error_analysis_test.xlsx")
    error_analysis_df.to_excel(error_path, index=False, engine='openpyxl')
    print(f"[INFO] Error Analysis successfully exported to {error_path}")

if __name__ == "__main__":
    evaluate_model()