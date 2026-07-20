import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import configuration as cfg
from preprocessing import clean_text

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def predict_batch_transformers(texts, model, tokenizer, batch_size=16):
    model.eval()
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, 
            truncation=True, 
            padding=True, 
            max_length=getattr(cfg, 'MAX_LEN', 128), 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            
    return np.concatenate(all_probs, axis=0)

def execute_inference(texts):
    if not texts:
        raise ValueError("[ERROR] Input sequence array is empty. Aborting computation.")
        
    df = pd.DataFrame({'original_text': texts})
    print(f"[INFO] Initializing Inference Pipeline. Schema: {'Hierarchical' if getattr(cfg, 'USE_HIERARCHICAL', False) else 'Flat'}")
    print(f"[INFO] Hardware Accelerator Bound: {device}")
    
    df['processed_text'] = df['original_text'].apply(clean_text)
    clean_texts_list = df['processed_text'].tolist()
    
    final_predictions = np.empty(len(clean_texts_list), dtype=object)
    confidence_scores = np.zeros(len(clean_texts_list), dtype=float)
    
    if getattr(cfg, 'USE_HIERARCHICAL', False):
        l1_dir = os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_hierarchical_lvl1")
        l2_dir = os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_hierarchical_lvl2")
        threshold_path = os.path.join(cfg.MODEL_EXPORT_DIR, cfg.THRESHOLD_FILE)
        
        if not (os.path.exists(l1_dir) and os.path.exists(l2_dir)):
            raise FileNotFoundError("[ERROR] Hierarchical model artifacts missing. Verify train.py execution.")
            
        print("[INFO] Loading Level 1 Sub-architecture...")
        tokenizer_l1 = AutoTokenizer.from_pretrained(l1_dir)
        model_l1 = AutoModelForSequenceClassification.from_pretrained(l1_dir).to(device)
        
        l1_map = load_json(os.path.join(cfg.MODEL_EXPORT_DIR, "label_mapping_lvl1.json"))
        id2label_l1 = {v: k for k, v in l1_map.items()}
        
        thresh_data = load_json(threshold_path)
        best_thresh = thresh_data.get("best_threshold", 0.5)
        target_class_idx = thresh_data.get("target_class_idx", 1)
        
        probs_l1 = predict_batch_transformers(clean_texts_list, model_l1, tokenizer_l1, batch_size=getattr(cfg, 'BATCH_SIZE', 16))
        probs_non_neutral = probs_l1[:, target_class_idx]
        
        neutral_mask = probs_non_neutral < best_thresh
        final_predictions[neutral_mask] = cfg.BINARY_TARGET_LABEL
        confidence_scores[neutral_mask] = np.max(probs_l1[neutral_mask], axis=1)
        
        non_neutral_mask = ~neutral_mask
        if non_neutral_mask.sum() > 0:
            print(f"[INFO] Routing {non_neutral_mask.sum()} instances to Level 2 Sub-architecture...")
            tokenizer_l2 = AutoTokenizer.from_pretrained(l2_dir)
            model_l2 = AutoModelForSequenceClassification.from_pretrained(l2_dir).to(device)
            
            l2_map = load_json(os.path.join(cfg.MODEL_EXPORT_DIR, "label_mapping_lvl2.json"))
            id2label_l2 = {v: k for k, v in l2_map.items()}
            
            l2_texts = np.array(clean_texts_list)[non_neutral_mask].tolist()
            probs_l2 = predict_batch_transformers(l2_texts, model_l2, tokenizer_l2, batch_size=getattr(cfg, 'BATCH_SIZE', 16))
            
            pred_indices_l2 = np.argmax(probs_l2, axis=1)
            final_predictions[non_neutral_mask] = [id2label_l2[idx] for idx in pred_indices_l2]
            confidence_scores[non_neutral_mask] = np.max(probs_l2, axis=1)
            
        df['l1_non_neutral_probability'] = probs_non_neutral
        
    else:
        model_dir = os.path.join(cfg.MODEL_EXPORT_DIR, "model_best_flat")
        if not os.path.exists(model_dir):
            raise FileNotFoundError("[ERROR] Flat model artifacts missing. Verify train.py execution.")
            
        print("[INFO] Loading Flat Baseline Architecture...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        
        flat_map = load_json(os.path.join(cfg.MODEL_EXPORT_DIR, "label_mapping_flat.json"))
        id2label = {v: k for k, v in flat_map.items()}
        
        probs = predict_batch_transformers(clean_texts_list, model, tokenizer, batch_size=getattr(cfg, 'BATCH_SIZE', 16))
        pred_indices = np.argmax(probs, axis=1)
        
        final_predictions = [id2label[idx] for idx in pred_indices]
        confidence_scores = np.max(probs, axis=1)
        
    df['predicted_label'] = final_predictions
    df['confidence_score'] = confidence_scores
    
    return df

if __name__ == "__main__":
    sample_texts = [
        "saya tidak mengerti dengan cara kerja sistem ini.",
        "dasar orang bodoh, tidak tahu aturan sama sekali!",
        "pemerintah seharusnya lebih tanggap terhadap isu ekonomi saat ini.",
        "aku akan mencari dan menghabisimu jika kau berani muncul lagi."
    ]
    
    try:
        results = execute_inference(sample_texts)
        print("\n" + "=" * 80)
        print(" INFERENCE RESULTS ".center(80, "="))
        print("=" * 80)
        
        display_columns = ['original_text', 'predicted_label', 'confidence_score']
        print(results[display_columns].to_string(index=False))
        
        export_path = os.path.join(cfg.MODEL_EXPORT_DIR, "inference_results.csv")
        results.to_csv(export_path, index=False)
        print(f"\n[INFO] Analytical output strictly exported to {export_path}")
        
    except Exception as e:
        print(f"\n[FATAL] Inference operation halted computationally: {e}")