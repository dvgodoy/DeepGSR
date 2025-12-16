
import os
import numpy as np
import torch
import torch.nn as nn
import evaluate
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm
import math

# Set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(1337)

def main():
    # 1. Load and Prepare Data
    print("Loading dataset...")
    try:
        dataset = load_dataset('dvgodoy/DeepGSR_sequences', split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Filter 'N'
    dataset = dataset.filter(lambda row: 'N' not in row['sequence'])

    # Apply filters - train on ALL organisms
    signal = 'PAS'
    motif = 'AATAAA'
    # organism = 'hs' # REMOVED: Training on ALL organisms now
    
    print(f"Filtering for signal={signal}, motif={motif}...")
    dataset = dataset.filter(lambda row: row['signal'] == signal and row['motif'] == motif)

    # Split dataset
    dataset = dataset.shuffle(seed=19)
    train_test = dataset.train_test_split(test_size=0.25, shuffle=False)
    train_val = train_test['train'].train_test_split(test_size=0.2, shuffle=False)
    dataset = DatasetDict({'train': train_val['train'], 'val': train_val['test'], 'test': train_test['test']})
    print("Dataset split:", dataset)

    # 2. Tokenization with DNABERT-2
    model_name = "zhihan1996/DNABERT-2-117M"
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    def preprocess_function(examples):
        # Construct input: upstream + downstream (removing motif)
        motif_len = len(motif)
        seqs = examples['sequence']
        processed_seqs = []
        for seq in seqs:
            upstream = seq[:300]
            downstream = seq[300+motif_len:]
            processed_seqs.append(upstream + downstream)
        
        return tokenizer(processed_seqs, padding="max_length", truncation=True, max_length=600)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # IMPORTANT: Keep 'organism' for testing logic later, do not remove it yet or hide it via set_format columns if we need to filter
    # However, for training we only need tensor columns. 
    # We will use set_format to specify tensor columns, but 'organism' remains available in the dataset object for filtering.
    
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"]) 

    # 3. Model Setup
    print(f"Loading model {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, trust_remote_code=True)
    model.to(device)

    # 4. Training Setup
    bsize = 16 
    
    # Training loaders
    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=bsize, shuffle=True)
    val_dataloader = DataLoader(tokenized_datasets['val'], batch_size=bsize)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

    num_epochs = 10 
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    # 5. Training Loop
    progress_bar = tqdm(range(num_training_steps))
    best_loss = float('inf')
    best_epoch = -1
    patience = 3 

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, batch['labels'])
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_losses.append(loss.item())
            progress_bar.update(1)

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                loss = loss_fn(logits, batch['labels'])
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_dnabert_model.pth')
            print("Saved best model.")
        elif (epoch - best_epoch) > patience:
            print(f"Early stopping at epoch #{epoch}")
            break

    # 6. Evaluation (Species-specific)
    print("Evaluating with species-specific testing...")
    model.load_state_dict(torch.load('best_dnabert_model.pth'))
    model.eval()

    metric1 = evaluate.load('precision', average=None) 
    metric2 = evaluate.load('recall', average=None)
    metric3 = evaluate.load('accuracy')

    # Replicate the nested loop structure: Split -> Organism
    splits = ['train', 'val', 'test']
    organisms = ['hs', 'bt', 'dm', 'mm']

    for split in splits:
        # Create subsets for each organism in this split
        # Note: Filtering works on the underlying HF dataset even if format is set for tensors
        subsets = [(org, tokenized_datasets[split].filter(lambda row: row['organism'] == org)) for org in organisms]
        
        for org, subset in subsets:
            print(f'Set: {split} / Organism: {org}')
            if len(subset) == 0:
                print("  No samples.")
                continue

            dl = DataLoader(subset, batch_size=bsize)
            
            # Reset metrics for this group
            # Evaluate.load returns a new metric object, or we can assume computation resets internal state?
            # evaluate metrics.compute() resets the state.
            
            with torch.no_grad():
                for batch in tqdm(dl, desc=f"{split}/{org}"):
                    # 'organism' column is not in batch because of set_format columns=['input_ids',...]
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)
                    
                    metric1.add_batch(predictions=preds, references=batch['labels'])
                    metric2.add_batch(predictions=preds, references=batch['labels'])
                    metric3.add_batch(predictions=preds, references=batch['labels'])
            
            # Compute and print metrics
            # Using average=None as in the example to see per-class performance if possible
            # Note: For binary classification with average=None, it returns score for class 0 and class 1
            try:
                prec = metric1.compute(average=None)
                rec = metric2.compute(average=None)
            except:
                # Fallback if average=None is not supported for some reason (though it is for standard metrics)
                prec = metric1.compute()
                rec = metric2.compute()
            
            acc = metric3.compute()
            print(f"{split} {org} Results: Precision: {prec}, Recall: {rec}, Accuracy: {acc}")

if __name__ == "__main__":
    main()
