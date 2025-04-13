import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BartTokenizer
from torchmetrics.text.rouge import ROUGEScore

#########################################
# Dataset and Collate Function
#########################################

class SummarizationDataset(Dataset):
    def __init__(self, dataset_split, tokenizer, max_input_length=256, max_target_length=128):
        self.dataset = dataset_split
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        source = item['article']
        target = item['highlights']
        source_enc = self.tokenizer(source, truncation=True, padding='max_length',
                                    max_length=self.max_input_length, return_tensors="pt")
        target_enc = self.tokenizer(target, truncation=True, padding='max_length',
                                    max_length=self.max_target_length, return_tensors="pt")
        return {
            'input_ids': source_enc.input_ids.squeeze(0),
            'attention_mask': source_enc.attention_mask.squeeze(0),
            'target_ids': target_enc.input_ids.squeeze(0)
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'target_ids': target_ids}

#########################################
# Synthetic Gradient Module
#########################################
# To reduce computational overhead, we use a simple single-layer module.
class SyntheticGradient(nn.Module):
    def __init__(self, hidden_dim):
        super(SyntheticGradient, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, h):
        return self.linear(h)
    
#########################################
# Seq2Seq Model with Encoder Decoupled via Synthetic Gradients
#########################################
class Seq2SeqGRUModelDecoupled(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx, dropout=0.3):
        super(Seq2SeqGRUModelDecoupled, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        # Synthetic gradient module for the encoder.
        self.sg_module = SyntheticGradient(hidden_dim)
    
    # Encoder forward pass
    def forward_encoder(self, src):
        embedded_src = self.embedding(src)
        embedded_src = self.dropout(embedded_src)
        encoder_outputs, hidden = self.encoder(embedded_src)
        return encoder_outputs, hidden
    
    # Decoder forward pass (receives a hidden state provided externally)
    def forward_decoder(self, trg, hidden):
        embedded_trg = self.embedding(trg)
        embedded_trg = self.dropout(embedded_trg)
        decoder_outputs, _ = self.decoder(embedded_trg, hidden)
        decoder_outputs = self.dropout(decoder_outputs)
        output = self.fc(decoder_outputs)
        return output

    # Inference is unchanged.
    def generate(self, src, sos_token, eos_token, max_len=128, beam_width=3):
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden = self.forward_encoder(src)
            batch_size = src.size(0)
            beams = [[(torch.tensor([sos_token], device=src.device), 0.0, hidden[:, i:i+1, :])]
                     for i in range(batch_size)]
            for _ in range(max_len):
                new_beams = []
                all_finished = True
                for i in range(batch_size):
                    temp_beams = []
                    for seq, score, h in beams[i]:
                        if seq[-1].item() == eos_token:
                            temp_beams.append((seq, score, h))
                            continue
                        last_token = seq[-1].unsqueeze(0).unsqueeze(0)
                        embedded = self.embedding(last_token)
                        embedded = self.dropout(embedded)
                        output, h_new = self.decoder(embedded, h)
                        logits = self.fc(output.squeeze(1))
                        log_probs = torch.log_softmax(logits, dim=-1)
                        topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                        for k in range(beam_width):
                            new_seq = torch.cat([seq, topk_indices[0, k].unsqueeze(0)], dim=0)
                            new_score = score + topk_log_probs[0, k].item()
                            temp_beams.append((new_seq, new_score, h_new))
                    temp_beams = sorted(temp_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                    new_beams.append(temp_beams)
                    if any(b[0][-1].item() != eos_token for b in temp_beams):
                        all_finished = False
                beams = new_beams
                if all_finished:
                    break
            final_outputs = []
            for i in range(batch_size):
                best_seq, best_score, _ = sorted(beams[i], key=lambda x: x[1], reverse=True)[0]
                final_outputs.append(best_seq)
        return final_outputs

#########################################
# Main Pipeline Setup
#########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
pad_idx = tokenizer.pad_token_id
sos_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
eos_token = tokenizer.eos_token_id

# Hyperparameters.
embed_dim = 256
hidden_dim = 512
num_layers = 1
dropout_rate = 0.3
batch_size = 32
num_epochs = 10
learning_rate = 0.003

# Load and reduce the dataset.
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_dataset = dataset["train"].shuffle(seed=42).select(range(len(dataset["train"]) // 2))
val_dataset = dataset["validation"].shuffle(seed=42).select(range(len(dataset["validation"]) // 2))
test_dataset = dataset["test"].shuffle(seed=42).select(range(len(dataset["test"]) // 2))

train_data = SummarizationDataset(train_dataset, tokenizer)
val_data   = SummarizationDataset(val_dataset, tokenizer)
test_data  = SummarizationDataset(test_dataset, tokenizer)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize the model.
vocab_size = tokenizer.vocab_size
model = Seq2SeqGRUModelDecoupled(vocab_size, embed_dim, hidden_dim, num_layers, pad_idx, dropout=dropout_rate).to(device)

# We use separate optimizers:
# - encoder_optimizer updates the encoder and its synthetic gradient module (and optionally shared embeddings)
# - decoder_optimizer updates the decoder and the classification head.
optimizer_encoder = optim.Adam(
    list(model.encoder.parameters()) + list(model.sg_module.parameters()) + list(model.embedding.parameters()),
    lr=learning_rate)
optimizer_decoder = optim.Adam(
    list(model.decoder.parameters()) + list(model.fc.parameters()) + list(model.embedding.parameters()),
    lr=learning_rate)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

#########################################
# Training Loop with Decoupled Encoder Update
#########################################
#
# In each iteration:
# 1. We run a forward pass through the encoder.
# 2. Predict a synthetic gradient for the encoder’s final output.
# 3. Detach encoder output when feeding to decoder (so no backprop flows from the decoder into encoder).
# 4. Update the decoder using normal backprop from the decoder loss.
# 5. Update the encoder “locally” by calling backward on the encoder’s output using the synthetic gradient as the signal.
#
# Note: This completely bypasses standard backpropagation through the encoder.
#########################################

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch in train_loader:
        # Zero both optimizers.
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # -------------------------
        # Encoder Forward Pass
        # -------------------------
        encoder_outputs, hidden = model.forward_encoder(input_ids)
        # Use the encoder's last time-step output for synthetic gradient prediction.
        encoder_last = encoder_outputs[:, -1, :]  # shape: (batch, hidden_dim)
        sg_pred = model.sg_module(encoder_last)
        
        # -------------------------
        # Decoder Forward Pass using Decoupled Encoder Hidden State
        # -------------------------
        # Detach the encoder's hidden state to prevent gradients flowing back.
        hidden_detached = hidden.detach()
        decoder_output = model.forward_decoder(target_ids[:, :-1], hidden_detached)
        loss_decoder = criterion(decoder_output.reshape(-1, decoder_output.shape[-1]),
                                   target_ids[:, 1:].reshape(-1))
        # Update decoder parameters with normal backprop.
        loss_decoder.backward()
        optimizer_decoder.step()
        
        # -------------------------
        # Encoder Update via Synthetic Gradient
        # -------------------------
        # Instead of receiving a gradient from the decoder loss, update the encoder by
        # "injecting" the synthetic gradient. Here we simulate a loss by taking the dot product
        # of the encoder output and its synthetic gradient.
        optimizer_encoder.zero_grad()
        synthetic_loss = (encoder_last * sg_pred).sum()
        synthetic_loss.backward()
        optimizer_encoder.step()
        
        total_loss += loss_decoder.item()  # monitoring decoder loss
        
    end_time = time.time()
    print("Epoch {}/{} - Decoder Loss: {:.4f}, Time: {:.2f}s".format(
          epoch+1, num_epochs, total_loss/len(train_loader), end_time-start_time))
    
#########################################
# Final Evaluation on the Test Set
#########################################
print("Final evaluation on test set:")
model.eval()
rouge = ROUGEScore()
predictions = []
references = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        outputs = model.generate(input_ids, sos_token, eos_token, beam_width=3)
        for pred_ids, tgt_ids in zip(outputs, target_ids):
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            tgt_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)
            predictions.append(pred_text)
            references.append(tgt_text)
scores = rouge(predictions, references)
print("Evaluation ROUGE scores:", scores)

