import os
import gc
import torch
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    MarianTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import PeftModel, get_peft_model, LoraConfig

class SafePeftTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("num_items_in_batch", None)
        return super().compute_loss(model, inputs, return_outputs)

# ------------------- UTILITIES -------------------
def clear_memory():
    """Liberar cache de GPU e coletar lixo."""
    gc.collect()
    torch.cuda.empty_cache()

# ------------------- DEVICE -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP16 = torch.cuda.is_available()

# ------------------- METRIC -------------------
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    """Decodifica predições e labels, formata e calcula BLEU."""
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # para BLEU: lista de tokens e lista de listas
    decoded_preds = [p.split() for p in decoded_preds]
    decoded_labels = [[l.split()] for l in decoded_labels]
    return {"bleu": bleu.compute(predictions=decoded_preds, references=decoded_labels)["bleu"]}

# ------------------- PATHS & LOCALES -------------------
PHASE1_ADAPTER = "outputs/lora_phase1"
PHASE2_OUTPUT   = "outputs/phase2"
TOKENIZER_MODEL = "txu_tokenizer.model"

# ------------------- TOKENIZER -------------------
tokenizer = MarianTokenizer.from_pretrained(
    PHASE1_ADAPTER,
    src_lang="en",
    tgt_lang="romance",
    sp_model_file=TOKENIZER_MODEL
)

# ------------------- LO-RA CONFIG -------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

# ------------------- MODEL LOADING -------------------
# 1) carrega modelo base
base_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")
# 2) aplica LoRA e carrega adapter treinado na fase1
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")
model = PeftModel.from_pretrained(model, PHASE1_ADAPTER)
# 3) move para GPU (se disponível)
model = PeftModel.from_pretrained(model, PHASE1_ADAPTER)
model.enable_adapter_layers()  # ✅ ← isso ativa os parâmetros LoRA
model.to(DEVICE)
model.train()
model.print_trainable_parameters()  # confirmação


# Manually freeze non-LoRA parameters
# for name, param in model.named_parameters():
#     if "lora_" not in name:
#         param.requires_grad = False


# limpa memória antes do treino
clear_memory()

# ------------------- DATASET -------------------
ds = load_from_disk("./augmented_dataset")
ds = ds.filter(lambda ex: ex["en"] and ex["txu"])
train_ds = ds["train"]
eval_ds  = ds["test"]

# preprocess function
def preprocess(examples):
    enc = tokenizer(
        examples["en"], truncation=True,
        padding="max_length", max_length=64
    )
    lbl = tokenizer(
        examples["txu"], truncation=True,
        padding="max_length", max_length=64
    )
    enc["labels"] = lbl["input_ids"]
    # 🧠 Força os labels a serem inteiros
    enc["labels"] = [[int(id) for id in seq] for seq in enc["labels"]]
    return enc

# aplica preprocessamento
train_dataset = train_ds.map(preprocess, batched=True).remove_columns(["en","txu"])
eval_dataset  = eval_ds.map(preprocess,  batched=True).remove_columns(["en","txu"])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ------------------- TRAINING ARGUMENTS -------------------
training_args = TrainingArguments(
    output_dir=PHASE2_OUTPUT,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    fp16=FP16,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    gradient_checkpointing=False,
    remove_unused_columns=False
)

# ------------------- TRAINER -------------------
trainer = SafePeftTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

# ------------------- TRAIN & EVAL -------------------
trainer.train()
clear_memory()
eval_metrics = trainer.evaluate()
print("Fase2 metrics:", eval_metrics)

# ------------------- SAVE OUTPUTS -------------------
trainer.save_model("outputs/final_model")
tokenizer.save_pretrained("outputs/final_model")
pd.DataFrame([eval_metrics]).to_csv(os.path.join(PHASE2_OUTPUT, "phase2_metrics.csv"), index=False)

# ------------------- PLOTS -------------------
hist = pd.DataFrame(trainer.state.log_history)
if "loss" in hist:
    plt.figure(); plt.plot(hist["step"], hist["loss"]); plt.title("Train Loss"); plt.show()
if "eval_loss" in hist:
    df_e = hist.dropna(subset=["eval_loss"])
    plt.figure(); plt.plot(df_e["epoch"], df_e["eval_loss"]); plt.title("Eval Loss"); plt.show()
if "bleu" in hist:
    df_b = hist.dropna(subset=["bleu"])
    plt.figure(); plt.plot(df_b["epoch"], df_b["bleu"]); plt.title("BLEU"); plt.show()
