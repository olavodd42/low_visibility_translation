from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, PreTrainedTokenizerFast, MarianTokenizer, EarlyStoppingCallback
from tokenizers import SentencePieceUnigramTokenizer
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader

import torch
import os
import warnings
import sentencepiece as spm
import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import gc

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Resolver problema do tokenizers|
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Desabilitar alguns warnings
warnings.filterwarnings("ignore")

bleu = evaluate.load("bleu")
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.split() for p in decoded_preds]
    decoded_labels = [[l.split()] for l in decoded_labels]
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["bleu"]}

def preprocess(example):
    inputs = tokenizer(example["en"], truncation=True, padding="max_length", max_length=64)
    targets = tokenizer(example["txu"], truncation=True, padding="max_length", max_length=64)
    inputs["labels"] = targets["input_ids"]
    return inputs
    
# Verifica se há GPU disponível e usa float16 para economizar memória
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = torch.cuda.is_available()

model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

# Carrega dataset já salvo no disco
dataset = load_from_disk("./augmented_dataset")


# Filtra entradas nulas em ambos os splits
dataset["train"] = dataset["train"].filter(lambda x: x["en"] is not None and x["txu"] is not None)
dataset["test"] = dataset["test"].filter(lambda x: x["en"] is not None and x["txu"] is not None)
            

tokenizer = MarianTokenizer.from_pretrained(
    model_name,
    src_lang="en",
    tgt_lang="romance",
    sp_model_file="txu_tokenizer.model",
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)

train_dataset = dataset["train"].map(preprocess, batched=True) \
                             .remove_columns(["en","txu"]) \
                             .shuffle(42).select(range(10000))
test_dataset  = dataset["test"].map(preprocess, batched=True) \
                             .remove_columns(["en","txu"]) \
                             .shuffle(42).select(range(1000))



lora_config_phase1 = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.1,
    bias="none"
)

model_phase1 = get_peft_model(base_model, lora_config_phase1)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=base_model)

clear_memory()

training_args = TrainingArguments(
    output_dir="outputs",
    eval_strategy="no",
    save_strategy="no",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=8, 
    fp16=USE_FP16,
    logging_dir="logs",
    save_total_limit=2,
    num_train_epochs=2,
    report_to="none",
    deepspeed="deepspeed_config.json",
    label_smoothing_factor=0.1,
    load_best_model_at_end=False,
    metric_for_best_model="bleu",
    greater_is_better=True,
    logging_steps=25,
    #eval_strategy="no",
)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
# )

# trainer.train()

trainer_phase1 = Trainer(
    model=model_phase1,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
trainer_phase1.train()
# train_dataset = train_dataset.remove_columns(["en", "txu"])
# test_dataset  = test_dataset.remove_columns(["en", "txu"])
# metrics = trainer_phase1.evaluate(eval_dataset=test_dataset)
dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=data_collator)

model_phase1.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        output = model_phase1.generate(input_ids=input_ids, max_new_tokens=64)
        output = output.cpu()
        
        all_preds.extend(output.tolist())
        all_labels.extend(labels.cpu().tolist())

clear_memory()
decoded_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(all_labels, skip_special_tokens=True)
# decoded_preds = [p.split() for p in decoded_preds]
# decoded_labels = [[l.split()] for l in decoded_labels]
refs = [[ref] for ref in decoded_labels]
trainer_phase1.save_model("outputs/lora_phase1")
tokenizer.save_pretrained("outputs/final_model")

print("BLEU:", bleu.compute(predictions=decoded_preds, references=refs))
history = trainer_phase1.state.log_history
df = pd.DataFrame(history)

# 1) Train Loss vs Step
if "loss" in df.columns:
    plt.figure()
    plt.plot(df["step"], df["loss"])
    plt.xlabel("Step")
    plt.ylabel("Train Loss")
    plt.title("Train Loss ao longo dos steps")
    plt.grid(True)
    plt.show()

# 2) Eval Loss vs Epoch
if "eval_loss" in df.columns:
    eval_df = df.dropna(subset=["eval_loss"])
    plt.figure()
    plt.plot(eval_df["epoch"], eval_df["eval_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    plt.title("Eval Loss por Época")
    plt.grid(True)
    plt.show()

# 3) BLEU vs Epoch
if "bleu" in df.columns:
    bleu_df = df.dropna(subset=["bleu"])
    plt.figure()
    plt.plot(bleu_df["epoch"], bleu_df["bleu"])
    plt.xlabel("Epoch")
    plt.ylabel("BLEU")
    plt.title("BLEU por Época")
    plt.grid(True)
    plt.show()


