# evaluate_model.py

import os
import gc
import torch
import evaluate
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, MarianTokenizer, DataCollatorForSeq2Seq
from datasets import load_from_disk

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

# 1) Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Carrega modelo e tokenizer
model_path = "./outputs/lora_phase1"
tokenizer = MarianTokenizer.from_pretrained(
    model_path,
    src_lang="en",
    tgt_lang="romance",
    sp_model_file="txu_tokenizer.model",
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
model.eval()

# 3) Prepara dataset de teste
ds = load_from_disk("./augmented_dataset")["test"]
ds = ds.filter(lambda x: x["en"] and x["txu"])  # filtra nulos

# aplica tokenização + labels
def preprocess(ex):
    enc = tokenizer(ex["en"], truncation=True, padding="max_length", max_length=64)
    lbl = tokenizer(ex["txu"], truncation=True, padding="max_length", max_length=64)
    enc["labels"] = lbl["input_ids"]
    return enc

ds = ds.map(preprocess, batched=False)
ds = ds.remove_columns(["en","txu"])
ds = ds.shuffle(seed=42).select(range(500))  # ou todo o split

# 4) DataCollator e DataLoader
collator = DataCollatorForSeq2Seq(tokenizer, model=model)
dataloader = DataLoader(ds, batch_size=1, collate_fn=collator)

# 5) Inferência e coleta
bleu = evaluate.load("bleu")
all_preds, all_refs = [], []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Gerando traduções"):
        input_ids = batch["input_ids"].to(DEVICE)
        labels   = batch["labels"].to(DEVICE)

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            num_beams=4,
            length_penalty=1.2,
            no_repeat_ngram_size=2
        )
        # decodifica strings
        preds = tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)
        refs  = tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True)

        all_preds.extend(preds)
        all_refs.extend(refs)

# 6) Calcular BLEU
references = [[r] for r in all_refs]
bleu_score = bleu.compute(predictions=all_preds, references=references)
print(f"BLEU on test subset: {bleu_score['bleu']:.4f}")

# 7) Salvar detalhes em CSV
df = pd.DataFrame({
    "reference": all_refs,
    "prediction": all_preds,
})
out_csv = "evaluation_results.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"Saved detailed results to {out_csv}")
