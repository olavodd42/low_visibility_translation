import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_from_disk
import os
import warnings
import sentencepiece as spm


# Resolver problema do tokenizers|
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Desabilitar alguns warnings
warnings.filterwarnings("ignore")

# Verifica se há GPU disponível e usa float16 para economizar memória
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = torch.cuda.is_available()

model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

# Carrega dataset já salvo no disco
dataset = load_from_disk("./augmented_dataset")


# Filtra entradas nulas em ambos os splits
dataset["train"] = dataset["train"].filter(lambda x: x["en"] is not None and x["txu"] is not None)
dataset["test"] = dataset["test"].filter(lambda x: x["en"] is not None and x["txu"] is not None)

# Junta todos os textos em txu.txt para treinar tokenizer
with open("data/txu.txt", encoding="utf-8") as fin, open("data/txu_corpus.txt", "w", encoding="utf-8") as fout:
    for line in fin:
        if line.strip():
            fout.write(line.strip() + "\n")

# Treinamento com vocabulário reduzido (ajuste voc_size se necessário)
spm.SentencePieceTrainer.Train(
    input='data/txu_corpus.txt',
    model_prefix='txu_tokenizer',
    vocab_size=16000,
    model_type='bpe', # ou 'unigram'
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

# Tokeniza os exemplos
def preprocess(example):
    inputs = tokenizer(example["en"], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(example["txu"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = dataset["train"].map(preprocess, batched=True)
test_dataset = dataset["test"].map(preprocess, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="outputs",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    fp16=USE_FP16,
    logging_dir="logs",
    save_total_limit=2,
    num_train_epochs=3,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

metrics = trainer.evaluate()
print("Avaliação no conjunto de teste:")
print(metrics)

model.save_pretrained("outputs/final_model")
tokenizer.save_pretrained("outputs/final_model")

