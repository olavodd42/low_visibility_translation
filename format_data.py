import pdftotext
import nlpaug.augmenter.word as naw
from datasets import load_dataset, DatasetDict, concatenate_datasets
import re
from tqdm import tqdm
from langdetect import detect, LangDetectException

def validate_entry(example):
    try:
        if (
            len(example["en"].strip()) < 10 or
            len(example["txu"].strip()) < 10 or
            example["en"].strip().lower() == example["txu"].strip().lower()
        ):
            return False
        lang = detect(example["txu"])
        return lang != "en"
    except:
        return False


def convert_to_txt(pdf_file, txt_file):
    """
    Converts a PDF file to a text file.
    """
    with open(pdf_file, "rb") as f:
        pdf = pdftotext.PDF(f)
    with open(txt_file, "w", encoding="utf-8") as f:
        for page in pdf:
            f.write(page + "\n")


def parallelize(txt1, txt2, res_file):
    """
    Merges two text files line-by-line with tab separation.
    """
    with open(txt1, encoding="utf-8") as f_en, open(txt2, encoding="utf-8") as f_txu:
        eng_lines = f_en.readlines()
        txu_lines = f_txu.readlines()

    with open(res_file, "w", encoding="utf-8") as out:
        for en, txu in zip(eng_lines, txu_lines):
            out.write(f"{en.strip()}\t{txu.strip()}\n")


def clean_parallel(input_path, output_path):
    cleaned = []
    removed = 0
    with open(input_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Filtrando paralelos"):
            if '\t' not in line:
                removed += 1
                continue
            parts = line.strip().split('\t')
            if len(parts) != 2:
                removed += 1
                continue
            en, txu = parts
            if len(en.strip()) < 10 or len(txu.strip()) < 10:
                removed += 1
                continue
            if en.strip().lower() == txu.strip().lower():
                removed += 1
                continue
            try:
                detected_lang = detect(txu)
                if detected_lang == "en":
                    removed += 1
                    continue
            except LangDetectException:
                removed += 1
                continue
            cleaned.append(f"{en.strip()}\t{txu.strip()}")

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(cleaned))

    # Export sample preview
    with open("data/parallel_preview.txt", "w", encoding="utf-8") as sample:
        sample.write("\n".join(cleaned[:10]))

    print(f"Total limpo: {len(cleaned)} | Removidos: {removed}")


def synonym_augmentation(batch):
    aug = naw.SynonymAug(aug_src='wordnet')
    return {
        'en': [aug.augment(text) for text in batch['en']],
        'txu': batch['txu']
    }


def random_augmentation(batch):
    aug = naw.RandomWordAug(action="swap")
    return {
        'en': [aug.augment(text) for text in batch['en']],
        'txu': batch['txu']
    }


def join_en(example):
    if isinstance(example["en"], list):
        return {"en": " ".join(example["en"])}
    return example


convert_to_txt("data/eng-t4t_all.pdf", "data/eng.txt")
convert_to_txt("data/txuNT_all.pdf", "data/txu.txt")
parallelize("data/eng.txt", "data/txu.txt", "data/parallel.txt")

# Limpeza: remove linhas problemáticas do paralelo
data_path = "data/parallel_clean.txt"
clean_parallel("data/parallel.txt", data_path)

data_txt = load_dataset(
    "csv",
    data_files=data_path,
    delimiter="\t",
    column_names=["en", "txu"],
    split="train"
)

extra = load_dataset(
    "csv",
    data_files="data/txu_samples.csv",  # troque pelo nome real
    delimiter=",",
    column_names=["txu", "en"],  # coluna 0 = txu, coluna 1 = en
    split="train",
)

extra = extra.map(lambda ex: {"en": ex["en"], "txu": ex["txu"]})

extra = extra.filter(validate_entry)
data = concatenate_datasets([data_txt, extra])

# Split
data = data.train_test_split(test_size=0.1, seed=42)

print('Train-test sets:')
print(data)

# Initial datasets
train_dataset = data['train']
test_dataset = data['test']

# Apply synonym augmentation
syn_train = train_dataset.map(synonym_augmentation, batched=True, batch_size=32).map(join_en)
syn_test = test_dataset.map(synonym_augmentation, batched=True, batch_size=32).map(join_en)

train_dataset = concatenate_datasets([train_dataset, syn_train])
test_dataset = concatenate_datasets([test_dataset, syn_test])

# Apply random augmentation
rand_train = train_dataset.map(random_augmentation, batched=True, batch_size=32).map(join_en)
rand_test = test_dataset.map(random_augmentation, batched=True, batch_size=32).map(join_en)

train_dataset = concatenate_datasets([train_dataset, rand_train])
test_dataset = concatenate_datasets([test_dataset, rand_test])

print('Augmented train data:')
print(train_dataset)
print('Augmented test data:')
print(test_dataset)

# Final DatasetDict and save
dataset = DatasetDict({
    'train': train_dataset.shuffle(seed=42),
    'test': test_dataset.shuffle(seed=42)
})

dataset.save_to_disk("augmented_dataset")
