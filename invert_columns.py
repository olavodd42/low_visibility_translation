import re

def invert_columns(input_path, output_path):
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            # Detecta tab, ou 2+ espaços
            parts = re.split(r'\t+|\s{2,}', line.strip())
            if len(parts) != 2:
                print(f"⚠️ Ignorado (sem 2 colunas claras): {line.strip()}")
                continue
            txu, en = parts
            fout.write(f"{en.strip()}\t{txu.strip()}\n")

def smart_split_line(line):
    tokens = line.strip().split()
    if len(tokens) < 4:
        return None  # muito curta
    for i in range(1, len(tokens)-1):
        kayapo_part = " ".join(tokens[:i])
        english_part = " ".join(tokens[i:])
        if all(word.isascii() and word[0].isalpha() for word in english_part.split()):
            return english_part, kayapo_part
    return None

def invert_smart(input_path, output_path):
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            result = smart_split_line(line)
            if result:
                en, txu = result
                fout.write(f"{en}\t{txu}\n")
            else:
                print(f"⚠️ Ignorado: {line.strip()}")


# Exemplo de uso
#invert_columns("data/txu_samples.txt", "data/parallel2.txt")
invert_smart("data/txu_samples.txt", "data/parallel2.txt")