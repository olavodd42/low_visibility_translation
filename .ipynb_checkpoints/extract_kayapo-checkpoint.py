import fitz           # pip install PyMuPDF
import re

def extract_kayapo(pdf_path, out_txt):
    doc = fitz.open(pdf_path)
    kayapo_lines = []
    # Regex que captura palavras com apóstrofo ou acentos típicos
    pattern = re.compile(r"[a-zA-ZỳỹãẽĩõũôêáéíóúÀ-ÖØ-öø-ÿʼ']+")

    for page in doc:
        text = page.get_text("text").splitlines()
        for line in text:
            # Filtra por linhas que tenham ao menos 2 tokens com caracteres Kayapó
            tokens = line.strip().split()
            if len(tokens) >= 2 and sum(bool(pattern.search(tok)) for tok in tokens) >= 2:
                # Exclui linhas em inglês/glose — basta bloquear linhas com apóstrofo de glossas
                if not re.search(r"'[A-Za-z ]+$", line):
                    kayapo_lines.append(line.strip())

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(kayapo_lines))

# Uso
extract_kayapo("data/KPSemCls.pdf", "kayapo_examples.txt")
print("Extração concluída! Veja kayapo_examples.txt")
