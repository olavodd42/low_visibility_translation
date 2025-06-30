from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
import torch


def translate(text: str, max_length=128) -> str:
    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=True,
                       padding="longest",
                       max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,              # faz busca em 4 hipóteses
            early_stopping=True,
            length_penalty=1.2,       # penaliza traduções curtas
            no_repeat_ngram_size=2    # evita repetir o mesmo n‑gram
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

model_path = "outputs/final_model"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

source_text = input("Insert the text in english: ")
print("Output:", translate(source_text))

