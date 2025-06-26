from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def translate(text: str, max_length=128) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

model_path = "outputs/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

source_text = input("Insert the text in english: ")
translated = translate(source_text)
print("Input:", source_text)
print("Output:", translated)
