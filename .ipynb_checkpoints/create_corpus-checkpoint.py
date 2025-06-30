# Exemplo: unir vários arquivos com frases txu

with open("data/kayapo_corpus.txt", "w", encoding="utf-8") as out:
    try:
        with open("data/parallel_clean.txt", encoding="utf-8") as f:
            for line in f:
                if '\t' in line:
                    txu = line.strip().split('\t')[1]
                else:
                    print(f'Error splitting line: {line}')

                out.write(txu.strip() + "\n")
        try:
            with open("data/txu_samples.csv", encoding="utf-8") as f:
                for line in f:
                    if ',' in line:
                        txu = line.strip().split(',')[0]
                    else:
                        print(f'Error splitting line: {line}')
    
                    out.write(txu.strip() + "\n")
        except Exception as e:
              print(f"Erro ao abrir 'data/txu_samples.csv': {e}")

    except Exception as e:
          print(f"Erro ao abrir 'data/parallel_clean.txt': {e}")

