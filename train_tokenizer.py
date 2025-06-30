import sentencepiece as spm


spm.SentencePieceTrainer.Train(
    input="data/kayapo_corpus.txt",
    model_prefix="txu_tokenizer",
    vocab_size=2000,
    model_type="unigram",
    character_coverage=1.0,        # cobre todos os caracteres
    bos_id=1,
    eos_id=2,
    unk_id=0,
    pad_id=3
)

# # Ajuste as extensões conforme geradas (pode ser .spm e .vocab):


# 