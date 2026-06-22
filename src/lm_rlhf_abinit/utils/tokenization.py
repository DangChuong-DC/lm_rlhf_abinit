
def tokenize_ds(batch, tokenizer, text_column="sentence"):
    tokenized_batch = tokenizer(batch[text_column])
    return tokenized_batch
