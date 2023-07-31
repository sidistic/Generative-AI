# 1. prepare dataset
# 2. Load pretrained tokenizer, call it with dataset -> encoding
# 3. Build pyTorch dataset with encodings
# 4. Load pretrained model
# 5. a) load trainer and Train int
#    b) native pytorch training loop

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model,
    training_args,
    train_dataset = tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["validation"],
    data_collator = data_collator,
    tokenizer = tokenizer,
)

trainer.train()