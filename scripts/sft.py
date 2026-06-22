from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from lm_rlhf_abinit.utils.config import load_python_config
from lm_rlhf_abinit.utils.tokenization import tokenize_ds

### CONFIGURATIONS
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_python_config(REPO_ROOT / "configs" / "gpt2_sst2.py")
MODEL_CONFIG = CONFIG["MODEL_CONFIG"]
DATASET_CONFIG = CONFIG["DATASET_CONFIG"]
TRAINING_CONFIG = CONFIG["SFT_TRAIN_CONFIG"]


def save_model(model, save_dir, model_name) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir / model_name)


def evaluate(model, val_loader, device, epoch) -> None:
    logger.info(f"Evaluating on validation set after {epoch + 1} epoch(s)")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_loss} after training for {epoch + 1} epoch(s)")


def train(
    model, train_loader, optimizer, scheduler, device, epoch, log_interval=100
) -> None:
    logger.info(f"Epoch {epoch + 1} - Training")
    model.train()
    for step, batch in enumerate(tqdm(train_loader)):
        batch = batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (step == 0) or ((step + 1) % log_interval) == 0:
            tqdm.write(f"- Step {step + 1} | LR: {scheduler.get_last_lr()[0]:.6f} ---> Loss: {loss.item():.4f}")


def main() -> None:
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
    lang_model = AutoModelForCausalLM.from_pretrained(MODEL_CONFIG["model_name"])
    lang_model.loss_type = "ForCausalLM"
    lang_model.config.loss_type = "ForCausalLM"

    ## define the pad token if it is not defined in the tokenizer
    ## in this case, we will use the EOS (End of Sequence) token as the pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        lang_model.config.pad_token_id = tokenizer.pad_token_id

    # load dataset
    dataset = load_dataset(
        DATASET_CONFIG["dataset_name"],
        cache_dir=DATASET_CONFIG["dataset_cache_dir"],
    )
    ds_train, ds_val = dataset["train"], dataset["validation"]

    ## intialize the tokenizer function as partial function so that 
    ## we can pass it to the map function of the HF dataset
    tokenize_fn = partial(
        tokenize_ds,
        tokenizer=tokenizer,
    )

    map_kwargs = dict(
        batched=True,
        batch_size=512,
        remove_columns=['idx', 'sentence', 'label'],
    )
    ds_train = ds_train.map(
        tokenize_fn,
        **map_kwargs,
    )
    ds_val = ds_val.map(
        tokenize_fn,
        **map_kwargs,
    )

    ### filter out samples with length less than MIN_LENGTH tokens
    ds_train = ds_train.filter(
        lambda x: len(x['input_ids']) >= DATASET_CONFIG["min_sequence_length"]
    )
    ds_val = ds_val.filter(
        lambda x: len(x['input_ids']) >= DATASET_CONFIG["min_sequence_length"]
    )
    ### set dataset format to PyTorch tensors
    ds_train.set_format(type='torch')
    ds_val.set_format(type='torch')

    # prepare the dataloaders
    ## define the collate function to pad the sequences in the batch
    collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Prepare for training loop
    optimizer = AdamW(lang_model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    num_epochs = TRAINING_CONFIG["num_epochs"]
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs*len(train_loader),
        eta_min=TRAINING_CONFIG["min_lr"],
    )

    _device = "cuda:3" if torch.cuda.is_available() else "cpu"
    lang_model.to(_device)

    # Training loop
    evaluate(lang_model, val_loader, _device, -1)
    for epoch in range(num_epochs):
        train(
            lang_model, train_loader, optimizer, scheduler, _device, epoch, 
            log_interval=TRAINING_CONFIG["log_interval"]
        )
        evaluate(lang_model, val_loader, _device, epoch)
        save_model(
            lang_model, 
            TRAINING_CONFIG["save_dir"], 
            TRAINING_CONFIG["new_model_name"]
        )


if __name__ == "__main__":
    main()
