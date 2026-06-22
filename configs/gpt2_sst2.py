MODEL_CONFIG = dict(
    model_name = "gpt2"
)

DATASET_CONFIG = dict(
    dataset_name = "stanfordnlp/sst2",
    dataset_cache_dir = "/DATA01/dc/datasets/HuggingFace/",
    min_sequence_length = 5,
)

SFT_TRAIN_CONFIG = dict(
    batch_size = 32,
    num_epochs = 1,
    learning_rate = 3e-5,
    min_lr = 1e-5,
    log_interval = 100,
    save_dir = "/home/dc/self_studies/nvda_preps/lm_rlhf_abinit/self_pretraineds/",
    new_model_name = "gpt2_sst2_sft",
)
