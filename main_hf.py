## ref : https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py

import os
import logging
import datasets
import torch
from datasets import load_dataset
import transformers
from transformers import (
    HfArgumentParser, 
    Seq2SeqTrainingArguments, 
    set_seed
)
from dataclasses  import dataclass, field
from typing import Optional
import sys

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pretraining to which model/config/tokenizer we are going to fine-tune from.
    """

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models)"
            )
        }
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    source_lang: str = field(default='en', metadata={"help": "Source language id for translation."})
    target_lang: str = field(default='de', metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default="wmt14", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default='de-en', metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

# See all possible argument is src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m %d %Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    +f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")
        

# Set seed before initializing model.
set_seed(training_args.seed)

# Get the datasets: you can either provide your own JSON training and evaluation file (see below)
# or just provide the name of the public datasets availabe on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the dataset Hub).
#
# For translation, only JSON files are supported, with one field named "translation" containing two keys for the
# source and target languages (unless you adapt what follows).
#
# In distributed training, the load_dataset function gurantee that only one local process can concurrently
# download the dataset.

# Downloading and loading a dataset from the hub.
raw_datasets = load_dataset(
    data_args.dataset_name,
    data_args.dataset_config_name,
    cache_dir=model_args.cache_dir,
    use_auth_token=True if model_args.use_auth_token else None,
)

print(">> train dataset sample")
for row in raw_datasets['train']:
    print(row)
    break
    