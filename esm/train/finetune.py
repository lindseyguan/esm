from datetime import datetime
import os

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict, Dataset
from esm.models.esm3 import ESM3
from huggingface_hub import login
from transformers import Trainer, TrainingArguments

import torch
from torch import nn
from tqdm import tqdm

from train import ESMTrainer

os.environ["WANDB_PROJECT"] = "esm3_finetune"
os.environ["DISABLE_ITERATIVE_SAMPLING_TQDM"] = "False"

def generate_run_name(num_train_epochs, 
                      learning_rate, 
                      batch_size,
                      output_root,
                      ) -> dict:
    """
    Given a config, generate string representing run
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"ep{num_train_epochs}_lr{learning_rate}_bs{batch_size}_{timestamp}"
    output = os.path.join(output_root, run_name)
    return output


def count_trainable_params(model: ESM3) -> int:
    """
    Counts number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


### Load dataset ###
data_files = {"train": "/mnt/shared/pepmlm/train.csv", 
              "val": "/mnt/shared/pepmlm/val.csv", 
              "test": "/mnt/shared/pepmlm/test.csv"}
dataset = load_dataset('csv', data_files=data_files)

### Initialize config ###
num_train_epochs = 5
learning_rate = 0.0001
batch_size = 4
output_root = './params'

output_dir = generate_run_name(num_train_epochs,
                               learning_rate,
                               batch_size,
                               output_root
                              )

training_args = TrainingArguments(output_dir,
                                  num_train_epochs=num_train_epochs,
                                  learning_rate=learning_rate,
                                  save_strategy='epoch',
                                  save_only_model=True,
                                  per_device_train_batch_size=batch_size,
                                  eval_strategy="epoch",
                                  do_eval=True,
                                  eval_on_start=False,
                                  per_device_eval_batch_size=batch_size,
                                  logging_strategy="steps",
                                  logging_steps=32,
                                  remove_unused_columns=False,
                                  lr_scheduler_type='cosine',
                                  warmup_ratio=0.05,
                                )


### Load model ###
# Fine-tune later transformer blocks and output heads
# TODO later: LoRA for trunk finetuning?
model = ESM3.from_pretrained("esm3-open")
unfreeze = ['output_heads', 
            'transformer.blocks.4', 
            'transformer.norm.weight'] 
for name, param in model.named_parameters():
    for u in unfreeze:
        if u in name:
            param.requires_grad = True
            break
    else:
        param.requires_grad = False

print(f"# Trainable Parameters: {count_trainable_params(model)}")
trainer = ESMTrainer(model=model,
                     training_args=training_args,
                     data=dataset,
                     # Masked positions during loss calculation are 0 
                     # while the dimension is the same, so reduction='sum'
                     loss_func=nn.CrossEntropyLoss(reduction='sum'),
                     device='cuda:0',
                     crop_size=512
                    )
trainer.train()
