import os

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict, Dataset
from huggingface_hub import login
from esm.models.esm3 import ESM3
from transformers import Trainer, TrainingArguments

from datetime import datetime
import torch
from torch import nn
from tqdm import tqdm

from train import ESMTrainer

os.environ["WANDB_PROJECT"] = "pepmlm_finetune"
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
num_train_epochs = 4
learning_rate = 1e-2
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
                                  save_strategy='no',
                                  per_device_train_batch_size=4,
                                  eval_strategy="steps",
                                  eval_steps=500,
                                  do_eval=True,
                                  eval_on_start=False,
                                  per_device_eval_batch_size=4,
                                  logging_strategy="steps",
                                  logging_steps=100,
                                  remove_unused_columns=False,
                                  # torch_empty_cache_steps=1,
                                )


### Load model ###
# Fine-tune sequence head only -- TODO later: LoRA for trunk finetuning?
model = ESM3.from_pretrained("esm3-open")
for name, param in model.named_parameters():
    print(name, param)
    if 'sequence' in name or 'transformer.blocks.47' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

print(f"# Trainable Parameters: {count_trainable_params(model)}")

exit()
trainer = ESMTrainer(model=model,
                     training_args=training_args,
                     data=dataset,
                     loss_func=nn.CrossEntropyLoss(),
                     device='cuda:0',
                     crop_size=512
                    )
trainer.train()
