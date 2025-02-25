import json
import os

import random
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

from datasets.dataset_dict import DatasetDict, Dataset
from esm.models.esm3 import ESM3, ESMProtein, LogitsConfig
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.constants import esm3 as C
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_utils import EvalLoopOutput, PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import ExportableState

import torch
from torch import nn
import torch.nn.functional as F


TOKEN_TO_ID = {tok: ind for ind, tok in enumerate(C.SEQUENCE_VOCAB)}
TOKEN_TO_ID['_'] = TOKEN_TO_ID['<mask>']

ID_TO_TOKEN = {ind: tok for ind, tok in enumerate(C.SEQUENCE_VOCAB)}

def seq_collator(features):
    batch = {}
    for batch_dict in features:
        for k in batch_dict:
            if k not in batch:
                batch[k] = []
            batch[k].append(batch_dict[k])
    return batch


def featurize(batch,
              model,
              crop_size: int = 512,
             ):
    """
    Featurizes samples in batch.
    """
    feat_batch = []
    for seq, rec in zip(batch['Sequence'], batch['Receptor Sequence']):
        mask_token = '_'
        seq1 = rec
        seq2 = mask_token*len(seq)

        order = random.choice([0, 1])
        if order:
            complex_seq = f'{seq1}|{seq2}'[-crop_size:]
            label_seq = f'{seq1}|{seq}'[-crop_size:]
        else:
            complex_seq = f'{seq2}|{seq1}'[:crop_size]
            label_seq = f'{seq}|{seq1}'[:crop_size]

        encoded = model.encode(ESMProtein(sequence=complex_seq))
        mask = (encoded.sequence == TOKEN_TO_ID[mask_token]).cpu()
        label = model.encode(ESMProtein(sequence=label_seq)).sequence.cpu()
        ft_dict = {'input': encoded, 'label': label, 'mask': mask, 
                   'debug_seq': label_seq, 'debug_mask': complex_seq
                   }
        feat_batch.append(ft_dict)
    return feat_batch

class ESMTrainer(Trainer):
    def __init__(
        self,
        model: ESM3,
        training_args: TrainingArguments,
        data: DatasetDict,
        loss_func: Callable = nn.CrossEntropyLoss(),
        device: str = 'cuda:0',
        crop_size: int = 512
    ):
        self.device = device
        self.model = model
        self.model.to(device=device) # Move to device
        self.training_args = training_args
        self.tokenize = EsmSequenceTokenizer() # global function
        self.featurize = featurize # global function
        self.data = data
        self.loss_func = loss_func
        self.crop_size = crop_size
        self.logits_config = LogitsConfig(sequence=True, 
                                          return_embeddings=False, return_hidden_states=False)

        self.data_featurized = {}

        self._stored_train_loss = torch.tensor(0.0).cpu()
        
        for split in ['train', 'val']:
            featurized = self.featurize(self.data[split],
                                        model=self.model, 
                                        crop_size=self.crop_size
                                       )
            self.data_featurized[split] = featurized

        super().__init__(
            model=model,
            args=training_args,
            data_collator=seq_collator,
            train_dataset=self.data_featurized['train'],
            eval_dataset=self.data_featurized['val'],
        )


    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.
        Uses accumulated loss.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        if "loss" in logs: # Only if train loss
            logs["loss"] = self._stored_train_loss
        self._stored_train_loss = torch.tensor(0.0).cpu() # Clear accumulated loss

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


    def compute_loss(self,
                     model,  
                     inputs, 
                     return_outputs=False,
                     num_items_in_batch=None,
                    ):
        # Make prediction
        # with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = []
        for i, protein_tensor in enumerate(inputs['input']):
            output = model.logits(protein_tensor, self.logits_config, grad=True).logits.sequence
            pred_softmax = F.softmax(output, dim=-1)
            pred_softmax = pred_softmax.squeeze(0)
            outputs.append(pred_softmax)

        masks = torch.cat(inputs['mask'], dim=0).unsqueeze(1)
        outputs = torch.cat(outputs, dim=0)
        masked_outputs = torch.where(masks, outputs, torch.tensor(0)).to(torch.float64)

        labels = torch.cat(inputs['label'], dim=0)
        one_hot = F.one_hot(labels, num_classes=64)
        masked_labels = torch.where(masks, one_hot, torch.tensor(0)).to(torch.float64)

        # normalize loss by unmasked length (this is why reduction should be "sum")
        num_masked = torch.sum(masks).to(torch.float64)
        loss = self.loss_func(masked_outputs, masked_labels) / num_masked
        self._stored_train_loss += loss.cpu() # Update accumulated loss

        # Profile memory usage
        # total_free_memory, _ = torch.cuda.mem_get_info(self.device)
        # total_gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
        # print(total_free_memory / 10**9, total_gpu_memory / 10**9)
        return (loss, masked_outputs) if return_outputs else loss


    def eval_loss(self,
                  model,  
                  inputs, 
                  return_outputs=False,
                  num_items_in_batch=None
                 ):
        outputs = []
        for i, protein_tensor in enumerate(inputs['input']):
            output = model.logits(protein_tensor, self.logits_config).logits.sequence
            pred_softmax = F.softmax(output, dim=-1)
            pred_softmax = pred_softmax.squeeze(0)
            outputs.append(pred_softmax)
        
        masks = torch.cat(inputs['mask'], dim=0).unsqueeze(1)
        outputs = torch.cat(outputs, dim=0)
        masked_outputs = torch.where(masks, outputs, torch.tensor(0))

        labels = torch.cat(inputs['label'], dim=0)
        one_hot = F.one_hot(labels, num_classes=64)
        masked_labels = torch.where(masks, one_hot, torch.tensor(0)).to(torch.float16)

        num_masked = torch.sum(masks).to(torch.float16)
        with torch.no_grad():
            # loss = self.loss_func(masked_outputs, masked_labels)
            loss = self.loss_func(masked_outputs, masked_labels) / num_masked

        return (loss, masked_outputs) if return_outputs else loss


    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only=False,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        """
        batch_size = self.training_args.eval_batch_size

        self.model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        global_loss = torch.tensor(0.0).cpu()
        count = 0
        all_outputs = []
        for batch in dataloader:
            if prediction_loss_only:
                loss = self.eval_loss(self.model, 
                                      batch, 
                                      return_outputs=False).cpu()
                global_loss += loss
                outputs = None
            else:
                loss, outputs = self.eval_loss(self.model, 
                                               batch, 
                                               return_outputs=True).cpu()
                global_loss += loss
                all_outputs.extend(outputs)
            count += 1
            if count > 50:
                break

        return EvalLoopOutput(predictions=all_outputs,
                              label_ids=None,
                              metrics={'eval_loss':global_loss},
                              num_samples=len(dataloader)
                             )


    def _save_metrics(self, metrics, output_dir):
        """
        Args:
            metrics (`Dict[str, float]`)
            output_dir (`str`)
        Return: None
        """
        output_dict = {k: (v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v) for k, v in metrics.items()}
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(output_dict, f)


    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        # print('Saving checkpoint to', output_dir)
        self.save_model(output_dir, _internal_call=True)
        if not metrics is None:
            self._save_metrics(metrics, output_dir)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # The huggingface version of this function also saves the Trainer state
        # and maybe removes some older checkpoints.
        # But that kept not working for me because the self.state.save_to_json()
        # call was crashing out (Tensor not JSON serializable)

