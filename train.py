import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import os
import json
import torch
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
from PIL import ImageFile
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)

from gui_actor.dataset import LazySupervisedDataset
from gui_actor.trainer import AGUVISTrainer, rank0_print, safe_save_model_for_hf_trainer
from gui_actor.utils import dump_args_to_json

from gui_actor.constants import (
    IGNORE_INDEX,
    ADDITIONAL_SPECIAL_TOKENS,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
)

from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer

apply_liger_kernel_to_qwen2_vl()

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attn_2_enabled: bool = field(default=True)
    model_type: str = field(default="qwen2vl", metadata={"help": "model type: qwen2vl or qwen25vl"})

@dataclass
class DataArguments:
    data_path: str = field(default=None)
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    min_pixels: Optional[int] = field(default=3136) # 2 * 2 * 28 * 28 = 56 * 56
    max_pixels: Optional[int] = field(default=5720064) # 5720064 = 114 * 64 * 28 * 28 = 3192 * 1792, 12845056 = 128 * 128 * 28 * 28
    max_conv_turns: Optional[int] = field(default=10) # 30 => 20 => 10


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    group_by_modality_length: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    
    unfreeze_all_parameters: bool = field(default=False)
    unfreeze_pointer_head: bool = field(default=True)
    unfreeze_lm_head: bool = field(default=False)
    unfreeze_base_model: bool = field(default=False)
    unfreeze_last_n_layers: int = field(default=-1)
    unfreeze_new_tokens: bool = field(default=True)
    unfreeze_visual: bool = field(default=False)
    pointer_loss_weight: float = field(default=0.1)
    lm_loss_weight: float = field(default=-1.0)

# def mask_embedding_grad(grad):
#     n_new_tokens = len(ADDITIONAL_SPECIAL_TOKENS)
#     mask = torch.zeros_like(grad)
#     mask[-n_new_tokens:] = 1.0
#     return grad * mask

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    new_vocab_size = len(tokenizer)    
    # Update base model and current model config
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = new_vocab_size
    else:
        model.config.vocab_size = new_vocab_size
    model.vocab_size = new_vocab_size

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def update_pointer_token_ids(model_config: transformers.PretrainedConfig, tokenizer: transformers.PreTrainedTokenizer):
    model_config.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
    model_config.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
    model_config.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
    rank0_print(f"Updated pointer token ids: {model_config.pointer_pad_token_id}, {model_config.pointer_start_token_id}, {model_config.pointer_end_token_id}")

def setup_params_to_update(model: transformers.PreTrainedModel, training_args: TrainingArguments):
    if training_args.unfreeze_all_parameters:
        rank0_print(f"Unfreezing all model parameters...")
        for p in model.parameters():
            p.requires_grad = True
    else:
        rank0_print(f"Freezing all model parameters...")
        for p in model.parameters():
            p.requires_grad = False

        if training_args.unfreeze_pointer_head:
            rank0_print(f"Unfreezing pointer head parameters...")
            # for p in model.pointer_head.parameters():
            #     p.requires_grad = True
            for p in model.multi_patch_pointer_head.parameters():
                p.requires_grad = True

        if training_args.unfreeze_lm_head:
            rank0_print(f"Unfreezing lm head parameters...")
            for p in model.lm_head.parameters():
                p.requires_grad = True
        
        if training_args.unfreeze_base_model: # including text tokens
            rank0_print(f"Unfreezing base model parameters...")
            for p in model.model.parameters():
                p.requires_grad = True

        if training_args.unfreeze_last_n_layers > 0:
            rank0_print(f"Unfreezing last {training_args.unfreeze_last_n_layers} layers of base model parameters...")
            for p in model.model.layers[-training_args.unfreeze_last_n_layers:].parameters():
                p.requires_grad = True

        if training_args.unfreeze_new_tokens:
            rank0_print(f"Unfreezing new tokens parameters via embedding hook...")
            model.model.embed_tokens.weight.requires_grad = True
            # Registering hook before Trainer initialization is invalid, so it is disabled
            # model.model.embed_tokens.weight.register_hook(mask_embedding_grad)
        
        if training_args.unfreeze_visual:
            rank0_print(f"Unfreezing visual parameters...")
            for p in model.visual.parameters():
                p.requires_grad = True

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0  # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = {
            "input_ids": input_ids,
            "labels": labels.long() if labels.dtype == torch.int32 else labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.concat([instance["pixel_values"] for instance in instances], dim=0)
            batch["image_grid_thw"] = torch.concat([instance["image_grid_thw"] for instance in instances], dim=0)

        if "coordinates" in instances[0]:
            batch["coordinates"] = [instance["coordinates"] for instance in instances]
            batch["visual_token_indices_of_coordinates"] = [instance["visual_token_indices_of_coordinates"] for instance in instances]

        if "multi_patch_labels" in instances[0]:
            batch["multi_patch_labels"] = [instance["multi_patch_labels"] for instance in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                processor: transformers.ProcessorMixin,
                                data_args: DataArguments,
                                training_args: TrainingArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, processor=processor, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return {"train_dataset": train_dataset, "eval_dataset": None, "data_collator": data_collator}


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if training_args.verbose_logging:
        rank0_print("Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    # set up model
    if model_args.model_type == "qwen2vl":
        model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2" if model_args.flash_attn_2_enabled else None,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
        )
    elif model_args.model_type == "qwen25vl":
        model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2" if model_args.flash_attn_2_enabled else None,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
        )
    else:
        raise ValueError(f"Invalid model type: {model_args.model_type}")
    model.config.use_cache = False
    model.reset_loss_weights(pointer_loss_weight=training_args.pointer_loss_weight, lm_loss_weight=training_args.lm_loss_weight)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    setup_params_to_update(model, training_args)

    # set up tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS},
        tokenizer=tokenizer,
        model=model,
    )
    update_pointer_token_ids(model.config, tokenizer)

    data_args.processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, min_pixels=data_args.min_pixels, max_pixels=data_args.max_pixels
    )
    data_args.processor.tokenizer = tokenizer

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        dump_args_to_json(model.config, data_args.processor, model_args, data_args, training_args, training_args.output_dir)

    data_module = make_supervised_data_module(tokenizer=tokenizer, processor=data_args.processor, data_args=data_args, training_args=training_args)

    trainer = AGUVISTrainer(
        model=model,
        processing_class=data_args.processor,
        args=training_args,
        **data_module,
    )

    # When LiteTrain, only update the gradient of the new tokens
    if training_args.unfreeze_new_tokens:
        emb_param = None
        for n, p in trainer.model.named_parameters():
            if n.endswith("model.embed_tokens.weight"):
                emb_param = p; break
        if emb_param is None:
            raise ValueError("embed_tokens.weight not found")

        n_new_tokens = len(ADDITIONAL_SPECIAL_TOKENS)
        def mask_grad(grad):
            grad[:-n_new_tokens] = 0.0
            return grad
        emb_param.register_hook(mask_grad)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
