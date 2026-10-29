# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
from pathlib import Path
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import DatasetDict, load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments


def patch_outer_forward(model):
    # Thinker returns mean loss and does not pass num_items_in_batch to its loss function.
    model.accepts_loss_kwargs = False
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. " "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    if isinstance(path, dict):
        from robohead_baseline.track1.local.finetuning.data import read_audio_record

        if sr != 16000:
            raise ValueError("Indexed recordings require 16 kHz audio")
        return read_audio_record(path)
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template([prefix_msgs], add_generation_prompt=True, tokenize=False)[0]
        return {
            "prompt": prompt,
            "audio": (
                {
                    key: ex[key]
                    for key in (
                        "audio",
                        "audio_interval_samples",
                        "audio_samples_sha256",
                        "audio_channel",
                        "audio_parts",
                        "speed",
                    )
                    if ex.get(key) is not None
                }
                if ex.get("audio_interval_samples") is not None
                else ex["audio"]
            ),
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            # Mask prefix tokens by their non-padding positions.
            attention = full_inputs["attention_mask"][i]
            labels[i, (attention.cumsum(0) <= pl) | (attention == 0)] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    val_cer = None

    def _load_best_model(self):
        if self.state.best_model_checkpoint:
            checkpoint = Path(self.args.output_dir) / Path(self.state.best_model_checkpoint).name
            self.state.best_model_checkpoint = str(checkpoint)
        return super()._load_best_model()

    def evaluation_loop(
        self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"
    ):
        if self.val_cer is not None and getattr(self.val_cer, "reference_file", None) and metric_key_prefix == "eval":
            from transformers.trainer_utils import EvalLoopOutput

            return EvalLoopOutput(
                predictions=None, label_ids=None, metrics=self.val_cer(self), num_samples=len(self.val_cer.rows)
            )
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        if self.val_cer is not None and metric_key_prefix == "eval":
            output.metrics.update(self.val_cer(self))
        return output

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


class FinalValEvaluationCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if args.do_eval and state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


def evaluate_pretrained(trainer, model_path, eval_file, resume=False):
    from robohead_baseline.track1.local.finetuning.common import model_identity, sha256, write_json

    path = Path(trainer.args.output_dir) / "baseline.json"
    identity = dict(model=model_identity(model_path), val_sha256=sha256(eval_file))
    if resume and path.exists():
        baseline = json.loads(path.read_text())
        if baseline["identity"] != identity:
            raise ValueError("Baseline model or Val manifest changed; use a new output_dir")
        return baseline["metrics"]
    print("Evaluating original Qwen3-ASR on full Val before loading training checkpoints", flush=True)
    scores = trainer.evaluate()
    if trainer.is_world_process_zero():
        write_json(path, dict(step=0, identity=identity, metrics=scores))
    return scores


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_max_new_tokens", type=int, default=256)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()


def load_training_datasets(train_file, eval_file):
    data = DatasetDict(train=load_dataset("json", data_files=train_file, split="train"))
    if eval_file:
        data["validation"] = load_dataset("json", data_files=eval_file, split="train")
    return data


def main(args_cli=None):
    args_cli = args_cli or parse_args()

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
        local_files_only=True,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    raw_ds = load_training_datasets(args_cli.train_file, args_cli.eval_file)
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=args_cli.sr)

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        per_device_eval_batch_size=getattr(args_cli, "eval_batch_size", args_cli.batch_size),
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        max_steps=getattr(args_cli, "max_steps", -1),
        seed=getattr(args_cli, "seed", 42),
        gradient_checkpointing=getattr(args_cli, "gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if args_cli.eval_file else "no",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        load_best_model_at_end=bool(args_cli.eval_file),
        metric_for_best_model=("eval_tcpcer" if getattr(args_cli, "eval_reference_file", None) else "eval_cer")
        if args_cli.eval_file
        else None,
        greater_is_better=False,
        prediction_loss_only=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        logging_dir=getattr(args_cli, "logging_dir", os.path.join(args_cli.output_dir, "tensorboard")),
        report_to="tensorboard",
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[
            FinalValEvaluationCallback(),
            MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path),
        ],
    )
    if args_cli.eval_file:
        from robohead_baseline.track1.local.finetuning.common import read_jsonl
        from robohead_baseline.track1.local.finetuning.validation import GenerativeValCER

        trainer.val_cer = GenerativeValCER(
            processor,
            read_jsonl(args_cli.eval_file),
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=args_cli.num_workers,
            max_new_tokens=getattr(args_cli, "eval_max_new_tokens", 256),
            reference_file=getattr(args_cli, "eval_reference_file", None),
            tcp_collar=getattr(args_cli, "tcp_collar", 5.0),
        )

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "thinker"):
        model.thinker.config.use_cache = False

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if args_cli.eval_file:
        evaluate_pretrained(trainer, args_cli.model_path, args_cli.eval_file, resume=bool(resume_from))

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    # Save a reloadable final/best model even when the final step is not a save step.
    trainer.save_model(os.path.join(args_cli.output_dir, "final"))
    if trainer.is_world_process_zero():
        copy_required_hf_files_for_qwen_asr(args_cli.model_path, os.path.join(args_cli.output_dir, "final"))
        trainer.save_state()


if __name__ == "__main__":
    main()
