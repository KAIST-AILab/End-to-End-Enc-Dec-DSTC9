import argparse
from argparse import Namespace
import json
import logging
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    GPT2DoubleHeadsModel,
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .dataset import (
    KnowledgeSelectionDataset,
    KnowledgeTurnDetectionDataset,
    ResponseGenerationDataset,
)
from .models import GPT2ClsDoubleHeadsModel
from .utils.argument import set_default_dataset_params, set_default_params, update_additional_params, verify_args
from .utils.evaluator import DetectionEvaluator, GenerationEvaluator, SelectionEvaluator
from .utils.model import (
    run_batch_detection,
    run_batch_generation,
    run_batch_selection_eval,
    run_batch_selection_train,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
__DEBUG__ = False


def _get_classes(args):
    if args.task.lower() == "generation":
        return (
            ResponseGenerationDataset, GPT2LMHeadModel, run_batch_generation, run_batch_generation, GenerationEvaluator)
    elif args.task.lower() == "selection":
        return (KnowledgeSelectionDataset, GPT2DoubleHeadsModel, run_batch_selection_train, run_batch_selection_eval,
                SelectionEvaluator)
    elif args.task.lower() == "detection":
        return (KnowledgeTurnDetectionDataset, GPT2ClsDoubleHeadsModel, run_batch_detection, run_batch_detection,
                DetectionEvaluator)
    else:
        raise ValueError("args.task not in ['generation', 'selection', 'detection'], got %s" % args.task)


def _parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str, help="JSON configuration file")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--history_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--knowledge_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for knowledge, will override that value in config.")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset.")
    parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                        help="knowledge file name.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--tfidf", action="store_true",
                        help="Restrict Knowledge Selection candidates to tf-idf search results.")
    parser.add_argument("--tfidf_file", type=str, default=None,
                        help="If set, the tfidf search result will be loaded not from the default path, but from this file instead.")
    parser.add_argument("--no_labels", action="store_true",
                        help="Read a dataset without labels.json. This option is useful when running "
                             "knowledge-seeking turn detection on test dataset where labels.json is not available.")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                             "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--negative_sample_method", type=str, choices=["all", "mix", "oracle"],
                        default="",
                        help="Negative sampling method for knowledge selection, will override the value in config.")
    parser.add_argument("--eval_all_snippets", action='store_true',
                        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--start", type=float, default=0.0, help="Slice start portion of dataset. Defaults to 0.0")
    parser.add_argument("--end", type=float, default=1.0, help="Slice end portion of dataset. Defaults to 1.0")
    args = parser.parse_args()
    return args, parser


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn_train,
          run_batch_fn_eval, evaluator) -> Tuple[int, float]:
    if args.local_rank in [-1, 0]:
        log_dir = os.path.join("runs", args.exp_name) if args.exp_name else None
        tb_writer = SummaryWriter(log_dir)
        args.output_dir = log_dir

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # for reproducibility

    for _ in train_iterator:
        local_steps = 0
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            loss, *_ = run_batch_fn_train(args, model, batch)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss / local_steps)
            if __DEBUG__ and step >= 20:
                break

        results = evaluate(args, eval_dataset, model, run_batch_fn_eval, evaluator, desc=str(global_step))
        if args.local_rank in [-1, 0]:
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("loss", tr_loss / local_steps, global_step)

            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            save_model(args, output_dir, model, tokenizer)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / local_steps


def evaluate(args, eval_dataset, model: PreTrainedModel, run_batch_fn, evaluator, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    # eval_batch_size for selection must be 1 to handle variable number of candidates
    force_singleton_batch = args.task in ["selection", "generation"]
    if force_singleton_batch:
        args.eval_batch_size = 1
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and (not force_singleton_batch or eval_dataset.args.eval_all_snippets):
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    _evaluator = evaluator(args, eval_dataset)
    data_infos = []
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            _evaluator.update(*run_batch_fn(args, model, batch))
        data_infos.append(batch[-1])
        if __DEBUG__:
            break

    result = _evaluator.done(data_infos)

    if args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results %s *****" % desc)
            writer.write("***** Eval results %s *****\n" % desc)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def update_args(args):
    # load args from params file and update the args Namespace
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)

        update_additional_params(params, args)
        args.update(params)
        args = Namespace(**args)

    args.params = params  # used for saving checkpoints
    set_default_params(args)
    return args


def get_dataset_args(args):
    dataset_args = Namespace(**args.dataset_args)
    set_default_dataset_params(dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task
    return dataset_args


def setup_distributed(args):
    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device


def load_model(args, model_class, dataset_class):
    # Load pretrained model and tokenizer
    # Barrier to make sure only the first process in distributed training download model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.eval_only:
        args.output_dir = args.checkpoint
        model = model_class.from_pretrained(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        # set output_past to False for DataParallel to work during evaluation
        config.output_past = False
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens(dataset_class.SPECIAL_TOKENS)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    # End of barrier to make sure only the first process in distributed training download model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()

    return model, tokenizer


def save_model(args, output_dir, model, tokenizer):
    logger.info("Saving model checkpoint to %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
        json.dump(args.params, jsonfile, indent=2, default=str)


def main(get_classes=_get_classes, parse_args=_parse_args):
    args, parser = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    verify_args(args, parser)
    args = update_args(args)
    dataset_args = get_dataset_args(args)

    setup_distributed(args)
    set_seed(args)

    dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval, evaluator = get_classes(dataset_args)
    model, tokenizer = load_model(args, model_class, dataset_class)

    logger.info("Training/evaluation parameters %s", args)
    if not args.eval_only:
        train_dataset = dataset_class(dataset_args, tokenizer, start=args.start, end=args.end, split_type="train",
                                      labels=True, tfidf=args.tfidf, tfidf_file=args.tfidf_file)
        eval_dataset = dataset_class(dataset_args, tokenizer, start=args.start, end=args.end, split_type="val",
                                     labels=True, tfidf=args.tfidf, tfidf_file=args.tfidf_file)

        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer, run_batch_fn_train,
                                     run_batch_fn_eval, evaluator)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices:
        # if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if args.local_rank in [-1, 0]:
            save_model(args, args.output_dir, model, tokenizer)

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            model.to(args.device)

    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        eval_dataset = dataset_class(dataset_args, tokenizer, start=args.start, end=args.end,
                                     split_type=args.eval_dataset, labels=not args.no_labels,
                                     labels_file=args.labels_file, tfidf=args.tfidf, tfidf_file=args.tfidf_file)
        result = evaluate(args, eval_dataset, model, run_batch_fn_eval, evaluator, desc=args.eval_desc or "val")

    return result


if __name__ == "__main__":
    main()
