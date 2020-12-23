import argparse
from argparse import Namespace
import json
import logging
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from .dataset import ResponseGenerationEvalDataset
from .main import load_model, set_seed, setup_distributed
from .utils.argument import update_additional_params
from .utils.evaluator import GenerationSampleEvaluator
from .utils.model import run_batch_generation_sample

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def _get_classes(args):
    return ResponseGenerationEvalDataset, GPT2LMHeadModel, run_batch_generation_sample, GenerationSampleEvaluator


def _parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument('--generate', action='store_true')
    parser.add_argument("--generation_params_file", type=str, default="",
                        help="JSON configuration file for generation-related configurations.")
    parser.add_argument("--dataroot", type=str, default="",
                        help="Path to dataset, will override the path in config.")
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


def evaluate(args, eval_dataset, model, tokenizer, run_batch_fn, evaluator, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,  # only support batch_size=1 for sampling right now
        collate_fn=eval_dataset.collate_fn
    )

    args.tokenizer = tokenizer
    _evaluator = evaluator(args, eval_dataset)
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            _evaluator.update(*run_batch_fn(args, model, batch, eval_dataset))
    result = _evaluator.done()

    if args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results %s *****" % desc)
            writer.write("***** Eval results %s *****\n" % desc)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main(get_classes=_get_classes, parse_args=_parse_args):
    args, parser = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # load args from params file and update the args Namespace
    args.params_file = os.path.join(args.checkpoint, "params.json")
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        update_additional_params(params, args)
        args.update(params)
        if len(args["generation_params_file"]) > 0:
            with open(args["generation_params_file"]) as fg:
                generation_params = json.load(fg)
            args.update(generation_params)
        args = Namespace(**args)

    args.params = params  # used for saving checkpoints
    dataset_args = Namespace(**args.dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task

    setup_distributed(args)
    set_seed(args)

    args.eval_only = True
    dataset_class, model_class, run_batch_sample, evaluator = get_classes(dataset_args)
    model, tokenizer = load_model(args, model_class, dataset_class)
    logger.info("Generation parameters %s", args)

    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        eval_dataset = dataset_class(dataset_args, tokenizer, start=args.start, end=args.end,
                                     split_type=args.eval_dataset, labels=not args.no_labels,
                                     labels_file=args.labels_file, tfidf=args.tfidf, tfidf_file=args.tfidf_file)
        result = evaluate(args, eval_dataset, model, tokenizer, run_batch_sample, evaluator, desc=args.eval_desc or "val")

    return result


if __name__ == "__main__":
    main()
