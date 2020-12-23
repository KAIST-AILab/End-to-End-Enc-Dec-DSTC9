import argparse

import torch
from transformers import (
    BertForMultipleChoice,
    BertForNextSentencePrediction,
)

from baseline.main import main as _main
from .dataset import KnowledgeSelectionDataset, KnowledgeTurnDetectionDataset
from .utils.evaluator import DetectionEvaluator, SelectionEvaluator
from .utils.model import run_batch_detection, run_batch_selection_eval, run_batch_selection_train


def _get_classes(args):
    if args.task.lower() == "selection":
        return KnowledgeSelectionDataset, BertForMultipleChoice, run_batch_selection_train, run_batch_selection_eval, SelectionEvaluator
    elif args.task.lower() == "detection":
        return KnowledgeTurnDetectionDataset, BertForNextSentencePrediction, run_batch_detection, run_batch_detection, DetectionEvaluator
    else:
        raise ValueError("args.task not in ['selection', 'detection'], got %s" % args.task)


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
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="temperature used for softmax sampling")
    parser.add_argument("--start", type=float, default=0.0, help="Slice start portion of dataset. Defaults to 0.0")
    parser.add_argument("--end", type=float, default=1.0, help="Slice end portion of dataset. Defaults to 1.0")
    args = parser.parse_args()
    return args, parser


def main():
    return _main(_get_classes, _parse_args)


if __name__ == "__main__":
    main()
