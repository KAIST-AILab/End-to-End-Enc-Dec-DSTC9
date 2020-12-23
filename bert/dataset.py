from itertools import chain
import logging

import torch

from baseline.dataset import BaseDataset
from baseline.utils.data import pad_ids

logger = logging.getLogger(__name__)


class BertbasedDataset(BaseDataset):
    SPECIAL_TOKENS = {}
    SPECIAL_TOKENS_VALUES = []

    def __init__(self, args, tokenizer, split_type, **dataset_walker_args):
        self.pad, self.cls, self.sep = tokenizer.convert_tokens_to_ids(['[PAD]', '[CLS]', '[SEP]'])

        super().__init__(args, tokenizer, split_type, **dataset_walker_args)


class KnowledgeSelectionDataset(BertbasedDataset):
    def _knowledge_to_string(self, doc, name=""):
        join_str = " "  # " %s " % self.knowledge_sep_token
        return join_str.join([name, doc["title"], doc["body"]])

    def _filter_snippet(self, snippet):
        if self.split_type != "train":
            return True
        if 'train_domain' in self.args:
            return snippet["domain"] in self.args.train_domain
        if 'ks_train_domain' in self.args:
            return snippet["domain"] in self.args.ks_train_domain
        return True

    def build_input_from_segments(self, knowledge, history):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}
        sequence = [[self.cls]] + history + [[self.sep] + knowledge + [self.sep]]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [int(i % 2) for i, s in enumerate(sequence) for _ in s]

        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": [],
        }

        candidate_keys = self._get_candidate_keys(example)
        if self.split_type == "train":
            candidate_keys = self._shrink_label_cands(example["knowledge"], candidate_keys)
        this_inst["candidate_keys"] = candidate_keys

        label_idx = 0
        if example["knowledge"] in candidate_keys:
            label_idx = candidate_keys.index(example["knowledge"])
        this_inst["label_idx"] = label_idx

        for key in candidate_keys:
            cand = self.get_snippet(*key)
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"]
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])

        return this_inst

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])
        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        ).view(batch_size, n_candidates, -1)

        token_type_ids = torch.tensor(
            pad_ids(token_type_ids, self.pad)
        ).view(batch_size, n_candidates, -1)

        label_idx = torch.tensor(label_idx)

        return input_ids, token_type_ids, label_idx, data_info


class KnowledgeTurnDetectionDataset(BertbasedDataset):
    def build_input_from_segments(self, history):
        """ Build a sequence of input from history """
        instance = {}
        sequence = [[self.cls]] + history[:-1] + [[self.sep] + history[-1] + [self.sep]]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [int(i % 2) for i, s in enumerate(sequence) for _ in s]

        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(example["history"])
        instance["label"] = example["knowledge_seeking"]
        instance["dialog_id"] = example["dialog_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch]
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        labels = torch.tensor(labels).long()

        return input_ids, token_type_ids, labels, data_info
