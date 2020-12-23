from itertools import chain
import logging
import random

import torch

from baseline.utils.data import collate_dicts, pad_nested_ids, pad_ids
from baseline.dataset import KnowledgeSelectionDataset, ResponseGenerationDataset

logger = logging.getLogger(__name__)


def first_few_hot(ones, total):
    if ones >= total:
        ones = total
    return [1] * ones + [0] * (total - ones)


class MyResponseGenerationDataset(ResponseGenerationDataset):
    def build_input_from_segments(self, knowledge=None, history=None, response=None, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}

        sequence = [[self.bos]] + history
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) %
             2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker + [[self.knowledge_tag] + knowledge] + [
            [self.speaker2] + response + ([self.eos] if with_eos else [])]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence[:-2]) for
                                      _ in s] + [self.knowledge_tag for _ in sequence[-2]] + [self.speaker2 for _ in
                                                                                              sequence[-1]]
        instance["lm_labels"] = ([-100] * sum(len(s)
                                              for s in sequence[:-1])) + [-100] + sequence[-1][1:]

        return instance, sequence


class MyResponseGenerationEvalDataset(MyResponseGenerationDataset):
    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch


class DetectionSelectionGenerationDataset(KnowledgeSelectionDataset):
    """Each instance is composed of:
        (a) history <speaker2> (input a)
        (b) knowledge candidates (input b)
        (c) knowledge label (mc_label)
        (d) gold response (lm_label)
    """

    def build_input_from_history(self, history):
        """ Build a sequence of input from history """
        instance = {}

        sequence = [[self.bos]] + history
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) %
             2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + \
            sequence_with_speaker + [[self.knowledge_tag]]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence[:-1]) for
                                      _ in s] + [self.knowledge_tag]
        return instance, sequence

    def build_input_from_knowledge(self, knowledge):
        """ Build a sequence of input from knowledge """
        instance = {}
        sequence = knowledge + [self.speaker2]
        instance["input_ids"] = sequence
        instance["token_type_ids"] = [self.knowledge_tag] * len(sequence)
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        return instance, sequence

    def build_input_from_response(self, response, with_eos=True):
        """ Build a sequence of input from last reply """
        instance = {}
        sequence = response + ([self.eos] if with_eos else [])
        instance["input_ids"] = sequence
        instance["token_type_ids"] = [self.speaker2] * len(sequence)
        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]
        this_inst = {"dialog_id": example["dialog_id"]}

        instance, _ = self.build_input_from_history(example["history"])
        this_inst["history"] = instance

        knowledges = []
        if example["knowledge"] is not None:
            knowledges.append(example["knowledge"])

        candidate_keys = self._get_candidate_keys(example)
        candidate_keys, labels = self._reduce_candidates(
            knowledges, candidate_keys, self.args.n_candidates)
        this_inst["candidate_keys"] = candidate_keys
        this_inst["label_idx"] = labels

        candidates = (self.get_snippet(*key) for key in candidate_keys)
        this_inst["candidates"] = collate_dicts(
            self.build_input_from_knowledge(cand)[0] for cand in candidates)

        instance, _ = self.build_input_from_response(example["response"])
        this_inst["response"] = instance

        return this_inst

    def _reduce_candidates(self, labels, candidates, target_size):
        """Reduces the number of candidates to target_size.
        Args:
            labels: ground truth subset of candidates
            candidates: possible candidates for ground truth
            target_size: number of final candidates
        """
        if self.split_type != "train":
            return candidates, [int(cand in labels) for cand in candidates]

        if len(labels) > target_size:
            return random.sample(labels, target_size), [1] * target_size
        negative_samples = [cand for cand in candidates if cand not in labels]
        negative_samples = random.sample(
            negative_samples, target_size - len(labels))
        label_idx = first_few_hot(len(labels), target_size)
        return labels + negative_samples, label_idx

    def collate_fn(self, batch):
        history = collate_dicts(ins["history"] for ins in batch)
        candidates = collate_dicts(ins["candidates"] for ins in batch)
        response = collate_dicts(ins["response"] for ins in batch)
        label_idx = [ins["label_idx"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        history = {k: torch.tensor(pad_ids(v, self.pad))
                   for k, v in history.items()}
        candidates = {k: torch.tensor(v) if k == 'mc_token_ids' else torch.tensor(
            pad_nested_ids(v, self.pad)) for k, v in candidates.items()}
        response = {k: torch.tensor(pad_ids(v, self.pad))
                    for k, v in response.items()}
        label_idx = torch.tensor(label_idx)

        return history, candidates, response, label_idx, data_info
