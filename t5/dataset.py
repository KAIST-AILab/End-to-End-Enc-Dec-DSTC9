from itertools import chain
import logging
import random

import torch

from baseline import dataset as base_dataset
from baseline.utils.data import pad_ids, collate_dicts
from gpt2.dataset import first_few_hot



logger = logging.getLogger(__name__)


class MyResponseGenerationDataset(base_dataset.ResponseGenerationDataset):
    def build_input_from_segments(self, knowledge=None, history=None, response=None, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}

        input_ids = [[self.bos]] + history
        input_ids_with_speaker = [
            [self.speaker1 if (len(input_ids) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(input_ids[1:])
        ]
        input_ids = [input_ids[0]] + input_ids_with_speaker + [[self.knowledge_tag] + knowledge]

        instance["input_ids"] = list(chain(*input_ids))
        instance["lm_labels"] = response + ([self.eos] if with_eos else [])

        return instance, input_ids

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        return input_ids, lm_labels


class MyResponseGenerationEvalDataset(MyResponseGenerationDataset):
    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch


class MySelectionGenerationDataset(base_dataset.KnowledgeSelectionDataset):
    """Each instance is composed of:
        (a) history <speaker2> (input a)
        (b) knowledge candidates (input b)
        (c) knowledge label (mc_label)
        (d) gold response (lm_label)
    """

    def build_input_from_segments(self, knowledge=None, history=None, response=None, lm_labels=False, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}

        input_ids = [[self.bos]] + history
        input_ids_with_speaker = [
            [self.speaker1 if (len(input_ids) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(input_ids[1:])
        ]
        input_ids = [input_ids[0]] + input_ids_with_speaker + [[self.knowledge_tag] + knowledge]

        instance["input_ids"] = list(chain(*input_ids))
        instance["lm_labels"] = response + ([self.eos] if with_eos else [])
        if not lm_labels:
            instance["lm_labels"] = [-100] * len(instance["lm_labels"])

        return instance, input_ids

    def __getitem__(self, index):
        example = self.examples[index]
        this_inst = {"dialog_id": example["dialog_id"]}

        candidate_keys = self._get_candidate_keys(example)
        if self.split_type == "train":
            candidate_keys = self._shrink_label_cands(example["knowledge"], candidate_keys)
        this_inst["candidate_keys"] = candidate_keys

        label_idx = 0
        if example["knowledge"] in candidate_keys:
            label_idx = candidate_keys.index(example["knowledge"])
        this_inst["mc_labels"] = label_idx

        instances = []
        for i, key in enumerate(candidate_keys):
            lm_labels = bool(i == label_idx)
            cand = self.get_snippet(*key)
            instance, _ = self.build_input_from_segments(cand, example["history"], example["response"], lm_labels)
            instances.append(instance)
        this_inst.update(collate_dicts(instances))

        return this_inst

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        lm_labels = [ids for ins in batch for ids in ins["lm_labels"]]
        mc_labels = [ins["mc_labels"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])
        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        ).view(batch_size, n_candidates, -1)

        lm_labels = torch.tensor(
            pad_ids(lm_labels, -100)
        ).view(batch_size, n_candidates, -1)

        mc_labels = torch.tensor(mc_labels)

        return input_ids, lm_labels, mc_labels, data_info


class MySelectionGenerationEvalDataset(MySelectionGenerationDataset):
    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch


class MyDetectionSelectionGenerationDataset(MySelectionGenerationDataset):
    """Each instance is composed of:
        (a) history <speaker2> (input a)
        (b) knowledge candidates (input b)
        (c) knowledge label (mc_label)
        (d) gold response (lm_label)
    """

    def __getitem__(self, index):
        example = self.examples[index]
        this_inst = {"dialog_id": example["dialog_id"]}

        knowledges = []
        if example["knowledge"] is not None:
            knowledges.append(example["knowledge"])

        candidate_keys = self._get_candidate_keys(example)
        candidate_keys, labels = self._reduce_candidates(
            knowledges, candidate_keys, self.args.n_candidates)
        this_inst["candidate_keys"] = candidate_keys
        this_inst["mc_labels"] = list(map(float, labels))

        instances = []
        for i, key in enumerate(candidate_keys):
            lm_labels = bool(labels[i])
            cand = self.get_snippet(*key)
            instance, _ = self.build_input_from_segments(cand, example["history"], example["response"], lm_labels)
            instances.append(instance)
        this_inst.update(collate_dicts(instances))

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

