from collections import defaultdict
from itertools import chain
import logging
import random

import torch
from tqdm import tqdm

from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader
from .utils.data import pad_ids, truncate_sequences

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, **dataset_walker_args):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.dataset_walker_args = dataset_walker_args
        self.dataset_walker = DatasetWalker(split_type, dataroot=self.dataroot, **dataset_walker_args)
        self.dialogs = self._prepare_conversations()

        self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)
        self.knowledge, self.snippets = self._prepare_knowledge()

        self.examples = self._create_examples()

    def _prepare_conversations(self):
        logger.info("Tokenize and encode the dialog data")
        dialogs = tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])
        return list(dialogs)

    def _prepare_knowledge(self):
        knowledge = self.knowledge_reader.knowledge
        self.knowledge_docs = self.knowledge_reader.get_doc_list()

        tokenized_snippets = defaultdict(lambda: defaultdict(dict))
        for snippet in filter(self._filter_snippet, self.knowledge_docs):
            name = snippet["domain"] + ' ' + (snippet["entity_name"] or '')
            _knowledge = self._knowledge_to_string(snippet["doc"], name=name)
            tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(_knowledge))

            domain, entity_id, doc_id = snippet["domain"], snippet["entity_id"], int(snippet["doc_id"])
            tokenized_snippets[domain][entity_id][doc_id] = tokenized_knowledge[:self.args.knowledge_max_tokens]
        return knowledge, tokenized_snippets

    def _create_examples(self):
        logger.info("Creating examples")
        _dialogs = tqdm(self.dialogs, disable=self.args.local_rank not in [-1, 0])
        return list(chain.from_iterable(map(self._create_example, _dialogs)))

    def _filter_snippet(self, snippet):
        if self.split_type != "train":
            return True
        if 'ks_train_domain' not in self.args:
            return True
        return snippet["domain"] in self.args.ks_train_domain

    def _knowledge_to_string(self, doc, name=""):
        return doc["body"]

    def _convert_knowledge_to_coord(self, knowledge):
        return knowledge.get('domain'), knowledge.get('entity_id'), knowledge.get('doc_id')

    def _glob_snippets(self, domain=None, entity_id=None, doc_id=None):
        coords = [(domain, entity_id, doc_id)]
        if domain is None:
            coords = ((d, _, _) for _, _, _ in coords for d in self.snippets)
        if entity_id is None:
            coords = ((d, e, _) for d, _, _ in coords for e in self.snippets[d])
        if doc_id is None:
            coords = ((d, e, i) for d, e, _ in coords for i in self.snippets[d][e])
        return coords

    def get_snippet(self, domain, entity_id, doc_id):
        return self.snippets[domain][entity_id][doc_id]

    def _get_knowledge(self, label):
        if "knowledge" in label:
            return label["knowledge"][0]
        return None

    def _get_tfidf_candidates(self, tfidf, used_knowledge=None):
        if tfidf:
            tfidf = tfidf[:self.args.tfidf_upto]
            tfidf_prefixes = map(self._convert_knowledge_to_coord, filter(self._filter_snippet, tfidf))
            tfidf_candidates = [c for prefix in tfidf_prefixes for c in self._glob_snippets(*prefix)]
            if used_knowledge and used_knowledge not in tfidf_candidates:
                tfidf_candidates.append(used_knowledge)
        else:
            tfidf_candidates = []
        return tfidf_candidates

    def _truncate_history(self, dialog):
        history = [
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"]))
            for turn in dialog
        ]
        truncated_history = history[-self.args.history_max_utterances:]
        truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)
        return truncated_history

    def _create_example(self, dialog):
        dialog_id = dialog["id"]
        label = dialog["label"]
        tfidf = dialog.get("tfidf")
        dialog = dialog["log"]
        if label is None:
            # This will only happen when running knowledge-seeking turn detection on test data
            # So we create dummy target here
            label = {"target": False}

        target = label["target"]
        if not target and "detection" not in self.args.task:
            # we only care about non-knowledge-seeking turns in turn detection task
            return

        knowledge = self._get_knowledge(label)
        if target and knowledge is not None:
            if not self._filter_snippet(knowledge):
                return

            knowledge_candidates = list(self._glob_snippets(knowledge['domain'], knowledge['entity_id']))
            if self.split_type == "train" and self.args.negative_sample_method == "oracle":
                if len(knowledge_candidates) < self.get_num_ks_candidates():
                    return

            knowledge_key = self._convert_knowledge_to_coord(knowledge)
            used_knowledge = knowledge_key
        else:
            knowledge_candidates = []
            used_knowledge = None

        tfidf_candidates = self._get_tfidf_candidates(tfidf, used_knowledge)

        gt_resp = label.get("response", "")
        tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))
        truncated_history = self._truncate_history(dialog)

        yield {
            "history": truncated_history,
            "knowledge": used_knowledge,
            "oracle_candidates": knowledge_candidates,
            "tfidf_candidates": tfidf_candidates,
            "response": tokenized_gt_resp,
            "response_text": gt_resp,
            "label": label,
            "knowledge_seeking": target,
            "dialog_id": dialog_id
        }

    def get_num_ks_candidates(self):
        return self.args.n_candidates

    def _get_candidate_keys(self, example):
        n_candidates = self.get_num_ks_candidates()
        tfidf = example["tfidf_candidates"]
        all_candidates = tfidf if len(tfidf) >= n_candidates else self._glob_snippets()
        oracle_candidates = example["oracle_candidates"]
        if not oracle_candidates:
            all_candidates = list(all_candidates)
            oracle_candidates = random.sample(all_candidates, k=n_candidates)

        if self.split_type != "train":
            if self.args.eval_all_snippets:
                return list(all_candidates)
            return oracle_candidates

        if self.args.negative_sample_method == "all":
            return list(all_candidates)
        elif self.args.negative_sample_method == "mix":
            return oracle_candidates + random.sample(list(all_candidates), k=len(oracle_candidates))
        elif self.args.negative_sample_method == "oracle":
            return oracle_candidates
        # although we have already checked for this, still adding this here to be sure
        raise ValueError(
            "negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

    def _shrink_label_cands(self, label, candidates):
        n_candidates = self.get_num_ks_candidates()
        shrunk_label_cands = candidates.copy()
        shrunk_label_cands.remove(label)
        shrunk_label_cands = random.sample(shrunk_label_cands, k=n_candidates - 1)
        shrunk_label_cands.append(label)
        random.shuffle(shrunk_label_cands)

        return shrunk_label_cands

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)


class BaselineDataset(BaseDataset):
    SPECIAL_TOKENS = SPECIAL_TOKENS
    SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES

    def __init__(self, args, tokenizer, split_type, **dataset_walker_args):
        self.bos = tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        super().__init__(args, tokenizer, split_type, **dataset_walker_args)

    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}

        sequence = [[self.bos] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in
                                      s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

        return instance, sequence


class ResponseGenerationEvalDataset(BaselineDataset):
    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch


class ResponseGenerationDataset(BaselineDataset):
    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(
            self.get_snippet(*example["knowledge"]),
            example["history"],
            example["response"]
        )
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        return input_ids, token_type_ids, lm_labels


class KnowledgeSelectionDataset(BaselineDataset):
    def _knowledge_to_string(self, doc, name=""):
        join_str = " %s " % self.knowledge_sep_token
        return join_str.join([name, doc["title"], doc["body"]])

    def build_input_from_segments(self, knowledge, history):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}

        sequence = [[self.bos]] + history
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker + [[self.knowledge_tag] + knowledge + [self.eos]]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence[:-1]) for
                                      _ in s] + [self.knowledge_tag for _ in sequence[-1]]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": [],
            "mc_token_ids": []
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
            candidate = self.get_snippet(*key)
            instance, _ = self.build_input_from_segments(candidate, example["history"])
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])
            this_inst["mc_token_ids"].append(instance["mc_token_ids"])

        return this_inst

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        mc_token_ids = [id for ins in batch for id in ins["mc_token_ids"]]
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

        lm_labels = torch.full_like(input_ids, -100)
        mc_token_ids = torch.tensor(mc_token_ids).view(batch_size, n_candidates)
        label_idx = torch.tensor(label_idx)

        return input_ids, token_type_ids, mc_token_ids, lm_labels, label_idx, data_info


class KnowledgeTurnDetectionDataset(BaselineDataset):
    def build_input_from_segments(self, history):
        """ Build a sequence of input from history """
        instance = {}

        sequence = [[self.bos]] + history[:-1] + [[self.knowledge_tag] + history[-1] + [self.eos]]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in
                                      s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

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
        mc_token_ids = [ins["mc_token_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch]
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        mc_token_ids = torch.tensor(mc_token_ids)
        lm_labels = torch.full_like(input_ids, -100)
        labels = torch.tensor(labels).float()

        return input_ids, token_type_ids, mc_token_ids, lm_labels, labels, data_info
