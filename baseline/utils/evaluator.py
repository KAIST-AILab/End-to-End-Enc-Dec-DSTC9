import math

import numpy as np
import sklearn

from .data import write_detection_preds, write_generation_preds, write_selection_preds
from .metrics import BLEU, CorpusNGramDiversity, METEOR, NGramDiversity, ROUGE, UnigramMetric


class Evaluator:
    def __init__(self, args, eval_dataset):
        self.args = args
        self.eval_dataset = eval_dataset
        self.dataset_walker = eval_dataset.dataset_walker

        self.steps = 0
        self.loss = 0.0
        self.preds = []
        self.labels = []

    def update(self, *batch_output):
        raise NotImplementedError

    def done(self, data_infos):
        raise NotImplementedError


class DetectionEvaluator(Evaluator):
    def update(self, *batch_output):
        loss, _, mc_logits, mc_labels = batch_output
        mc_logits = mc_logits > 0.0
        self.preds.append(mc_logits.detach().cpu().numpy())
        self.labels.append(mc_labels.detach().cpu().numpy())
        self.loss += loss.mean().item()
        self.steps += 1

    def done(self, data_infos):
        self.loss /= self.steps
        labels = np.concatenate(self.labels)
        pred_ids = np.concatenate(self.preds)
        accuracy = np.sum(pred_ids == labels) / len(labels)
        precision = sklearn.metrics.precision_score(labels, pred_ids)
        recall = sklearn.metrics.recall_score(labels, pred_ids)
        result = {"loss": self.loss, "accuracy": accuracy, "precision": precision, "recall": recall}
        if self.args.output_file:
            write_detection_preds(self.dataset_walker, self.args.output_file, data_infos, pred_ids)
        return result


class SelectionEvaluator(Evaluator):
    def update(self, *batch_output):
        loss, _, mc_logits, mc_labels = batch_output
        self.preds.append(mc_logits.detach().cpu().numpy())
        self.labels.append(mc_labels.detach().cpu().numpy())
        self.loss += loss.mean().item()
        self.steps += 1

    def done(self, data_infos):
        self.loss /= self.steps
        all_labels = np.array(self.labels).reshape(-1)
        all_pred_ids = np.array([np.argmax(logits) for logits in self.preds])
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
        result = {"loss": self.loss, "accuracy": accuracy}
        if self.args.output_file:
            sorted_pred_ids = [np.argsort(logits.squeeze())[::-1] for logits in self.preds]
            write_selection_preds(self.dataset_walker, self.args.output_file, data_infos, sorted_pred_ids, topk=5)
        return result


class GenerationEvaluator(SelectionEvaluator):
    def done(self, data_infos):
        self.loss /= self.steps
        perplexity = math.exp(self.loss)
        result = {"perplexity": perplexity, "loss": self.loss}
        return result


class GenerationSampleEvaluator:
    def __init__(self, args, eval_dataset):
        self.args = args
        self.eval_dataset = eval_dataset
        self.dataset_walker = eval_dataset.dataset_walker

        self.metrics = [
            UnigramMetric(metric="recall"),
            UnigramMetric(metric="precision"),
            NGramDiversity(n=1),
            NGramDiversity(n=2),
            NGramDiversity(n=3),
            NGramDiversity(n=4),
            CorpusNGramDiversity(n=1),
            CorpusNGramDiversity(n=2),
            CorpusNGramDiversity(n=3),
            CorpusNGramDiversity(n=4),
            BLEU(),
            METEOR(),
            ROUGE()
        ]
        self.all_output_texts = []
        self.dialog_ids = []
        self.do_evaluate = False

    def update(self, *batch_output):
        sampled_output_ids, ground_truth, dialog_id = batch_output
        tokenizer = self.eval_dataset.tokenizer
        sampled_output_text = tokenizer.decode(sampled_output_ids, skip_special_tokens=True)
        self.all_output_texts.append(sampled_output_text)
        self.dialog_ids.append(dialog_id)

        if ground_truth.strip() != "":
            self.do_evaluate = True
            for metric in self.metrics:
                metric.update((sampled_output_text, ground_truth))

    def done(self):
        if self.args.output_file:
            write_generation_preds(self.dataset_walker, self.args.output_file, self.dialog_ids, self.all_output_texts)
        if not self.do_evaluate:
            return {}
        return {metric.name(): metric.compute() for metric in self.metrics}
