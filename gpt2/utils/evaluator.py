import math

import numpy as np
import sklearn

from .utils.data import write_detection_selection_preds


class DetectionSelectionGenerationEvaluator:
    def __init__(self, args, eval_dataset):
        self.args = args
        self.eval_dataset = eval_dataset
        self.dataset_walker = eval_dataset.dataset_walker

        self.steps = 0
        self.mc_loss, self.lm_loss = 0.0, 0.0
        self.target_preds, self.target_labels = [], []
        self.cand_preds, self.cand_labels = [], []

    def update(self, *batch_output):
        lm_loss, mc_loss, mc_logits, mc_labels = batch_output

        # ktdks
        if mc_logits.max() < 0.5:
            self.target_preds.append(False)
            self.cand_preds.append(None)
        else:
            self.target_preds.append(True)
            _, cand_pred = mc_logits[0, :].topk(k=5)
            self.cand_preds.append(cand_pred.detach().cpu().numpy())

        target_label = bool((mc_labels == 1).any())
        self.target_labels.append(target_label)
        self.cand_labels.append(mc_labels[0, :].detach().cpu().numpy())

        self.mc_loss += mc_loss.mean().item()
        self.lm_loss += lm_loss.mean().item()
        self.steps += 1

    def done(self, data_infos):
        self.mc_loss /= self.steps
        self.lm_loss /= self.steps
        result = {
            "loss": self.mc_loss * self.args.mc_coefficient + self.lm_loss,
            "mc_loss": self.mc_loss,
            "lm_loss": self.lm_loss,
            "perplexity": math.exp(self.lm_loss),
        }

        target_preds = np.array(self.target_preds)
        target_labels = np.array(self.target_labels)
        result = {
            **result,
            "ktd_accuracy": np.mean(target_preds == target_labels),
            "precision": sklearn.metrics.precision_score(target_labels, target_preds),
            "recall": sklearn.metrics.recall_score(target_labels, target_preds)
        }

        ks_recall = []
        for target, cand_pred, cand_label in zip(target_preds * target_labels, self.cand_preds, self.cand_labels):
            if target:
                # True Positives Only
                ks_recall.append(cand_label[cand_pred[0]])
        result["ks_accuracy"] = sum(ks_recall) / \
            len(ks_recall) if ks_recall else 0

        if self.args.output_file:
            write_detection_selection_preds(self.dataset_walker, self.args.output_file, data_infos, target_preds,
                                            self.cand_preds, topk=5)
        return result
