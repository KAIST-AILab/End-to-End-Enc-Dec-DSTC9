from torch.distributions.categorical import Categorical

from baseline.utils import evaluator


class DetectionEvaluator(evaluator.DetectionEvaluator):
    def update(self, *batch_output):
        loss, mc_logits, mc_labels = batch_output
        if self.args.temperature == 0.0:
            mc_logits = mc_logits.argmax(dim=-1)
        else:
            sampler = Categorical(logits=mc_logits / self.args.temperature)
            mc_logits = sampler.sample()

        self.preds.append(mc_logits.detach().cpu().numpy())
        self.labels.append(mc_labels.detach().cpu().numpy())
        self.loss += loss.mean().item()
        self.steps += 1


class SelectionEvaluator(evaluator.SelectionEvaluator):
    def update(self, *batch_output):
        loss, mc_logits, mc_labels = batch_output
        self.preds.append(mc_logits.detach().cpu().numpy())
        self.labels.append(mc_labels.detach().cpu().numpy())
        self.loss += loss.mean().item()
        self.steps += 1
