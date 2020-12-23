from baseline.main import main as _main
from .dataset import DetectionSelectionGenerationDataset
from .models import GPT2End2EndModel
from .utils.evaluator import DetectionSelectionGenerationEvaluator
from .utils.model import run_batch_eval, run_batch_train

def _get_classes(args):
    if args.task.lower() == "detection-selection-generation":
        dataset = DetectionSelectionGenerationDataset
        evaluator = DetectionSelectionGenerationEvaluator
        return dataset, GPT2End2EndModel, run_batch_train, run_batch_eval, evaluator
    else:
        raise ValueError(
            "args.task not in ['detection-selection-generation'], got %s" % args.task)


def main():
    return _main(_get_classes)


if __name__ == "__main__":
    main()
