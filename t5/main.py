from transformers import (
    T5ForConditionalGeneration,
)

from baseline.main import main as _main
from baseline.utils.evaluator import GenerationEvaluator, SelectionEvaluator
from end2end.utils.evaluator import DetectionSelectionGenerationEvaluator, IntentSnippetGenerationEvaluator
from model.utils.evaluator import SelectionGenerationEvaluator
from .dataset import (
    MyDetectionSelectionGenerationDataset,
    MyResponseGenerationDataset,
    MySelectionGenerationDataset,
    T5IntentSnippetDataset,
)
from .models import T5DoubleHeadsModel, T5DoubleHeadsMultiLabelModel
from .utils.model import (
    my_run_batch_e2e_eval,
    my_run_batch_eval,
    my_run_batch_generation,
    my_run_batch_selection_eval,
    my_run_batch_selection_train,
    my_run_batch_train,
)


def _get_classes(args):
    if args.task.lower() == 'detection-selection-generation':
        # Train detection & selection & generation model
        # Evaluate on detection & selection
        dataset = MyDetectionSelectionGenerationDataset
        evaluator = DetectionSelectionGenerationEvaluator
        return dataset, T5DoubleHeadsMultiLabelModel, my_run_batch_train, my_run_batch_e2e_eval, evaluator
    elif args.task.lower() == 'selection-generation':
        # Train selection & generation model
        return (MySelectionGenerationDataset, T5DoubleHeadsModel, my_run_batch_train, my_run_batch_eval,
                SelectionGenerationEvaluator)
    elif args.task.lower() == "selection":
        # Eval selection using ground-truth target
        return (
            MySelectionGenerationDataset, T5DoubleHeadsModel, my_run_batch_selection_train, my_run_batch_selection_eval,
            SelectionEvaluator)
    elif args.task.lower() == "generation":
        # Train generation-only model
        return (
            MyResponseGenerationDataset, T5ForConditionalGeneration, my_run_batch_generation, my_run_batch_generation,
            GenerationEvaluator)
    else:
        raise ValueError(
            "args.task not in ['detection-selection-generation', 'selection-generation', 'selection', 'generation'], got %s" % args.task)


def main():
    return _main(_get_classes)


if __name__ == "__main__":
    main()
