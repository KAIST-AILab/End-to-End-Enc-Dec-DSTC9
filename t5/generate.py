from transformers import T5ForConditionalGeneration

from baseline.generate import main as _main
from baseline.utils.evaluator import GenerationSampleEvaluator
from .dataset import MyResponseGenerationEvalDataset, MySelectionGenerationEvalDataset
from .models import T5DoubleHeadsModel, T5DoubleHeadsMultiLabelModel
from .utils.model import my_run_batch_generation_sample


def _get_classes(args):
    run_batch_sample = my_run_batch_generation_sample
    evaluator = GenerationSampleEvaluator
    if args.task.lower() == 'detection-selection-generation':
        # Eval detection & selection & generation model on generation
        args.task = "generation"
        dataset = MySelectionGenerationEvalDataset
        return dataset, T5DoubleHeadsMultiLabelModel, run_batch_sample, evaluator
    elif args.task.lower() == 'selection-generation':
        # Eval selection & generation model on generation
        return MySelectionGenerationEvalDataset, T5DoubleHeadsModel, run_batch_sample, evaluator
    elif args.task.lower() == "generation":
        # Eval generation-only model
        return MyResponseGenerationEvalDataset, T5ForConditionalGeneration, run_batch_sample, evaluator
    else:
        raise ValueError(
            "args.task not in ['selection-generation', 'generation'], got %s" % args.task)


def main():
    return _main(_get_classes)


if __name__ == "__main__":
    main()
