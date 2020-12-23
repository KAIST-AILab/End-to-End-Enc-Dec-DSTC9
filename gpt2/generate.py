from baseline.generate import main as _main
from baseline.utils.evaluator import GenerationSampleEvaluator
from model.dataset import MyResponseGenerationEvalDataset
from model.utils.model import my_run_batch_generation_sample
from .models import GPT2End2EndModel


def _get_classes(args):
    args.task = "generation"
    return MyResponseGenerationEvalDataset, GPT2End2EndModel, my_run_batch_generation_sample, GenerationSampleEvaluator


def main():
    return _main(_get_classes)


if __name__ == "__main__":
    main()
