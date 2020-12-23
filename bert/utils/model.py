import logging

import torch

logger = logging.getLogger(__name__)


def run_batch_selection_train(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_labels = batch
    model_outputs = model(
        input_ids=input_ids, token_type_ids=token_type_ids, labels=mc_labels
    )
    mc_loss, mc_logits = model_outputs[:2]
    return mc_loss, mc_logits, mc_labels


def run_batch_selection_eval(args, model, batch):
    candidates_per_forward = args.max_candidates_per_forward_eval * (
        args.n_gpu if isinstance(model, torch.nn.DataParallel) else 1)
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_labels = batch
    all_mc_logits = []
    for index in range(0, input_ids.size(1), candidates_per_forward):
        model_outputs = model(
            input_ids=input_ids[0, index:index + candidates_per_forward].unsqueeze(1),
            token_type_ids=token_type_ids[0, index:index + candidates_per_forward].unsqueeze(1)
        )
        mc_logits = model_outputs[0]
        all_mc_logits.append(mc_logits.detach())
    all_mc_logits = torch.cat(all_mc_logits, dim=0).unsqueeze(0)
    return torch.tensor(0.0), all_mc_logits, mc_labels


def run_batch_detection(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, labels = batch
    model_outputs = model(
        input_ids=input_ids, token_type_ids=token_type_ids, next_sentence_label=labels
    )
    cls_loss, cls_logits = model_outputs[:2]
    return cls_loss, cls_logits, labels
