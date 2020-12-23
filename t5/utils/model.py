import logging

import torch

logger = logging.getLogger(__name__)


def my_run_batch_train(args, model, batch):
    """Do both selection and generation"""
    input_ids, lm_labels, mc_labels = [x.to(args.device) for x in batch if isinstance(x, torch.Tensor)]

    lm_loss, lm_logits, mc_loss, mc_logits, *_ = model(
        input_ids=input_ids,
        lm_labels=lm_labels.view(-1, lm_labels.size(-1)),
        mc_labels=mc_labels)

    loss = mc_loss * args.mc_coefficient + lm_loss
    return loss, lm_logits, mc_logits, mc_labels


def my_run_batch_eval(args, model, batch):
    candidates_per_forward = args.max_candidates_per_forward_eval * (
        args.n_gpu if isinstance(model, torch.nn.DataParallel) else 1)
    input_ids, lm_labels, mc_labels = [x.to(args.device) for x in batch if isinstance(x, torch.Tensor)]

    # selection
    all_mc_logits = []
    gt_hidden, gt_labels = None, None
    for index in range(0, input_ids.size(1), candidates_per_forward):
        mc_logits, *encoder_outputs = model(
            input_ids=input_ids[0, index:index + candidates_per_forward].unsqueeze(1),
            encoder_only=True
        )
        all_mc_logits.append(mc_logits.detach())
        pos = mc_labels.item() - index
        if 0 <= pos < candidates_per_forward:
            gt_hidden = tuple(item[pos, ...].unsqueeze(0) for item in encoder_outputs)
            gt_labels = lm_labels[0, pos + index].unsqueeze(0)
    all_mc_logits = torch.cat(all_mc_logits, dim=0).unsqueeze(0)

    # generation
    lm_loss, lm_logits, *_ = model(
        lm_labels=gt_labels,
        encoder_outputs=gt_hidden,
    )
    return lm_loss, lm_logits, all_mc_logits, mc_labels


def my_run_batch_e2e_eval(args, model, batch):
    candidates_per_forward = args.max_candidates_per_forward_eval * (
        args.n_gpu if isinstance(model, torch.nn.DataParallel) else 1)
    input_ids, lm_labels, mc_labels = [x.to(args.device) for x in batch if isinstance(x, torch.Tensor)]
    n_candidates = input_ids.size(1)
    target = (mc_labels.sum() == 1)  # 아니어도 괜찮음

    # selection
    all_mc_loss, all_mc_logits = torch.tensor(0.0), []
    gt_hidden, gt_labels = None, None
    for index in range(0, n_candidates, candidates_per_forward):
        _mc_labels = mc_labels[:, index:index + candidates_per_forward]
        mc_loss, mc_logits, *encoder_outputs = model(
            input_ids=input_ids[:, index:index + candidates_per_forward],
            mc_labels=_mc_labels,
            encoder_only=True,
            reduction='sum'
        )
        all_mc_loss += mc_loss.detach()
        all_mc_logits.append(mc_logits.detach())
        if target and _mc_labels.bool().any():
            _, pos = _mc_labels.nonzero(as_tuple=True)
            gt_hidden = tuple(item[pos, ...] for item in encoder_outputs)
            gt_labels = lm_labels[0, pos + index]
        del _mc_labels, mc_logits, encoder_outputs
    all_mc_loss /= n_candidates
    all_mc_logits = torch.cat(all_mc_logits, dim=1)

    # generation
    if gt_labels is not None:
        lm_loss, *_ = model(lm_labels=gt_labels, encoder_outputs=gt_hidden)
    else:
        lm_loss = torch.tensor(0.0)

    return lm_loss, all_mc_loss, all_mc_logits, mc_labels


def my_run_batch_selection_train(args, model, batch):
    input_ids, _, mc_labels = [x.to(args.device) for x in batch if isinstance(x, torch.Tensor)]
    mc_loss, mc_logits, *_ = model(
        input_ids=input_ids,
        mc_labels=mc_labels,
        encoder_only=True
    )
    return mc_loss, torch.tensor([]), mc_logits, mc_labels


def my_run_batch_selection_eval(args, model, batch):
    candidates_per_forward = args.max_candidates_per_forward_eval * (
        args.n_gpu if isinstance(model, torch.nn.DataParallel) else 1)
    input_ids, _, mc_labels = [x.to(args.device) for x in batch if isinstance(x, torch.Tensor)]
    all_mc_logits = []
    for index in range(0, input_ids.size(1), candidates_per_forward):
        mc_logits, *_ = model(
            input_ids=input_ids[0, index:index + candidates_per_forward].unsqueeze(1),
            encoder_only=True
        )
        all_mc_logits.append(mc_logits.detach())
    all_mc_logits = torch.cat(all_mc_logits, dim=0).unsqueeze(0)
    return torch.tensor(0.0), torch.tensor([]), all_mc_logits, mc_labels


def my_run_batch_generation(args, model, batch):
    input_ids, lm_labels = [x.to(args.device) for x in batch if isinstance(x, torch.Tensor)]
    loss, lm_logits, *_ = model(input_ids=input_ids, lm_labels=lm_labels)
    return loss, lm_logits, torch.tensor([]), torch.tensor([])


def my_run_batch_generation_sample(args, model, batch, dataset):
    special_tokens_ids = args.tokenizer.convert_tokens_to_ids(dataset.SPECIAL_TOKENS_VALUES)
    special_tokens_ids.remove(dataset.eos)
    special_tokens_ids = [[i] for i in special_tokens_ids]

    example = batch[0]
    knowledge_key, history = example["knowledge"], example["history"]
    knowledge = dataset.get_snippet(*knowledge_key)
    response_text = example["response_text"]
    dialog_id = example["dialog_id"]

    instance, _ = dataset.build_input_from_segments(knowledge, history, [], with_eos=False)
    input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
    current_output = model.generate(
        input_ids=input_ids,
        max_length=args.max_length,
        min_length=args.min_length,
        do_sample=(not args.no_sample),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        pad_token_id=dataset.pad,
        bos_token_id=dataset.bos,
        eos_token_id=dataset.eos,
        bad_words_ids=special_tokens_ids,
    )[0]

    return current_output, response_text, dialog_id
