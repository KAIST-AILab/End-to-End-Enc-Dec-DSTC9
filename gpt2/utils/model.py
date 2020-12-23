import torch


def _recursive_map(fn, obj):
    if isinstance(obj, (tuple, list)):
        return [_recursive_map(fn, item) for item in obj]
    if isinstance(obj, dict):
        return {k: _recursive_map(fn, v) for k, v in obj.items()}
    return fn(obj)


def batch_to_device(batch, device):
    return _recursive_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, batch)


def _lookup_past(item, labels, targets):
    """item: tensor of shape [2, batch_size, n_candidates, ...]
       labels: tensor of shape [batch_size, n_candidates]"""
    for i, (label, target) in enumerate(zip(labels, targets)):
        if target:
            idx, = label.nonzero(as_tuple=True)
            yield item[:, i, idx, ...]


def run_batch_train(args, model, batch):
    """Do both selection and generation"""
    history, candidates, response, mc_labels, data_info = batch_to_device(batch, args.device)
    batch_size = mc_labels.shape[0]

    model.set_task(None)
    past = model(**history)[1]

    model.set_task('selection')
    n_candidates = candidates['input_ids'].size(1)
    expanded_past = tuple(item.repeat_interleave(
        n_candidates, dim=1) for item in past)
    mc_loss, _, past, *_ = model(**candidates,
                                 past=expanded_past, labels=mc_labels)

    model.set_task('generation')
    if 'target' in data_info:
        target = data_info['target']
    else:
        target = [b.bool().any() for b in mc_labels]

    if any(target):
        input_ids = torch.stack([ids for ids, t in zip(response['input_ids'], target) if t], dim=0)
        token_type_ids = torch.stack([ids for ids, t in zip(response['token_type_ids'], target) if t], dim=0)

        rest = past[0].shape[2:]
        past = (item.view(2, batch_size, -1, *rest) for item in past)
        past = (list(_lookup_past(item, mc_labels, target)) for item in past)
        past = [torch.cat(item, dim=1) for item in past]
        lm_loss, *_ = model(input_ids=input_ids,
                            token_type_ids=token_type_ids, labels=input_ids, past=past)
    else:
        lm_loss = torch.tensor(0.0)

    loss = mc_loss * args.mc_coefficient + lm_loss
    return loss,


def run_batch_eval(args, model, batch):
    """For mid-train evaluation"""
    candidates_per_forward = args.max_candidates_per_forward_eval * (
        args.n_gpu if isinstance(model, torch.nn.DataParallel) else 1)
    history, candidates, response, mc_labels, _ = batch_to_device(batch, args.device)
    n_candidates = candidates['input_ids'].size(1)

    model.set_task(None)
    past = model(**history)[1]

    model.set_task('selection')
    expanded_past = [item.expand(-1, candidates_per_forward, -1, -1, -1) for item in past]
    future = None
    all_mc_loss, all_mc_logits = torch.tensor(0.0), []
    for index in range(0, n_candidates, candidates_per_forward):
        if n_candidates - index < candidates_per_forward:
            expanded_past = [
                item.expand(-1, (n_candidates - index), -1, -1, -1) for item in past]

        def chop(x):
            return x[:, index:index + candidates_per_forward]

        _labels = chop(mc_labels)
        mc_loss, mc_logits, present, *_ = model(
            input_ids=chop(candidates['input_ids']),
            token_type_ids=chop(candidates['token_type_ids']),
            labels=_labels,
            mc_token_ids=chop(candidates['mc_token_ids']),
            past=expanded_past,
            reduction='sum'
        )
        all_mc_loss += mc_loss.detach()
        all_mc_logits.append(mc_logits.detach())
        if _labels.bool().any():
            _, idx = _labels.nonzero(as_tuple=True)
            future = tuple(item[:, idx, ...] for item in present)
        del _labels, mc_logits, present, _
    all_mc_loss /= n_candidates
    all_mc_logits = torch.cat(all_mc_logits, dim=1)

    if future is not None:
        model.set_task('generation')
        lm_loss, *_ = model(**response, past=future, labels=response['input_ids'])
    else:
        lm_loss = torch.tensor(0.0)

    return lm_loss, all_mc_loss, all_mc_logits, mc_labels
