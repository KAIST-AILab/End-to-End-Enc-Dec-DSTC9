from collections import Counter
import json


def load(path):
    with open(path, 'r') as f:
        return json.load(f)


def dump(obj, path):
    with open(path, 'w') as f:
        return json.dump(obj, f, indent=2)


def vote(*items):
    counts = Counter(items)
    max_count = max(counts.values())
    max_items = [k for k, v in counts.items() if v == max_count]
    return max_items[0], items.index(max_items[0])


def ensemble(items):
    ktd = [item['target'] for item in items]
    ktd, _ = vote(*ktd)
    if not ktd:
        return {'target': False}

    items = [item for item in items if item['target']]
    ks = [item['knowledge'][0] for item in items]
    _, ks_idx = vote(*ks)
    return items[ks_idx]


if __name__ == '__main__':
    # higher priority first
    preds = ['prediction0.json', 'prediction1.json', 'prediction2.json']
    preds = [load(pred) for pred in preds]
    outputs = [ensemble(items) for items in zip(*preds)]
    dump(outputs, 'ensemble.json')
