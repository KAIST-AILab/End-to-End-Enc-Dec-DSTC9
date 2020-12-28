from collections import Counter, defaultdict
import json
import os
import re

import nltk
from tqdm.auto import tqdm

stopwords = nltk.corpus.stopwords.words('english')


def load(path):
    with open(path) as f:
        return json.load(f)


def tokenize(text):
    tokens = re.split(r'\W+', text.lower())
    return [token for token in tokens if token not in stopwords]


def get_common_words(logs):
    corpus = '\n'.join('\n'.join(turn['text'] for turn in log) for log in logs)
    counter = Counter(tokenize(corpus))
    common_words, _ = zip(*counter.most_common())
    return common_words[:250]


def is_sub_token(tokens1, tokens2):
    if len(tokens1) > len(tokens2):
        return False
    for i in range(len(tokens2) - len(tokens1) + 1):
        cand = tokens2[i:i + len(tokens1)]
        if cand == tokens1:
            return True
    return False


def can_be_found(target, name, entity_list):
    return any(is_sub_token(target, entity) for entity in entity_list if entity != name)


def unique_parts(tokens, entity_list, common_words):
    cands = []
    for n in range(1, len(tokens)):
        for i in range(len(tokens) - n + 1):
            cand = tokens[i:i + n]
            if n == 1 and cand[0] in common_words:
                continue
            if can_be_found(cand, tokens, entity_list):
                continue
            cands.append(cand)
        if cands:
            break
    return tuple(cands) or (tokens,)


def create_entity_map(know, common_words):
    entity_tokens = [tokenize(entity['name'] or domain) for domain,
                     d in know.items() for entity in d.values()]
    fingerprints = (unique_parts(tokens, entity_tokens, common_words)
                    for tokens in entity_tokens)

    entity_keys = ({'domain': domain, 'entity_id': entity_id}
                   for domain, d in know.items() for entity_id in d.keys())
    entity_map = {
        variant: [entity_key]
        for entity_key, fingerprint in zip(entity_keys, fingerprints) for variant in fingerprint
    }
    return entity_map


def off_by_one_letter(shorter, longer):
    if len(shorter) + 1 != len(longer):
        return False
    idx = [x == y for x, y in zip(longer, list(shorter) + [None])].index(False)
    assert longer[:idx] == shorter[:idx]
    if longer[idx + 1:] == shorter[idx:]:
        return True
    return False


def fuzzy_match(a, b):
    if a == b:
        return True
    if len(a) != len(b):
        return off_by_one_letter(a, b) or off_by_one_letter(b, a)
    if len(a) == 1:
        return False
    delta = [{x, y} for x, y in zip(a, b) if x != y]
    if len(delta) <= 1 or delta.count(delta[0]) == len(delta):
        return True
    return False


def find_all_entities(history, entity_map):
    history = tokenize(history)
    # {(more, tokens, to, match): [entity_id, ...]}
    remaining = defaultdict(list)
    for token in history:
        new_remaining = defaultdict(list)

        remaining = {**remaining, **entity_map}
        for name_tokens, entity_ids in remaining.items():
            if fuzzy_match(token, name_tokens[0]):
                if name_tokens[1:]:
                    new_remaining[name_tokens[1:]].extend(entity_ids)
                else:
                    yield from entity_ids
        remaining = new_remaining
    # return None


def search_rule_based(log, entity_map):
    history = '\n'.join(turn['text'] for turn in log)
    found_entities = set(find_all_entities(history, entity_map))
    found_entities |= {{'domain': 'taxi', 'entity_id': '*'},
                       {'domain': 'train', 'entity_id': '*'}}
    return list(found_entities)


def intersect(tfidf_result, rule_based_result, upto=None):
    if upto is not None:
        tfidf_result = tfidf_result[:upto]
    return [item for item in tfidf_result if item in rule_based_result]


if __name__ == "__main__":
    dataroot = 'data'
    split = 'val'
    logs = load(os.path.join(dataroot, split, 'logs.json'))
    know = load(os.path.join(dataroot, 'knowledge.json'))
    tfidf = load(os.path.join(dataroot, split, 'tfidf-raw.json'))

    common_words = get_common_words(logs)
    entity_map = create_entity_map(know, common_words)
    rule_based = (search_rule_based(log, entity_map) for log in tqdm(logs))
    intersection = [intersect(t, r) or t for t, r in zip(tfidf, rule_based)]

    with open(os.path.join(dataroot, split, 'tfidf.json'), 'w') as f:
        json.dump(intersection, f, indent=2)
