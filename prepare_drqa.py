import json
import os
import argparse

from DrQA.drqa import retriever


def load(p):
    with open(p) as f:
        return json.load(f)


def format_id(domain_name, entity_name=None):
    return entity_name or domain_name


def format_snippet(snippet):
    title = snippet['title']
    body = snippet['body']
    return 'Question: {}\nAnswer: {}'.format(title, body)


def record_iterator(knowledge):
    for domain_name, domain in knowledge.items():
        for entity in domain.values():
            entity_name = format_id(domain_name, entity['name'])
            document = 'Frequently Asked Questions for {}\n\n'.format(
                entity_name)
            document += '\n\n'.join(map(format_snippet,
                                        entity['docs'].values()))
            yield {'id': entity_name, 'text': document}


def format_log(log, upto=None):
    if upto:
        log = log[-upto:]
    return '\n'.join(turn['text'] for turn in log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',
                        choices=['document', 'query'])
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--split',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--model_path')
    parser.add_argument('--n-docs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--save_path', required=True)
    args = parser.parse_args()

    dataroot = args.dataroot
    split = args.split

    if args.stage == 'document':
        knowledge = load(os.path.join(dataroot, 'knowledge.json'))
        records = '\n'.join(map(json.dumps, record_iterator(knowledge)))
        with open(args.save_path, 'w') as f:
            f.write(records)

    elif args.stage == 'query':
        logs = load(os.path.join(dataroot, split, 'logs.json'))
        dialogs = map(format_log, logs)

        ranker = retriever.get_class('tfidf')(tfidf_path=args.model_path)
        closest_docs = ranker.batch_closest_docs(
            dialogs, k=args.n_docs, num_workers=args.num_workers
        )
        docs, _ = zip(*closest_docs)

        with open(args.save_path, 'w') as f:
            json.dump(docs, f, indent=2)
