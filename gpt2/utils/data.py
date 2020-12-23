import json
import logging
import os

logger = logging.getLogger(__name__)


def write_detection_selection_preds(dataset_walker, output_file, data_infos, target_preds, sorted_pred_ids, topk=5):
    # Flatten the data_infos
    data_infos = [
        {
            "dialog_id": info["dialog_ids"][i],
            "candidate_keys": info["candidate_keys"][i]
        }
        for info in data_infos
        for i in range(len(info["dialog_ids"]))
    ]

    new_labels = [{"target": False}] * len(dataset_walker)
    # Update the dialogs with selected knowledge
    for info, target_pred, sorted_pred_id in zip(data_infos, target_preds, sorted_pred_ids):
        dialog_id = info["dialog_id"]
        candidate_keys = info["candidate_keys"]

        new_label = {"target": bool(target_pred)}
        if target_pred:
            snippets = []
            for pred_id in sorted_pred_id[:topk]:
                domain, entity_id, doc_id = candidate_keys[pred_id]
                snippet = {
                    "domain": domain,
                    "entity_id": "*" if entity_id == "*" else int(entity_id),
                    "doc_id": int(doc_id)
                }
                snippets.append(snippet)
            new_label["knowledge"] = snippets

        new_labels[dialog_id] = new_label

    if os.path.dirname(output_file) and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w") as jsonfile:
        logger.info("Writing predictions to {}".format(output_file))
        json.dump(new_labels, jsonfile, indent=2)
