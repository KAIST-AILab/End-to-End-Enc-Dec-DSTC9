# End-to-End Document-Grounded Conversation with Encoder-Decoder Pre-Trained Language Model
Our codes are developed based on the [DSTC9 Track 1 repository](https://github.com/alexa/alexa-with-dstc9-track1-dataset). Please visit the [track repository](https://github.com/alexa/alexa-with-dstc9-track1-dataset) or the [track overview paper](https://arxiv.org/abs/2006.03533) for more information about the challenge.

## Dataset
Clone the `data` directory of the [DSTC9 Track 1 dataset repository](https://github.com/alexa/alexa-with-dstc9-track1-dataset).

## Environment
Requires Python version 3.6.
```bash
pip install -r requirements.txt
```

## Snippet Filtering
```bash
./drqa.sh
python snippet_filtering.py
```

## Train
```bash
./train.sh
```

## Evaluate
```bash
./evaluate.sh
```
