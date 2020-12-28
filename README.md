[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# End-to-End Document-Grounded Conversation with Encoder-Decoder Pre-Trained Language Model
Our codes are developed based on the [DSTC9 Track 1 repository](https://github.com/alexa/alexa-with-dstc9-track1-dataset).

Please visit the track repository or the [track overview paper](https://arxiv.org/abs/2006.03533) for more information about the challenge.

## Dataset
Copy the `data` directory from the [DSTC9 Track 1 repository](https://github.com/alexa/alexa-with-dstc9-track1-dataset).

## Environment
Requires Python version 3.6.
```bash
pip install -r requirements.txt
```

## Snippet Filtering
Required before performing the training and evaluation.
```bash
./drqa.sh
python snippet_filtering.py
```

## Training
```bash
./train.sh
```

## Evaluation
```bash
./evaluate.sh
```
