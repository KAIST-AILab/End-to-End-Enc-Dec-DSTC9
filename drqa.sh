#!/bin/bash
git clone https://github.com/facebookresearch/DrQA.git
cd DrQA
pip install -r requirements.txt
python setup.py develop

dataroot="../data"
splits=("train" "val")

doc_path="${dataroot}/drqa.txt"
echo "Building document at ${doc_path}..."
python ../prepare_drqa.py document \
  --dataroot ${dataroot} \
  --save_path ${doc_path}

db_path="${dataroot}/drqa.db"
echo "Building database at ${db_path}..."
python scripts/retriever/build_db.py ${doc_path} ${db_path}
python scripts/retriever/build_tfidf.py ${db_path} ${dataroot}
model_path="${dataroot}/drqa-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"

for split in $splits
do
  save_path="${dataroot}/${split}/tfidf-raw.json"
  echo "Saving ${split} query results at ${save_path}..."
  python ../prepare_drqa.py query \
    --dataroot ${dataroot} \
    --split ${split} \
    --model_path ${model_path} \
    --save_path ${save_path}
done

cd ..
