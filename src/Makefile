.PHONY: data_analysis data_cleaning bert roberta

python:= /var/scratch/dknguyen/anaconda3/envs/twitter/bin/python3.10 
jupyter := python -m jupyter

notebooks/data_analysis.html:
	jupyter nbconvert --to html --execute notebooks/data_analysis.ipynb

notebooks/data_cleaning.html:
	jupyter nbconvert --to html --execute notebooks/data_cleaning.ipynb

data_analysis: notebooks/data_analysis.html
data_cleaning: notebooks/data_cleaning.html

DEBUG ?= True
RUN_ID ?= 

outputs/bert_default/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/bert_default.yaml

outputs/bert_default/${RUN_ID}/submission.csv: outputs/bert_default/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/bert_default.yaml

outputs/roberta-base_default/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/roberta-base_default.yaml

outputs/roberta-base_default/${RUN_ID}/submission.csv: outputs/roberta-base_default/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/roberta-base_default.yaml

bert_%: outputs/bert_default/${RUN_ID}/submission.csv
	:

roberta-base_%: outputs/roberta-base_default/${RUN_ID}/submission.csv
	:

bert: bert_default
roberta: roberta-base_default