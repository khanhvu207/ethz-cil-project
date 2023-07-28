.PHONY: data_analysis data_cleaning bert roberta-base roberta-large timelm

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

outputs/bert_attention/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/bert_attention.yaml

outputs/roberta-base_default/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/roberta-base_default.yaml

outputs/roberta-base_attention/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/roberta-base_attention.yaml

outputs/roberta-large_default/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/roberta-large_default.yaml

outputs/roberta-large_attention/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/roberta-large_attention.yaml

outputs/timelm_default/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/timelm_default.yaml

outputs/timelm_attention/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/timelm_attention.yaml

outputs/albert-base_attention/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/albert-base_attention.yaml

outputs/albert-large_attention/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/albert-large_attention.yaml

outputs/albert-xlarge_attention/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/albert-xlarge_attention.yaml

outputs/xlm-roberta-base_attention/${RUN_ID}/${RUN_ID}.ckpt:
	python train.py --run_id=${RUN_ID} \
		--debug=${DEBUG} \
		--config_path=configs/xlm-roberta-base_attention.yaml

outputs/bert_default/${RUN_ID}/submission.csv: outputs/bert_default/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/bert_default.yaml

outputs/bert_attention/${RUN_ID}/submission.csv: outputs/bert_attention/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/bert_attention.yaml

outputs/roberta-base_default/${RUN_ID}/submission.csv: outputs/roberta-base_default/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/roberta-base_default.yaml

outputs/roberta-base_attention/${RUN_ID}/submission.csv: outputs/roberta-base_attention/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/roberta-base_attention.yaml

outputs/roberta-large_default/${RUN_ID}/submission.csv: outputs/roberta-large_default/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/roberta-large_default.yaml

outputs/roberta-large_attention/${RUN_ID}/submission.csv: outputs/roberta-large_attention/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/roberta-large_attention.yaml

outputs/timelm_default/${RUN_ID}/submission.csv: outputs/timelm_default/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/timelm_default.yaml

outputs/timelm_attention/${RUN_ID}/submission.csv: outputs/timelm_attention/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/timelm_attention.yaml

outputs/albert-base_attention/${RUN_ID}/submission.csv: outputs/albert-base_attention/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/albert-base_attention.yaml

outputs/albert-large_attention/${RUN_ID}/submission.csv: outputs/albert-large_attention/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/albert-large_attention.yaml

outputs/albert-xlarge_attention/${RUN_ID}/submission.csv: outputs/albert-xlarge_attention/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/albert-xlarge_attention.yaml

outputs/xlm-roberta-base_attention/${RUN_ID}/submission.csv: outputs/xlm-roberta-base_attention/${RUN_ID}/${RUN_ID}.ckpt
	python predict.py --run_id=${RUN_ID} \
		--ckpt_path=$< \
		--config_path=configs/xlm-roberta-base_attention.yaml

bert_%: outputs/bert_%/${RUN_ID}/submission.csv
	:

roberta-base_%: outputs/roberta-base_%/${RUN_ID}/submission.csv
	:

roberta-large_%: outputs/roberta-large_%/${RUN_ID}/submission.csv
	:

timelm_%: outputs/timelm_%/${RUN_ID}/submission.csv
	:

albert-base_%: outputs/albert-base_%/${RUN_ID}/submission.csv
	:

albert-large_%: outputs/albert-large_%/${RUN_ID}/submission.csv
	:

albert-xlarge_%: outputs/albert-xlarge_%/${RUN_ID}/submission.csv
	:

xlm-roberta-base_%: outputs/xlm-roberta-base_%/${RUN_ID}/submission.csv
	: