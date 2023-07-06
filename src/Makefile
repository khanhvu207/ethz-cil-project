.PHONY: data_analysis data_cleaning

jupyter := /var/scratch/dknguyen/anaconda3/envs/twitter/bin/python3.10 -m jupyter

notebooks/data_analysis.html:
	jupyter nbconvert --to html --execute notebooks/data_analysis.ipynb

notebooks/data_cleaning.html:
	jupyter nbconvert --to html --execute notebooks/data_cleaning.ipynb

data_analysis: notebooks/data_analysis.html
data_cleaning: notebooks/data_cleaning.html