# Post-OCR processing with BERT and NMT models
To use this code, just run "python3 run.py" in src/pipeline folder to run the inference of the tagger and translator models. Tagger and translator models can be switched between those available at: https://drive.google.com/drive/folders/16RwyO8deQD9UDbHj_ccElEhHq5gnqKQQ?usp=sharing. After downloading the folder, just paste its content into src/models.
Evaluation folder contains the code to run and save evaluation of the models. Evaluations are produced after inference.
io/process_data.py can download and pre-process data and create datasets from mC4 or other files provided as .csv files.
