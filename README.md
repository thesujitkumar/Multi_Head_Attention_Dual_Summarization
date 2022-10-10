# PyTorch implementation of Multi-head Attention Dual Summary model from the following paper:
**Detecting Incongruent News Articles Using Multi-head Attention  Dual Summarization, AACL-IJCNLP 2022**

## Data Preprocessing:
1. Collect FNC, ISOT and NELA Dataset as per source and  procedure reported in paper.

2. Create a folder data --> data_name_Data --> Raw_Data --> copy the train, text and development set csv files.

Run following commands for preprocessing:

**FNC Dataset**

Development set:
```sh
python preprocessing/FNC_Preparsing.py --data 'data/FNC_Mix_Data'  --data_name FNC_Mix  --input_file 'FNC_Bin_Dev_Mix.csv' --data_type Dev
```
Training  set:
```sh
python preprocessing/FNC_Preparsing.py --data 'data/FNC_Mix_Data'  --data_name FNC_Mix  --input_file 'FNC_Bin_Train_Mix.csv' --data_type Train
```
Test   set:
```sh
python preprocessing/FNC_Preparsing.py --data 'data/FNC_Mix_Data'  --data_name FNC_Mix  --input_file 'FNC_Bin_Test_Mix.csv' --data_type Test
```

**NELA Dataset**

Development set:
```sh
python preprocessing/NELA_Preparsing.py --data 'data/NELA_Data'  --data_name NELA  --input_file 'dev.csv' --data_type dev
```
Training  set:
```sh
python preprocessing/NELA_Preparsing.py --data 'data/NELA_Data'  --data_name NELA  --input_file 'train.csv' --data_type Train
```
Test   set:
```sh
python preprocessing/NELA_Preparsing.py --data 'data/NELA_Data'  --data_name NELA  --input_file 'test.csv' --data_type test
```

**ISOT Dataset**

Development set:
```sh
python preprocessing/ISOT_Preparsing.py --data 'data/ISOT_Data'  --data_name ISOT  --input_file 'ISOT_dev_ver-2.csv' --data_type dev
```
Training  set:
```sh
python preprocessing/ISOT_Preparsing.py --data 'data/ISOT_Data'  --data_name ISOT  --input_file 'ISOT_train_ver-2.csv' --data_type train
```
Test   set:
```sh
python preprocessing/ISOT_Preparsing.py --data 'data/ISOT_Data'  --data_name ISOT  --input_file 'ISOT_test_ver-2.csv' --data_type test
```
## Gnerate Embeding

**ISOT Dataset**
```sh
python create_embedding.py --data data/ISOT_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200  --data_name ISOT
```   
**FNC Dataset**
```sh
python create_embedding.py --data data/FNC_Mix_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200  --data_name FNC_Mix
```

**NELA Datatset**

```sh
python create_embedding.py --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200  --data_name NELA
```












Commands to Run experiments:   
## CDS Model: 
**FNC Dataset**:
```sh 
python main.py --run_type final --model_name CDS --data data/FNC_Mix_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15   --file_len 5000 --max_num_word 12  
```

**NELA Dataset :**
```sh
python main.py --run_type final --model_name CDS --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12
```

**ISOT DataSet :**
```sh
python main.py --run_type final --model_name CDS --data data/ISOT_Data/Parsed_Data  --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 5 --max_num_sent 5   --file_len 5000 --max_num_word 12 
```
## MANS Models : 

**FNC DataSet**
```sh
python main.py --run_type final --model_name MANS --data data/FNC_Mix_Data/Parsed_Data  --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15 --domain_feature 0  --file_len 5000 --max_num_word 12 --number_head 8  
```
**ISOT Dataset**
```sh
python main.py --run_type final --model_name MANS --data data/ISOT_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 5 --max_num_sent 5  --file_len 5000 --max_num_word 12 --number_head 8
```
**NELA Dataset**
```sh
python main.py --run_type final --model_name MANS --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 --number_head 1
```

## MAT = (MADS*) : MADS model without convolutions summary elememnt: 

**ISOT Dataset**
```sh
python main.py --run_type final --model_name MAT --data data/ISOT_Data/Parsed_Data --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 5 --max_num_sent 5   --file_len 5000 --max_num_word 12 --number_head 8
```
**FNC Dataset**
```sh
python main.py --run_type final --model_name MAT --data data/FNC_Mix_Data/Parsed_Data   --glove  data/glove/ --emb_name GLOVE  --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 --number_head 8 
```
**NELA Dataset**
```sh
python main.py --run_type final --model_name MAT --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15   --file_len 5000 --max_num_word 12 --number_head 1 
```
## MAS:

**FNC Dataset**
```sh
python main.py --run_type final --model_name MAS --data data/FNC_Mix_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 -number_head 8
```
**ISOT Dataset**
```sh
python main.py --run_type final --model_name MAS --data data/ISOT_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 5 --max_num_sent 5   --file_len 5000 --max_num_word 12 --number_head 8
```
**NELA Dataset**
```sh
python main.py --run_type final --model_name MAS --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 --number_head 1
```
## MADS(BILSTM)

**FNC Dataset**
```sh
python main.py --run_type final --model_name MADS_BILSTM --data data/FNC_Mix_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 --number_head 8 --beta 0.5
```
**NELA DataSet**

```sh
python main.py --run_type final --model_name MADS_BILSTM --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15   --file_len 5000 --max_num_word 12 --number_head 1 --beta 0.5
```
**ISOT Dataset**
```sh
python main.py --run_type final --model_name MADS_BILSTM --data data/ISOT_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 5 --max_num_sent 5   --file_len 5000 --max_num_word 12 --number_head 8 --beta 0.5
```

## MADS (S-BERT)

**FNC Dataset**
```sh
python main_SBERT_Models.py --run_type final --model_name MADS_S_BERT --data data/FNC_Mix_Data/Parsed_Data   --input_dim 384 --mem_dim 384 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 --number_head 2 --beta 0.5
```
**ISOT Dataset**
```sh
python main_SBERT_Models.py --run_type final --model_name MADS_S_BERT --data data/ISOT_Data/Parsed_Data   --input_dim 384 --mem_dim 384 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 18 --max_num_sent 5   --file_len 5000 --max_num_word 12 --number_head 2 --beta 0.5
```

**NELA Dataset**

```sh
python main_SBERT_Models.py --run_type final --model_name MADS_S_BERT --data data/NELA_Data/Parsed_Data  --input_dim 384 --mem_dim 384 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15   --file_len 5000 --max_num_word 12 --number_head 2 --beta 0.5
```

## Testing: Load Model:
```sh
python load_model.py  --model_name "Name of model"  --data "data Path"  --glove  data/glove/ --emb_name  GLOVE --input_dim  --mem_dim  --hidden_dim   --data_name data name --max_num_para  --max_num_sent  --expname --number_head --beta
```
