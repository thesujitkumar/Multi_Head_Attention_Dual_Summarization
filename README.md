# Detecting Incongruent News Articles Using Multi-head Attention Dual Summarization

This is the official implementation of the paper [Detecting Incongruent News Articles Using Multi-head Attention Dual Summarization](https://aclanthology.org/2022.aacl-main.70/) **Accepted and published in preeeding of ```AACL-IJCNLP-2022```**. If you use this code or our results in your research, we'd appreciate you cite our paper as following:


```
@inproceedings{kumar2022detecting,
  title={Detecting incongruent news articles using multi-head attention dual summarization},
  author={Kumar, Sujit and Kumar, Gaurav and Singh, Sanasam Ranbir},
  booktitle={Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={967--977},
  year={2022}
}
```

## Dependencies

* python3
* pytorch
* tqdm
* sklearn
* pandas
* openpyxl


## Data Preprocessing

**To Run the code, kindly follow step-by-step instructions:**
1. Collect the FNC, ISOT, and NELA datasets as per the source and procedure reported in the paper.
2. Create a folder data --> data_name_Data --> Raw_Data --> Copy the train, text, and development set .csv files.

**Run the following commands for preprocessing:**

**FNC Dataset**

Development set:
```python preprocessing/FNC_Preparsing.py --data 'data/FNC_Mix_Data'  --data_name FNC_Mix  --input_file 'FNC_Bin_Dev_Mix.csv' --data_type Dev```

Training  set:
```python preprocessing/FNC_Preparsing.py --data 'data/FNC_Mix_Data'  --data_name FNC_Mix  --input_file 'FNC_Bin_Train_Mix.csv' --data_type Train```

Test set:
```python preprocessing/FNC_Preparsing.py --data 'data/FNC_Mix_Data'  --data_name FNC_Mix  --input_file 'FNC_Bin_Test_Mix.csv' --data_type Test```

**NELA Dataset**

Development set:
```python preprocessing/NELA_Preparsing.py --data 'data/NELA_Data'  --data_name NELA  --input_file 'dev.csv' --data_type dev```

Training  set:
```python preprocessing/NELA_Preparsing.py --data 'data/NELA_Data'  --data_name NELA  --input_file 'train.csv' --data_type Train```

Test set:
```python preprocessing/NELA_Preparsing.py --data 'data/NELA_Data'  --data_name NELA  --input_file 'test.csv' --data_type test```

**ISOT Dataset**

Development set:
```python preprocessing/ISOT_Preparsing.py --data 'data/ISOT_Data'  --data_name ISOT  --input_file 'ISOT_dev_ver-2.csv' --data_type dev```

Training  set:
```python preprocessing/ISOT_Preparsing.py --data 'data/ISOT_Data'  --data_name ISOT  --input_file 'ISOT_train_ver-2.csv' --data_type train```

Test set:
```python preprocessing/ISOT_Preparsing.py --data 'data/ISOT_Data'  --data_name ISOT  --input_file 'ISOT_test_ver-2.csv' --data_type test```

## Generate Embeddings

**ISOT Dataset**
```python create_embedding.py --data data/ISOT_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200  --data_name ISOT```   

**FNC Dataset**
```python create_embedding.py --data data/FNC_Mix_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200  --data_name FNC_Mix```

**NELA Datatset**
```python create_embedding.py --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200  --data_name NELA```












## Training :   

### CDS Model: 
**FNC Dataset**:
```python main.py --run_type final --model_name CDS --data data/FNC_Mix_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15   --file_len 5000 --max_num_word 12```

**NELA Dataset :**
```python main.py --run_type final --model_name CDS --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12```

**ISOT DataSet :**
```python main.py --run_type final --model_name CDS --data data/ISOT_Data/Parsed_Data  --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 5 --max_num_sent 5   --file_len 5000 --max_num_word 12```


### MANS Models : 

**FNC DataSet**
```python main.py --run_type final --model_name MANS --data data/FNC_Mix_Data/Parsed_Data  --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15 --domain_feature 0  --file_len 5000 --max_num_word 12 --number_head 8```

**ISOT Dataset**
```python main.py --run_type final --model_name MANS --data data/ISOT_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 5 --max_num_sent 5  --file_len 5000 --max_num_word 12 --number_head 8```

**NELA Dataset**
```python main.py --run_type final --model_name MANS --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 --number_head 1```

### MAT = (MADS*) : MADS model without convolutions summary elememnt: 


**ISOT Dataset**
```python main.py --run_type final --model_name MAT --data data/ISOT_Data/Parsed_Data --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 5 --max_num_sent 5   --file_len 5000 --max_num_word 12 --number_head 8```

**FNC Dataset**
```python main.py --run_type final --model_name MAT --data data/FNC_Mix_Data/Parsed_Data   --glove  data/glove/ --emb_name GLOVE  --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 --number_head 8```

**NELA Dataset**
```python main.py --run_type final --model_name MAT --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15   --file_len 5000 --max_num_word 12 --number_head 1```
### MAS:

**FNC Dataset**
```python main.py --run_type final --model_name MAS --data data/FNC_Mix_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 -number_head 8```

**ISOT Dataset**
```python main.py --run_type final --model_name MAS --data data/ISOT_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 5 --max_num_sent 5   --file_len 5000 --max_num_word 12 --number_head 8```

**NELA Dataset**
```python main.py --run_type final --model_name MAS --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 --number_head 1```
### MADS(BILSTM)

**FNC Dataset**
```python main.py --run_type final --model_name MADS_BILSTM --data data/FNC_Mix_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 --number_head 8 --beta 0.5```

**NELA DataSet**
```python main.py --run_type final --model_name MADS_BILSTM --data data/NELA_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15   --file_len 5000 --max_num_word 12 --number_head 1 --beta 0.5```

**ISOT Dataset**
```python main.py --run_type final --model_name MADS_BILSTM --data data/ISOT_Data/Parsed_Data   --glove  data/glove/ --emb_name  GLOVE --input_dim 200 --mem_dim 100 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 5 --max_num_sent 5   --file_len 5000 --max_num_word 12 --number_head 8 --beta 0.5```

### MADS (S-BERT)

**FNC Dataset**
```python main_SBERT_Models.py --run_type final --model_name MADS_S_BERT --data data/FNC_Mix_Data/Parsed_Data   --input_dim 384 --mem_dim 384 --hidden_dim 100 --epoch 40 --data_name FNC_Mix --max_num_para 18 --max_num_sent 15  --file_len 5000 --max_num_word 12 --number_head 2 --beta 0.5```

**ISOT Dataset**
```python main_SBERT_Models.py --run_type final --model_name MADS_S_BERT --data data/ISOT_Data/Parsed_Data   --input_dim 384 --mem_dim 384 --hidden_dim 100 --epoch 40 --data_name ISOT --max_num_para 18 --max_num_sent 5   --file_len 5000 --max_num_word 12 --number_head 2 --beta 0.5```

**NELA Dataset**

```python main_SBERT_Models.py --run_type final --model_name MADS_S_BERT --data data/NELA_Data/Parsed_Data  --input_dim 384 --mem_dim 384 --hidden_dim 100 --epoch 40 --data_name NELA --max_num_para 18 --max_num_sent 15   --file_len 5000 --max_num_word 12 --number_head 2 --beta 0.5```



## Evaluation
For testing:  
```python load_model.py  --model_name "Name of model"  --data "data Path"  --glove  data/glove/ --emb_name  GLOVE --input_dim  --mem_dim  --hidden_dim   --data_name data name --max_num_para  --max_num_sent  --expname --number_head --beta```  


## Contact
sujitkumar@iitg.ac.in, kumar.sujit474@gmail.com
