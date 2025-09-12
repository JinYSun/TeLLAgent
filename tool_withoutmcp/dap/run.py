import os
import pandas as pd

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from .util.utils import *
from rdkit import Chem
from tqdm import tqdm
from .train import markerModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0 '

device_count = torch.cuda.device_count()
device_biomarker = torch.device('cuda' if torch.cuda.is_available() else "cpu")

device = torch.device('cpu')
a_model_name = 'DeepChem/ChemBERTa-10M-MLM'
d_model_name = 'DeepChem/ChemBERTa-10M-MTR'

tokenizer = AutoTokenizer.from_pretrained(a_model_name)
d_tokenizer = AutoTokenizer.from_pretrained(d_model_name)

#--biomarker Model
##-- hyper param config file Load --##
config = load_hparams('tool/dap/config/predict.json')
config = DictX(config)
model = markerModel(config.d_model_name, config.p_model_name,
                              config.lr, config.dropout, config.layer_features, config.loss_fn, config.layer_limit, config.pretrained['chem'], config.pretrained['prot'])
 
model = markerModel.load_from_checkpoint(config.load_checkpoint,strict=False)
model.eval()
model.freeze()
    
if device_biomarker.type == 'cuda':
    model = torch.nn.DataParallel(model)
    
def get_marker(drug_inputs, prot_inputs):
    output_preds = model(drug_inputs, prot_inputs)
   
    predict = torch.squeeze( (output_preds)).tolist()

    # output_preds = torch.relu(output_preds)
    # predict = torch.tanh(output_preds)
    # predict = predict.squeeze(dim=1).tolist()

    return predict


def marker_prediction(smiles, aas):
    try:
        aas_input = []
        for ass_data in aas:
            aas_input.append(' '.join(list(ass_data)))
    
        a_inputs = tokenizer(smiles, padding='max_length', max_length=510, truncation=True, return_tensors="pt")
        # d_inputs = tokenizer(smiles, truncation=True, return_tensors="pt")
        a_input_ids = a_inputs['input_ids'].to(device)
        a_attention_mask = a_inputs['attention_mask'].to(device)
        a_inputs = {'input_ids': a_input_ids, 'attention_mask': a_attention_mask}

        d_inputs = d_tokenizer(aas_input, padding='max_length', max_length=510, truncation=True, return_tensors="pt")
        # p_inputs = prot_tokenizer(aas_input, truncation=True, return_tensors="pt")
        d_input_ids = d_inputs['input_ids'].to(device)
        d_attention_mask = d_inputs['attention_mask'].to(device)
        d_inputs = {'input_ids': d_input_ids, 'attention_mask': d_attention_mask}
 
        output_list = get_marker(a_inputs, d_inputs)
        
      
        return output_list

    except Exception as e:
        print(e)
        return {'Error_message': e}


def smiles_aas_test(smile_acc,smile_don):
    
    mola =  Chem.MolFromSmiles(smile_acc) 
    smile_acc = Chem.MolToSmiles(mola,   canonical=True)
    mold =  Chem.MolFromSmiles(smile_don)  
    smile_don = Chem.MolToSmiles(mold, canonical=True)

    batch_size = 1
 
    datas = []
    marker_list = []
    marker_datas = []

    
     
    marker_datas.append([smile_acc,smile_don])
    if len(marker_datas) == batch_size:
            marker_list.append(list(marker_datas))
            marker_datas.clear()

    if len(marker_datas) != 0:
        marker_list.append(list(marker_datas))
        marker_datas.clear()
        
    for marker_datas in tqdm(marker_list, total=len(marker_list)):
        smiles_d , smiles_a  = zip(*marker_datas)
        output_pred = marker_prediction(list(smiles_d), list(smiles_a) )
        if len(datas) == 0:
            datas = output_pred
        else:
            datas = datas + output_pred
    
    # ## -- Export result data to csv -- ##
    # df = pd.DataFrame(datas)
    # df.to_csv('./results/predictData_nontonon_bindingdb_test.csv', index=None)

    # print(df)
    return datas
        
 

if __name__ == '__main__':
    
    a = smiles_aas_test('CC(C)CCCC(C)CCC1=C(/C=C2\C(=O)C3=C(C=C(F)C(F)=C3)C2=C(C#N)C#N)SC2=C1N(CCC(C)CCCC(C)C)C1=C2C2=NSN=C2C2=C1N(CCC(C)CCCC(C)C)C1=C2SC(/C=C2\C(=O)C3=C(C=C(F)C(F)=C3)C2=C(C#N)C#N)=C1CCC(C)CCCC(C)C','CCCCC(CC)CC1=C(F)C=C(C2=C3C=C(C4=CC=C(C5=C6C(=O)C7=C(CC(CC)CCCC)SC(CC(CC)CCCC)=C7C(=O)C6=C(C6=CC=C(C)S6)S5)S4)SC3=C(C3=CC(F)=C(CC(CC)CCCC)S3)C3=C2SC(C)=C3)S1')                         
    
