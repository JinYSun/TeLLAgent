import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
 
import gc, os
import numpy as np
import pandas as pd
 
from scipy.stats import pearsonr
from .util.utils import *
#from .util.attention_flow import *

import torch
import torch.nn as nn

import sklearn as sk
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoConfig, AutoTokenizer, RobertaModel, BertModel
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

class markerDataset(Dataset):
    def __init__(self, list_IDs, labels, df_dti, d_tokenizer, p_tokenizer):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti

        self.d_tokenizer = d_tokenizer
        self.p_tokenizer = p_tokenizer

    

    def convert_data(self, acc_data, don_data):
        

        d_inputs = self.d_tokenizer(acc_data, return_tensors="pt")
        p_inputs = self.d_tokenizer(don_data, return_tensors="pt")

        acc_input_ids = d_inputs['input_ids']
        acc_attention_mask = d_inputs['attention_mask']
        acc_inputs = {'input_ids': acc_input_ids, 'attention_mask': acc_attention_mask}

        don_input_ids = p_inputs['input_ids']
        don_attention_mask = p_inputs['attention_mask']
        don_inputs = {'input_ids': don_input_ids, 'attention_mask': don_attention_mask}

        return acc_inputs, don_inputs

    def tokenize_data(self, acc_data, don_data):
        
        tokenize_acc = ['[CLS]'] + self.d_tokenizer.tokenize(acc_data) + ['[SEP]']
       
        tokenize_don = ['[CLS]'] + self.p_tokenizer.tokenize(don_data) + ['[SEP]']

        return tokenize_acc, tokenize_don

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        acc_data = self.df.iloc[index]['acceptor']
        don_data = self.df.iloc[index]['donor']
    
        d_inputs = self.d_tokenizer(acc_data, padding='max_length', max_length=400, truncation=True, return_tensors="pt")
        p_inputs = self.p_tokenizer(don_data, padding='max_length', max_length=400, truncation=True, return_tensors="pt")

        d_input_ids = d_inputs['input_ids'].squeeze()
        d_attention_mask = d_inputs['attention_mask'].squeeze()
        p_input_ids = p_inputs['input_ids'].squeeze()
        p_attention_mask = p_inputs['attention_mask'].squeeze()

        labels = torch.as_tensor(self.labels[index], dtype=torch.float)

        dataset = [d_input_ids, d_attention_mask, p_input_ids, p_attention_mask, labels]
        return dataset


class markerDataModule(pl.LightningDataModule):
    def __init__(self, task_name, acc_model_name, don_model_name, num_workers, batch_size,  traindata_rate = 1.0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_name = task_name
       
        self.traindata_rate = traindata_rate
        
        self.d_tokenizer = AutoTokenizer.from_pretrained(acc_model_name)
        self.p_tokenizer = AutoTokenizer.from_pretrained(don_model_name)

        self.df_train = None
        self.df_val = None
        self.df_test = None

        self.load_testData = True

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def get_task(self, task_name):
        if task_name.lower() == 'OSC':
            return './dataset/OSC/'

        elif task_name.lower() == 'merge':
            self.load_testData = False
            return './dataset/MergeDataset'

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from
        # a single process in distributed settings.
        dataFolder = './dataset/OSC'
        
        self.df_train = pd.read_csv(dataFolder + '/train.csv')
        self.df_val = pd.read_csv(dataFolder + '/val.csv')
       
        ## -- Data Lenght Rate apply -- ##
        traindata_length = int(len(self.df_train) * self.traindata_rate)
        validdata_length = int(len(self.df_val) * self.traindata_rate)

        self.df_train = self.df_train[:traindata_length]
        self.df_val = self.df_val[:validdata_length]

        if self.load_testData is True:
            self.df_test = pd.read_csv(dataFolder + '/test.csv')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = markerDataset(self.df_train.index.values, self.df_train.Label.values, self.df_train,
                                                  self.d_tokenizer, self.p_tokenizer)
            self.valid_dataset = markerDataset(self.df_val.index.values, self.df_val.Label.values, self.df_val,
                                                  self.d_tokenizer, self.p_tokenizer)

        if self.load_testData is True:
            self.test_dataset = markerDataset(self.df_test.index.values, self.df_test.Label.values, self.df_test,
                                                self.d_tokenizer, self.p_tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class markerModel(pl.LightningModule):
    def __init__(self, acc_model_name, don_model_name, lr, dropout, layer_features, loss_fn = "smooth", layer_limit = True, d_pretrained=True, p_pretrained=True):
        super().__init__()
        self.lr = lr
        self.loss_fn = loss_fn
        self.criterion = torch.nn.MSELoss()
        self.criterion_smooth = torch.nn.SmoothL1Loss()
        # self.sigmoid = nn.Sigmoid()

        #-- Pretrained Model Setting
        acc_config = AutoConfig.from_pretrained("seyonec/SMILES_BPE_PubChem_100k_shard00")
        if d_pretrained is False:
            self.d_model = RobertaModel(acc_config)
            print('acceptor model without pretraining')
        else:
            self.d_model = RobertaModel.from_pretrained(acc_model_name, num_labels=2,
                                                        output_hidden_states=True,
                                                        output_attentions=True)
        
        don_config = AutoConfig.from_pretrained("seyonec/SMILES_BPE_PubChem_100k_shard00")

        if p_pretrained is False:
            self.p_model = RobertaModel(don_config)
            print('donor model without pretraining')
        else:
            self.p_model = RobertaModel.from_pretrained(don_model_name,
                                                        output_hidden_states=True,
                                                        output_attentions=True)
            
        #-- Decoder Layer Setting
        layers = []
        firstfeature = self.d_model.config.hidden_size + self.p_model.config.hidden_size
        for feature_idx in range(0, len(layer_features) - 1):
            layers.append(nn.Linear(firstfeature, layer_features[feature_idx]))
            firstfeature = layer_features[feature_idx]

            if feature_idx is len(layer_features)-2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    
        layers.append(nn.Linear(firstfeature, layer_features[-1]))
        
        self.decoder = nn.Sequential(*layers)

        self.save_hyperparameters()

    def forward(self, acc_inputs, don_inputs):
 
        d_outputs = self.d_model(acc_inputs['input_ids'], acc_inputs['attention_mask'])
        p_outputs = self.p_model(don_inputs['input_ids'], don_inputs['attention_mask'])

        outs = torch.cat((d_outputs.last_hidden_state[:, 0], p_outputs.last_hidden_state[:, 0]), dim=1)
        outs = self.decoder(outs)        

        return outs

    def attention_output(self, acc_inputs, don_inputs):
 
        d_outputs = self.d_model(acc_inputs['input_ids'], acc_inputs['attention_mask'])
        p_outputs = self.p_model(don_inputs['input_ids'], don_inputs['attention_mask'])

        outs = torch.cat((d_outputs.last_hidden_state[:, 0], p_outputs.last_hidden_state[:, 0]), dim=1)
        outs = self.decoder(outs)        

        return d_outputs['attentions'], p_outputs['attentions'], outs

    def training_step(self, batch, batch_idx):
 
        acc_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
       
        don_inputs = {'input_ids': batch[2], 'attention_mask': batch[3]}
         
        labels = batch[4]

        output = self(acc_inputs, don_inputs)
        logits = output.squeeze(dim=1)
        
        if self.loss_fn == 'MSE':
            loss = self.criterion(logits, labels)
        else:
            loss = self.criterion_smooth(logits, labels)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
       # print("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        acc_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        don_inputs = {'input_ids': batch[2], 'attention_mask': batch[3]}
        labels = batch[4]
        
        output = self(acc_inputs, don_inputs)
        logits = output.squeeze(dim=1)
        
      
        if self.loss_fn == 'MSE':
            loss = self.criterion(logits, labels)
        else:
            loss = self.criterion_smooth(logits, labels)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, logger=True)
       # print("valid_loss", loss)
        return {"logits": logits, "labels": labels}

    def validation_step_end(self, outputs):
        return {"logits": outputs['logits'], "labels": outputs['labels']}

    def validation_epoch_end(self, outputs):
        preds = self.convert_outputs_to_preds(outputs)
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0), dtype=torch.int)

        mae, mse, r2,r = self.log_score(preds, labels)
                   
        self.log("mae", mae, on_step=False, on_epoch=True, logger=True)
        self.log("mse", mse, on_step=False, on_epoch=True, logger=True)

        self.log("r2", r2, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        acc_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        don_inputs = {'input_ids': batch[2], 'attention_mask': batch[3]}
        labels = batch[4]

        output = self(acc_inputs, don_inputs)
        logits = output.squeeze(dim=1)

        if self.loss_fn == 'MSE':
            loss = self.criterion(logits, labels)
        else:
            loss = self.criterion_smooth(logits, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"logits": logits, "labels": labels}

    def test_step_end(self, outputs):
        return {"logits": outputs['logits'], "labels": outputs['labels']}

    def test_epoch_end(self, outputs):
        preds = self.convert_outputs_to_preds(outputs)
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0), dtype=torch.int)

        mae, mse, r2,r = self.log_score(preds, labels)

        self.log("mae", mae, on_step=False, on_epoch=True, logger=True)
        self.log("mse", mse, on_step=False, on_epoch=True, logger=True)
        self.log("r2", r2, on_step=False, on_epoch=True, logger=True)
        self.log("r", r, on_step=False, on_epoch=True, logger=True)
    def configure_optimizers(self):
    
        param_optimizer = list(self.named_parameters())
        
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0001
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay_rate": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
        )
        return optimizer

    def convert_outputs_to_preds(self, outputs):
        logits = torch.cat([output['logits'] for output in outputs], dim=0)
        return logits

    def log_score(self, preds, labels):
        y_pred = preds.detach().cpu().numpy()
        y_label = labels.detach().cpu().numpy()
  
        mae = mean_absolute_error(y_label, y_pred)
        mse =  mean_squared_error(y_label, y_pred)
        r2=r2_score(y_label, y_pred)
        r = pearsonr(y_label, y_pred)
        print(f'\nmae : {mae}')        
        print(f'mse : {mse}')
        print(f'r2 : {r2}')
        print(f'r : {r}')

        return mae, mse, r2, r


def main_wandb(config=None):
    try:
        if config is not None:
            wandb.init(config=config, project=project_name)
        else:
            wandb.init(settings=wandb.Settings(console='off'))
    
        config = wandb.config
        pl.seed_everything(seed=config.num_seed)
 
        dm = markerDataModule(config.task_name, config.d_model_name, config.p_model_name,
                                 config.num_workers, config.batch_size, config.prot_maxlength, config.traindata_rate)
        dm.prepare_data()
        dm.setup()
 
        model_type = str(config.pretrained['chem'])+"To"+str(config.pretrained['prot'])
        #model_logger = WandbLogger(project=project_name)
        checkpoint_callback = ModelCheckpoint(f"{config.task_name}_{model_type}_{config.lr}_{config.num_seed}", save_top_k=1, monitor="mae", mode="max")
    
        trainer = pl.Trainer(
                             max_epochs=config.max_epoch,
                             precision=16,
                             #logger=model_logger,
                             callbacks=[checkpoint_callback],
                             accelerator='cpu',log_every_n_steps=40
                             )


        if config.model_mode == "train":
            model = markerModel(config.d_model_name, config.p_model_name,
                               config.lr, config.dropout, config.layer_features, config.loss_fn, config.layer_limit, config.pretrained['chem'], config.pretrained['prot'])
            model.train()
            trainer.fit(model, datamodule=dm)

            model.eval()
            trainer.test(model, datamodule=dm)

        else:
            model = markerModel.load_from_checkpoint(config.load_checkpoint)
            
            model.eval()
            trainer.test(model, datamodule=dm)
            
    except Exception as e:
        print(e)


def main_default(config):
    try:
        config = DictX(config)
        pl.seed_everything(seed=config.num_seed)
        
        dm = markerDataModule(config.task_name, config.d_model_name, config.p_model_name,
                                 config.num_workers, config.batch_size, config.traindata_rate)
        
        dm.prepare_data()
        dm.setup()   
        model_type = str(config.pretrained['chem'])+"To"+str(config.pretrained['prot'])
       # model_logger = TensorBoardLogger("./log", name=f"{config.task_name}_{model_type}_{config.num_seed}")
        checkpoint_callback = ModelCheckpoint(f"{config.task_name}_{model_type}_{config.lr}_{config.num_seed}", save_top_k=1, monitor="mse", mode="max")
    
        trainer = pl.Trainer(
                             max_epochs=config.max_epoch,
                             precision= 32,
                            # logger=model_logger,
                             callbacks=[checkpoint_callback],
                             accelerator='cpu',log_every_n_steps=40
                             )

        
        if config.model_mode == "train":
            model = markerModel(config.d_model_name, config.p_model_name,
                               config.lr, config.dropout, config.layer_features, config.loss_fn, config.layer_limit, config.pretrained['chem'], config.pretrained['prot'])
            
            model.train()
            
            trainer.fit(model, datamodule=dm)

            model.eval()
            trainer.test(model, datamodule=dm)
            
        else:
            model = markerModel.load_from_checkpoint(config.load_checkpoint)
            
            model.eval()
            trainer.test(model, datamodule=dm)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    using_wandb = False
    
    if using_wandb == True:
        #-- hyper param config file Load --##
        config = load_hparams('config/config_hparam.json')
        project_name = config["name"]
   
        main_wandb(config)

        ##-- wandb Sweep Hyper Param Tuning --##
        # config = load_hparams('config/config_sweep_bindingDB.json')
        # project_name = config["name"]
        # sweep_id = wandb.sweep(config, project=project_name)
        # wandb.agent(sweep_id, main_wandb)

    else:
        config = load_hparams('config/config_hparam.json')
        
        main_default(config)  
