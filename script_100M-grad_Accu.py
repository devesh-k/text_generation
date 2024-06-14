import matplotlib.pyplot as plt
import seaborn as sns
import torch.distributed as dist
import torch, transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoModel, AutoConfig,AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling



import torch.nn as nn
import torch.nn.functional as F
#from datasets import load_dataset
import pandas as pd, numpy as np
from torch import cuda
import datetime
import warnings,itertools
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

# Ignore all warnings
warnings.filterwarnings('ignore')
#pip install transformers bitsandbytes>=0.39.0 -q
import zipfile
import os
import math
import argparse

parser = argparse.ArgumentParser(description='BabyLM training script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default= 15,
                    help='number of epochs to train')
parser.add_argument('--num-nodes', type=int, default=1,
                    help='Number of available nodes/hosts')
parser.add_argument('--node-id', type=int, default=0,
                    help='Unique ID to identify the current node/host')
parser.add_argument('--num-gpus', type=int, default=4,
                    help='Number of GPUs in each node')
# parser.add_argument('--context_length', type=float, default=None,
#                     help='context length for tokenizer')
# parser.add_argument('--target-accuracy', type=float, default=.85,
#                     help='Target accuracy to stop training')
# parser.add_argument('--patience', type=int, default=2,
#                     help='Number of epochs that meet target before stopping')
# Define global tr loass, val loss , context len as global arguements


args = parser.parse_args()

# WORLD_SIZE = args.num_gpus * args.num_nodes
# os.environ['MASTER_ADDR'] = 'localhost' 
# os.environ['MASTER_PORT'] = '9985' 

#define global_vars:
global_tr_loss = torch.inf
global_val_loss = torch.inf


#broadcast global variables at the beginning of training
def broadcast_global_variable(rank):
    global global_tr_loss
    global global_val_loss
    device = torch.device(f'cuda:{rank}')
    global_tr_loss_tensor = torch.tensor(global_tr_loss).to(device)
    global_tr_loss = global_tr_loss_tensor
       
    global_val_loss_tensor = torch.tensor(global_val_loss).to(device)
    global_val_loss = global_val_loss_tensor
    dist.broadcast(torch.tensor(global_tr_loss), src=0)
    dist.broadcast(torch.tensor(global_val_loss), src=0)
    
    print(f"Process {rank}: global_tr_loss after broadcast = {global_tr_loss.item()}")
    print(f"Process {rank}: global_val_loss after broadcast = {global_val_loss.item()}")

    

    

class dataset_pyt(Dataset):
    def __init__(self, df, tokenizer, max_length  ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
                                
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        #print(f"length of text ->{len(text)}")
        #print(f"text ->{text}")
        #encodings = tokenizer(text, truncation=True, max_length= self.max_length, return_overflowing_tokens=True, padding = 'max_length',return_tensors='pt')
        encodings = self.tokenizer(text, truncation=True, max_length= self.max_length, return_overflowing_tokens=True, padding = False)
        # check the length of the encoded list
        
        #x_dict['input_id'] = input_ids_list
        #x_dict['attention_mask'] = input_ids_list
                
        #print(f"x_dict = {x_dict}")             
#       print(f"inside the loader and input_id = {input_ids} and its shape is {input_ids.shape}")
        #labels = input_id_list
        #input_ids = torch.tensor(input_id_list)
        #attention_mask = torch.tensor(attention_mask_list)
        #labels = torch.tensor(labels)
        #print(f"inside the loader and input_id shape= {input_ids.shape} attention_mask_shape is {attention_mask.shape} and label shape is {labels.shape}")
        #print(f"encoding = {encodings}")
               
        return encodings
        
    def __len__(self):
        #return the length of the dataframe
        return len(self.df)

def custom_collate_fn(batch ):
    x_dict = {}
    #print("CUSTOM COllate")
    #print(f"bacth = {batch}")
    input_ids_list = []
    att_mask_list = [] 
    for elem,item in enumerate(batch):
        #print(f"element {elem}")
        #print(f"item = {item}")
        #check whether there are any nested lists :
        if len(item['input_ids']) > 1:
            #print(f"flattening the list")
            input_id_tensor = torch.tensor(list(itertools.chain(*item['input_ids'])))
            #print(f"shape of the flattended tensor = {input_id_tensor.shape}")
            input_ids_list.append(input_id_tensor[:1024])
            att_mask_tensor = torch.tensor(list(itertools.chain(*item['attention_mask'])))
            att_mask_list.append(att_mask_tensor[:1024])
        else:
            input_id_tensor = torch.tensor(item['input_ids']).squeeze(0)
            input_ids_list.append(input_id_tensor)
            att_mask_tensor = torch.tensor(item['attention_mask']).squeeze(0)
            #print(f"shape of att_mask_tensor tensor = {att_mask_tensor.shape}")
            att_mask_list.append(att_mask_tensor)
    #attention_mask = [item['attention_mask'].squeeze(0) for item in batch]

    # Pad sequences to the same length
    #print(f"len of input_id_list = {len(input_ids_list)}")
    #print(f"len of attmask_list = {len(att_mask_list)}")
    #input_id_tensor = torch.tensor(input_ids_list).squeeze()
    #att_mask_tensor = torch.tensor(att_mask_list).squeeze()
    #print("*********************")
    #print(f"input_ids_list = {input_ids_list}")
    
        
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2",return_tensors = "pt" , truncate = True)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.eos_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(att_mask_list, batch_first=True, padding_value=0)
    #print(f"shape of input_id tensor post padding -{input_ids.shape}")
    #print(f"shape of attention_masks tensor post padding -{attention_mask.shape}")
    x_dict['input_ids'] = input_ids
    x_dict['attention_mask'] = attention_mask
    
    return x_dict

@torch.no_grad
def eval_model(val_loader, model,global_rank, epoch , device):
    global global_val_loss
    model.eval()
    model.to(device)
    e = epoch+1
    val_loss_list = []
    #criterion = torch.nn.BCEWithLogitsLoss()
    if global_rank == 0:
        print(f"inside validation data for epoch {e}")
    for ind,x_dict  in enumerate(val_loader):
        ids = x_dict['input_ids'].to(device=device, non_blocking=True)
        att_mask = x_dict['attention_mask'].to(device=device, non_blocking=True)
        labels = ids
        
        with autocast():
            model_output = model(input_ids = ids ,attention_mask = att_mask, labels = labels)
            act_loss = model_output.loss
        
        val_loss_list.append(act_loss)
        del ids,att_mask,labels           
    mean_val_loss = torch.mean(torch.tensor(val_loss_list)).to(device)
    #dist.barrier()
    #broadcast_global_variable(global_rank) 
        
    if mean_val_loss < global_val_loss:
        if global_rank ==0:
            print(f"Val loss has decreased on {global_rank} process reducing the global validation loss from {global_val_loss:.2f} to {mean_val_loss:.2f}")
        dist.barrier()
        global_val_loss = mean_val_loss
        broadcast_global_variable(global_rank) 
        dist.barrier()
        if global_rank ==0:
            current_datetime = datetime.datetime.now()
            # Extract date and time components
            current_date = str(current_datetime.date())
            current_time = str(current_datetime.time()).split('.')[0]
            file_name = 'model'+'_'+current_date+'_'+current_time+'.pth'
            path = os.path.join("model",file_name)
            print(f"saving the model {file_name} on device {global_rank}")
            torch.save(model.module.state_dict(), path)
    # Save model configuration
#         config_file_name = "config"+ current_date+'_'+current_time
#         print(f"saving the config to {config_file_name}")
#         model.config.save_pretrained(os.path.join("model",config_file_name))
        #model.save_pretrained(path)
    else:
        if global_rank == 0:
            print(f"No improvement in validation loss-->epoch= {e} and global val loss is {global_val_loss:.2f}")

    

#tr_model = train_model(train_loader, val_loader, model =  model , device = device , rank = global_rank    
def train_model(train_loader,val_loader,model, device , global_rank,num_epoch = args.epochs):
    global global_tr_loss
    scaler = GradScaler()
    model.train()
    device = device
    print(f"inside train model. Device = {device}")
    optimizer = torch.optim.AdamW(params =  model.parameters(), lr= 5e-5)
    model.to(device)
    import time
    scheduler = transformers.get_cosine_schedule_with_warmup( optimizer= optimizer, num_warmup_steps =len(train_loader)*num_epoch*.1 ,num_training_steps= len(train_loader)*num_epoch,last_epoch = -1 )
    accumulation_steps = 2  # Number of steps to accumulate gradients
    for i in range (num_epoch):
        epoch_start_time = time.time()
        epoch_train_loss = []
        for ind,x_dict in enumerate(train_loader):
            #id_list = x_dict['input_ids']
            if ind %10000 == 0:
                batch_time = time.time()
                dist.barrier()
                duration = batch_time - epoch_start_time
                if global_rank == 0:
                    print(f"executing epoch:{i+1}, it took {duration/60} mins from beginning of epoch till batch#{ind}")
            
            
            ids = x_dict['input_ids'].to(device=device, non_blocking=True)
            att_mask = x_dict['attention_mask'].to(device=device, non_blocking=True)
            labels = ids
                       
            with autocast():
                model_output = model(input_ids = ids ,attention_mask = att_mask, labels = labels)
                act_loss = model_output.loss
            
            #loss calculation                   
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(act_loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            #act_loss = torch.distributed.all_reduce(act_loss, op=dist.ReduceOp.AVG)
            epoch_train_loss.append(act_loss)
            #print(f"current LR->{scheduler.get_last_lr()}")
            
            del ids,att_mask,labels
            
        #batch processing complete    
        
        mean_loss = torch.mean(torch.tensor(epoch_train_loss)).to(device)
        
        if mean_loss < global_tr_loss:
            dist.barrier()
            if global_rank == 0:
                print(f"training loss has decreased on {global_rank} process reducing the global loss from {global_tr_loss:.2f} to {mean_loss:.2f}")
            global_tr_loss = mean_loss
            broadcast_global_variable(global_rank)        
            dist.barrier()
            #checking validation metrices
            eval_model(val_loader, model,global_rank, epoch = i , device = device)
            
        else:
            if global_rank == 0:
                print(f"No improvement in training loss..the global training loss is -->{global_tr_loss:.2f} ")
                print(f" epoch= {i+1} and mean train loss is {torch.mean(torch.tensor(epoch_train_loss)):.2f}")
        
        
    
    return model

            


def read_text(directory):
    directory = os.path.join('.','data','unzip_text_100M',str(directory))  # Replace with your directory path
    print(f"directory :{directory}")
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    print(f"files:{files}")
    text_content = []
    # Read each file
    total_lines = 0
    for filenum,filename in enumerate(files):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            #first_line = file.read()
            #print(f"filename :{filename}->first few lines {first_line}")
            #continue
            lines_list = [line.strip() for line in open(file_path, 'r')]
            print(f"the file:{filename} added {len(lines_list)} rows to the list")
            total_lines+=len(lines_list)
            text_content.append(lines_list)
    
    flattened_list = list(itertools.chain(*text_content))
    assert (len(flattened_list) == total_lines) , f"Expected {len(flattened_list)} to be equal to {total_lines}" 
    
    return flattened_list


def worker(local_rank, args):
    global global_tr_loss
    global global_val_loss 
    global_rank = args.node_id * args.num_gpus + local_rank 
    dist.init_process_group(backend='nccl',world_size=WORLD_SIZE,rank=global_rank )

    #broadcast_global_variable(global_rank)
    context_length = None
    
    min_text_len = 0
    #print(global_tr_loss)
    model_path = os.path.join("model")
    directory = os.path.join('.','data','unzip_text_100M')  # Replace with your directory path

    # dataframe and pre-process code comes here:
    df_train = pd.DataFrame(read_text("train_100M"), columns=['text'])
    df_val = pd.DataFrame(read_text("dev"), columns=['text'])

    #let's create a new column called 'length' on our dataframe to analyze the text

    df_train['length'] = df_train['text'].apply(lambda x: len(x))
    df_val['length'] = df_val['text'].apply(lambda x: len(x))
    df_train.head()

    print(f"the range of length in the train set is {max(df_train.length)} down to {min(df_train.length)}")

    ## Lets check the the distributions

    sns.histplot(df_train['length'], bins='auto')
    plt.title('Histogram of text lengths')
    plt.show()
    array = np.arange(0, 2000, step=100)

    sns.histplot(df_train['length'], bins= array)
    plt.title('Histogram with length of text from 0 to  Model max capacity')
    plt.show()


    # Let us check the number of rows whose length > 1024(the defualt length that the tokenizer can process)
    exceed_tok_len = sum(df_train['length']> 1024)/len(df_train)*100
    print(f"fewer than {exceed_tok_len}% of rows have text more than the length of model")

    #Lets check for the number of rows where length is 0
    sum(df_train['length'] == min_text_len)
    print(f"training dataframe has {sum(df_train['length'] == min_text_len)} rows with {min_text_len} text length")

    #Remoing the rows with min_len text

    df_train = df_train[df_train['length'].astype(bool)]
    df_val = df_val[df_val['length'].astype(bool)]

    #assert statement here to check if there are still any rows still with 0 text
    assert (sum(df_train['length'] == min_text_len) == 0) , f" there are still rows with {min_text_len} left"


    # resetting the index:
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
       

    # calculate the mean text length so that you can define the context length for the tokenizer
    #calc the average len of the text:
    mean_len = int(df_train.length.mean())
    #print(f"The average length of the text is {mean_len}")
    power = math.ceil(math.log2(mean_len))
    print(power)
    context_length = 2**power
    context_length
    #print(f"The context length is {context_length} ")
    #define the tokeinizer and the model:
    # Test the tokenizer:
    device = torch.device("cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2",return_tensors = "pt" , truncate = True, max_length  = context_length ,return_overflowing_tokens=True , padding = False)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    tokenizer.save_pretrained("./model")

    train_dataset = dataset_pyt(df_train,tokenizer = tokenizer , max_length = context_length)
    val_dataset = dataset_pyt(df_val,tokenizer = tokenizer, max_length = context_length)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=WORLD_SIZE,rank=global_rank )
    test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,num_replicas=WORLD_SIZE,rank=global_rank)


    train_loader = DataLoader(train_dataset,batch_size = args.batch_size , num_workers = 4, pin_memory = True, collate_fn = custom_collate_fn , sampler = train_sampler)
    print(f"Length of train loader is {len(train_loader)}")
    val_loader = DataLoader(val_dataset,batch_size = args.batch_size*3,  collate_fn = custom_collate_fn , sampler = test_sampler,num_workers = 4, pin_memory = True)
    #train_loader,val_loader,model, device , num_epoch = args.epochs
    #broadcast global vars
    broadcast_global_variable(global_rank)
    dist.barrier()    
    tr_model = train_model(train_loader, val_loader, model =  model , device = device , global_rank = global_rank)


if __name__ == '__main__':
    torch.multiprocessing.spawn(worker, nprocs=args.num_gpus, args=(args,)) 
