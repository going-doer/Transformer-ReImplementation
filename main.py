import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import BPETokenizer
from datasets import WmtDataset

import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()

    # Run Settings
    parser.add_argument('--datapath',
                        default='./datasets/dev/news-test2008.en')
    parser.add_argument('--savepath',
                        default='./outputs')
    parser.add_argument('--mode',
                        default='train')

    args = parser.parse_args()
    return args

def main(args):
    if args.mode=='train':
        # Load Tokenizer
        tokenizer = BPETokenizer(args.datapath, args.savepath, "en") # TODO: 각 언어별로 tokenizer을 따로 해야 하는 건지
        tokenizer.fit()

        # Load Dataset
        train_dataset = WmtDataset(type="train", tokenizer=tokenizer, datapath="./datasets")

        # DataLoader
        train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=2,  # args.batch
                                    shuffle=True,
                                    collate_fn=train_dataset.collate_fn)
        
        # Test 
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, outputs = batch
            print("inputs: ", inputs.shape) # TODO: 길이 맞추기? 
            print("outputs: ", outputs.shape)
            
            if step > 10:
                break # for test

if __name__=="__main__":
    main(get_args())