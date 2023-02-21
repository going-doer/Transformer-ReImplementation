import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from einops import rearrange

# from tokenizer import BPETokenizer
from tokenizer import WordPieceTokenizer
from datasets import WmtDataset
from model import TransformerModel

import argparse
import logging
import random
import numpy as np
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

def get_args():
    parser = argparse.ArgumentParser()

    # Run Settings
    parser.add_argument('--max_seq_len', 
                        default=50, type=int)
    parser.add_argument('--learning_rate', 
                        default=0.0001, type=float)
    parser.add_argument('--epoch',
                        default=100, type=int)
    parser.add_argument('--batch',
                        default=64, type=int)
    parser.add_argument('--seed',
                        default=0, type=int)
    parser.add_argument('--vocab_size',
                        default=16000, type=int)
    parser.add_argument('--datapath',
                        default='./outputs')
    parser.add_argument('--langpair',
                        default='de-en')
    parser.add_argument('--model_name',
                        default='model')
    parser.add_argument('--mode',
                        default='train')

    # tokenization
    parser.add_argument('--l', 
                        default=0, type=int)
    parser.add_argument('--alpha', 
                        default=0, type=float)

    args = parser.parse_args()

    logger.info(f"device: {device}, n_gpu: {n_gpu}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    return args

def main(args):
    if args.mode=='train':
        # Load Tokenizer
        # tokenizer = BPETokenizer(args.datapath, args.savepath, "en") 
        # tokenizer.fit()
        tokenizer = WordPieceTokenizer(args.datapath).load_model()

        # Load Dataset
        train_dataset = WmtDataset(type="train", 
                                    tokenizer=tokenizer, 
                                    max_seq_len=args.max_seq_len,
                                    datapath="./datasets")
        val_dataset = WmtDataset(type="val", 
                                    tokenizer=tokenizer, 
                                    max_seq_len=args.max_seq_len,
                                    datapath="./datasets")

        # DataLoader
        train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=2,  # args.batch
                                    shuffle=True,
                                    collate_fn=train_dataset.collate_fn)

        val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=2,  # args.batch
                                    shuffle=True,
                                    collate_fn=val_dataset.collate_fn)

        # Model
        model = TransformerModel(d_model=512, 
                                num_heads=8, 
                                num_encoder=6, 
                                num_decoder=6, 
                                input_vocab_size=args.vocab_size, 
                                output_vocab_size=args.vocab_size, 
                                dropout=0.1)

        criterion = nn.NLLLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-09)

        best_loss = float("inf")
        cnt = 0
        train_global_step = 0
        val_global_step = 0
        for epoch in range(args.epoch):
            train_loss = 0
            val_loss = 0
            train_total = 0
            val_total = 0

            # train
            model.train()
            for _, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                inputs, outputs = batch
                targets = outputs # 정답
                bos_tokens = torch.ones(outputs.size()[0], 1).long()*2 # 2 means sos token
                outputs = torch.cat((bos_tokens, outputs), dim=-1)
                outputs = outputs[:, :-1] 

                output_prob = model(inputs, outputs)
                # print("output_prob.size(): ", output_prob.size())
                # print("output_prob2.size(): ", output_prob.view(-1, len(tokenizer)).size())
                # print("target.siz():", targets.size())
                # print("target2.size(): ", targets.view(-1).size())
                # loss = criterion(output_prob.view(-1, len(tokenizer)), targets.view(-1)) 
                loss = criterion(rearrange(output_prob, 'b h d -> (b h) d'), rearrange(targets, 'b h -> (b h)'))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_total += 1
                if (train_global_step + 1) % 10 == 0:
                    print("train!outputs=", targets.tolist()[0])
                    print("train!predict=", torch.argmax(output_prob, dim=-1).tolist()[0])
                train_global_step += 1
                # break
            
            train_loss /= train_total
            print("train!outputs=", outputs.tolist()[0])
            print("train!predict=", torch.argmax(output_prob, dim=-1).tolist()[0])

            # val
            model.eval()
            with torch.no_grad():
                for step, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                    inputs, outputs = batch
                    targets = outputs
                    bos_tokens = torch.ones(outputs.size()[0], 1).long()*2
                    outputs = torch.cat((bos_tokens, outputs), dim=-1)
                    outputs = outputs[:, :-1]
                    output_prob = model(inputs, outputs) 
                    # loss = criterion(output_prob.view(-1, len(tokenizer)), targets.view(-1))
                    loss = criterion(rearrange(output_prob, 'b h d -> (b h) d'), rearrange(targets, 'b h -> (b h)'))

                    val_loss += loss.item() * len(outputs) # 확인을 위해서 
                    val_total += len(outputs)
                    if (val_global_step + 1)%10 == 0:
                        pass # tensorboard 사용
                    val_global_step += 1

                val_loss /= val_total
                print("val!outputs=", tokenizer.decode(outputs.tolist())[0])
                print("val!predict=", tokenizer.decode(torch.argmax(output_prob, dim=-1).tolist())[0])

            # result
            print(f"Epoch {epoch+1}/{args.epoch}, Train_Loss{train_loss:.3f}, Val_Loss: {val_loss:.3f}")
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"outputs/{args.model_name}.pt")
                print("model saved!")
                cnt = 0
            else:
                cnt += 1
    elif args.mode == 'test':
        # Load tokenizer
        tokenizer = WordPieceTokenizer(args.datapath).load_model()

        # Load model
        model = TransformerModel(d_model=512,
                                num_heads=8,
                                num_encoder=6,
                                num_decoder=6,
                                input_vocab_size=len(tokenizer),
                                output_vocab_size=len(tokenizer),
                                dropout=0.1).to(device)
        
        model.load_state_dict(torch.load(f"./outputs/{args.model_name}.pt"))
        model.eval()

        def translate(inputs):
            max_length = 50 
            input_len = len(inputs)
            inputs = torch.tensor([tokenizer.transform(input, max_length=max_length) for input in inputs]).cuda()
            outputs = torch.tensor([[2]]*input_len).cuda() # 2 means sos token
            for i in range(max_length):
                prediction = model(inputs, outputs)
                prediction = torch.argmax(prediction, dim=-1)[:, -1] # get final token
                outputs = torch.cat((outputs, prediction.view(-1, -1)), dim=-1) # TODO: einops로 변경하기.
            outputs = outputs.tolist()
            cleanoutput = []
            for i in outputs:
                try:
                    eos_idx = i.index(3)
                    i = i[:eos_idx]
                except:
                    pass
                cleanoutput.append(i)
            outputs = cleanoutput
            return tokenizer.decode(outputs)
            

        with open(args.eval_input, mode='r', encoding='utf-8') as f:
            inputs = f.readlines()
        outputs = []

        batch_size = 32
        for minibatch in tqdm([inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]):
            outputs += translate(minibatch)
        
        with open(args.eval_output, mode='w', encoding='utf-8') as f:
            f.write('\n'.join(outputs) + '\n')


if __name__ == "__main__":
    main(get_args())

if __name__=="__main__":
    main(get_args())