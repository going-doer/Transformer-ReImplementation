import os
from torch.utils.data import Dataset, DataLoader
import torch
import logging
LOGGER = logging.getLogger()

class WmtDataset(Dataset):
    def __init__(self, tokenizer, datapath, src_lang ="en", dst_lang="de", type='train', max_seq_len=512):
        super().__init__()

        # if type=="test-dataset":
        #     datasets = ["commoncrawl", "europarl-v7", "news-commentary-v9"]
        #     dst = []
        #     src = []
        #     for dataset in datasets:
        #         dst_path = os.path.join(datapath, "dev", f"news-test2008.{dst_lang}" )
        #         src_path = os.path.join(datapath, "dev", f"news-test2008.{src_lang}" )
                
        #         with open(dst_path, encoding='utf-8') as f:
        #             tmpdst = f.readlines()
        #         with open(src_path, encoding='utf-8') as f:
        #             tmpsrc = f.readlines()
        #         assert(len(tmpdst)==len(tmpsrc))
        #         dst += tmpdst
        #         src += tmpsrc
        # elif type=="train":
        if type=="train":
            datasets = ["commoncrawl", "europarl-v7"] # "news-commentary-v9" 가 읽어들이는데 있어서 문제가 있어 보임.
            dst = []
            src = []
            for dataset in datasets:
                dst_path = os.path.join(datapath, type, dataset, f"{dataset}.{dst_lang}-{src_lang}.{dst_lang}" )
                src_path = os.path.join(datapath, type, dataset, f"{dataset}.{dst_lang}-{src_lang}.{src_lang}" )
                with open(dst_path, encoding='utf-8') as f:
                    tmpdst = f.readlines()
                with open(src_path, encoding='utf-8') as f:
                    tmpsrc = f.readlines()
                
                assert(len(tmpdst)==len(tmpsrc))
                dst += tmpdst
                src += tmpsrc
            # dst_path = os.path.join(datapath, type, f"train.{dst_lang}" )
            # src_path = os.path.join(datapath, type, f"train.{src_lang}" )
            
            # with open(dst_path, encoding='utf-8') as f:
            #     dst = f.readlines()
            # with open(src_path, encoding='utf-8') as f:
            #     src = f.readlines()

            # print(len(dst), len(src))
            # assert(len(dst)==len(src))

        elif type=="dev" or type=="val":
            type="dev"
            datasets = ["newssyscomb2009", "news-test2008", "newstest2009", "newstest2010", "newstest2011", "newstest2012", "newstest2013"]
            dst = []
            src = []
            for dataset in datasets:
                dst_path = os.path.join(datapath, type, f"{dataset}.{dst_lang}" )
                src_path = os.path.join(datapath, type, f"{dataset}.{src_lang}" )
                with open(dst_path, encoding='utf-8') as f:
                    tmpdst = f.readlines()
                with open(src_path, encoding='utf-8') as f:
                    tmpsrc = f.readlines()
                assert(len(tmpdst)==len(tmpsrc))
                dst += tmpdst
                src += tmpsrc

        elif type=="test":
            src_path = os.path.join(datapath, type, f"newstest2014{dst_lang}{src_lang}-src.{src_lang}" ) #en
            dst_path = os.path.join(datapath, type, f"newstest2014{dst_lang}{src_lang}-src.{dst_lang}" ) #en

            with open(src_path, encoding='utf-8') as f:
                src = f.readlines()
            with open(dst_path, encoding='utf-8') as f:
                dst = f.readlines()

        assert(len(dst)==len(src))
        self.len = len(src)
        self.input = src
        self.output = dst
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        input = self.input[idx]
        output = self.output[idx]
        
        encoded_input = self.tokenizer.transform(input)
        encoded_output = self.tokenizer.transform(output)

        # before padded []
        # print(f"encoded_input.shape: {len(encoded_input)}")
        # print(f"encoded_output.shape: {len(encoded_output)}")
        
        # To flow the gradient loss, tensor type must be float64
        return (torch.tensor(encoded_input, dtype=torch.float64, requires_grad=True),
                torch.tensor(encoded_output, dtype=torch.float64, requires_grad=True))
        

    def collate_fn(self, data):
        """Creates mini-batch tensors from list of tuples (src_seq, trg_seq).
        We should build a custom collate_fn rathern than using default collate_fn,
        because merging sequences (including padding) is not supported in default.
        Sequences are padded to the maximum length of mini-batch sequences (dynamic padding).
        Args:
            data: list of tuple(src_seq, trg_seq).
                - src_seq: torch tensor of shape (?); variable length;
                - trg_seq: torch tensor of shape (?); variable length;
        Returns:
            src_seqs: torch tensor of shape (batch_size, padded_length).
            src_lengths: list of length (batch_size); valid length for each padded source sequences.
            trg_seqs: torch tensor of shape (batch_size, padded length).
            trg_lengths: list of length (batch_size); valid length for each padded target sequences.
        """

        def padded(sequences):
            padded_seqs = torch.zeros(len(sequences), self.max_seq_len, requires_grad=True).long()
            if torch.cuda.is_available():
                padded_seqs = padded_seqs.cuda()

            
            for i, seq in enumerate(sequences):
                padded_seqs[i][:min(self.max_seq_len, len(seq))] = seq[:min(self.max_seq_len, len(seq))]

            return padded_seqs


        data.sort(key=lambda x: len(x[0]), reverse=True) 
        src_seqs, trg_seqs = zip(*data)

        padded_src_seqs = padded(src_seqs)
        padded_trg_seqs = padded(trg_seqs)

        # padded_src_seqs = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True)
        # padded_trg_seqs = torch.nn.utils.rnn.pad_sequence(trg_seqs, batch_first=True)

        if torch.cuda.is_available():
            padded_src_seqs.to('cuda')
            padded_trg_seqs.to('cuda')

        return (padded_src_seqs, padded_trg_seqs)