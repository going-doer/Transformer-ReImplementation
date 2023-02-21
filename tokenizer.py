import argparse
import logging
import sentencepiece as spm
LOGGER = logging.getLogger()

def get_args():
    parser = argparse.ArgumentParser()

    # Run Settings
    parser.add_argument('--vocab_size', 
                        default=16000, type=int)
    parser.add_argument('--datapath', 
                        default='./outputs')
    parser.add_argument('--src_path', 
                        default='./datasets/train/train.en')
    parser.add_argument('--dest_path', 
                        default='./datasets/train/train.de')
    parser.add_argument('--langpair',
                        default='de-en')
    
    args = parser.parse_args()
    return args

class WordPieceTokenizer(object):
    def __init__(self, datapath, vocab_size=16000, l=0, alpha=0, n=0):
        logging.info(f"vocab_size={vocab_size}")
        self.templates = "--input={} --model_prefix={} --vocab_size={} --bos_id=2 --eos_id=3 --pad_id=0 --unk_id=1"
        self.vocab_size = vocab_size
        self.spm_path = f"{datapath}/sp"

        # for subword regularization
        self.l = l
        self.alpha = alpha
        self.n = n

    def transform(self, sentence, max_length=0):
        if self.l and self.alpha:
            x = self.sp.SampleEncodeAsIds(sentence, self.l, self.alpha)
        elif self.n:
            x = self.sp.NBestEncodeAsIds(sentence, self.n)
        else:
            x = self.sp.EncodeAsIds(sentence)
        if max_length>0:
            pad = [0]*max_length
            pad[:min(len(x), max_length)] = x[:min(len(x), max_length)]
            x = pad
        return x

    def fit(self, input_file):
        cmd = self.templates.format(input_file, self.spm_path, self.vocab_size, 0)
        spm.SentencePieceTrainer.Train(cmd)

    def load_model(self):
        file = self.spm_path + ".model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(file)
        self.sp.SetEncodeExtraOptions('eos')
        print(f"load_model {file}")
        return self

    def decode(self, encoded_sentences):
        decoded_output = []
        for encoded_sentence in encoded_sentences:
            x = self.sp.DecodeIds(encoded_sentence)
            decoded_output.append(x)

        return decoded_output
    
    def __len__(self):
        return len(self.sp)

def main():
    args = get_args()
    tokenizer = WordPieceTokenizer(datapath=args.datapath, 
                                    vocab_size=args.vocab_size)
    tokenizer.fit(",".join([args.src_path, args.dest_path]))                                    

if __name__=="__main__":
    main()
