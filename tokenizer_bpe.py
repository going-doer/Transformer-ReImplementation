import argparse
import logging
import re, collections, os
from bs4 import BeautifulSoup
import pickle
from tqdm import tqdm
LOGGER = logging.getLogger()

def get_args():
    parser = argparse.ArgumentParser()

    # Run Settings
    parser.add_argument('--datapath',
                        default='./datasets/train/train.vocab')
    parser.add_argument('--savepath',
                        default='./outputs')
    parser.add_argument('--lang',
                        default='en')
    parser.add_argument('--vocab_size',
                        default=16000)
    parser.add_argument('--num_merges',
                        default=10)
    
    args = parser.parse_args()
    return args

class BPETokenizer(object):
    def __init__(self, datapath, savepath, lang="en", vocab_size=16000, num_merges=100):
        logging.info(f"data path={datapath}")
        self.datapath = datapath

        bpe_path = f"{savepath}/bpe"
        if not os.path.exists(bpe_path):
            os.makedirs(bpe_path)

        # TODO: 전체 train dataset으로 학습하는데 너무 오랜 시간이 걸려 잠시 보류
        # TODO: bpe_Codes_revers 삭제해야 함.
        # self.bpe_codes_path = os.path.join(bpe_path, f'bpe_codes_{lang}.pkl')
        # self.bpe_codes_reverse_path = os.path.join(bpe_path, f'bpe_codes_reverse_{lang}.pkl')
        self.bpe_codes_path = os.path.join(bpe_path, f'bpe_codes.pkl')
        self.bpe_codes_reverse_path = os.path.join(bpe_path, f'bpe_codes_reverse.pkl')
        self.bpe_vocab_binary_path = os.path.join(bpe_path, f'vocabulary.pkl')
        self.bpe_vocab_path = os.path.join(bpe_path, f'vocab.txt')

        self.bpe_codes={}
        self.bpe_codes_reverse={}
        self.vocab_dict = {}

        self.vocab_size = vocab_size
        self.num_merges = num_merges

    def train(self):
        def get_stats(dictionary):
            pairs = collections.defaultdict(int)
            for word, freq in dictionary.items():
                symbols = word.split()
                for i in range(len(symbols)-1):
                    pairs[symbols[i], symbols[i+1]] += freq
            # print('현재 pairs 들의 빈도수: ', dict(pairs))
            return pairs
        
        def merge_dictionary(pair, v_in):
            v_out = {}
            bigram = re.escape(' '.join(pair)) # 특수문자는 escape
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            for word in v_in:
                w_out = p.sub(''.join(pair), word) # 다 바꾸기.
                v_out[w_out] = v_in[word]
            return v_out
            
        if os.path.splitext(self.datapath) == ".sgm":
            with open (self.datapath,"r", encoding='utf-8')as f:
                contents =f.read()
                soup = BeautifulSoup(contents, 'html.parser')
                sgm_lines = soup.findAll('seg') # lines.text    

            lines = []
            for line in sgm_lines:
                lines.append(line.text) 
        else:
            with open (self.datapath,"r", encoding='utf-8')as f:
                lines =f.readlines()

        
        dictionary = {}
        dictionary = collections.defaultdict(int)
        vocabulary = set()
        for line in lines:
            #print(line.text)
            for word in line.rstrip().split(" "):
                dict_word = ""
                for character in word:
                    dict_word += character + " "
                    vocabulary.add(character)

                dict_word += "</w>"
                
                dictionary[dict_word] += 1

        bpe_codes = {}
        bpe_codes_reverse = {}

        

        # hyperparameter
        for i in tqdm(range(self.num_merges)):
            pairs = get_stats(dictionary)
            try:
                best = max(pairs, key=pairs.get)
            except:
                # print(f"----------- last iteration {i} -----------")
                break
            
            dictionary = merge_dictionary(best, dictionary)

            bpe_codes[best] = i
            bpe_codes_reverse[best[0]+best[1]] = best

            # print(f"new merge: {best}")
            # print(f"dictionary: {dictionary}")
            vocabulary.add(''.join(best))
            if len(vocabulary) > self.vocab_size:  # hyperparameter
                print("----------- vocabulary size: 16000 -----------")
                break
        
        self.bpe_codes = bpe_codes
        self.bpe_codes_reverse = bpe_codes_reverse

        self.vocab = vocabulary

        vocab_dict = {}
        for i, vocab in enumerate(vocabulary):
            vocab_dict[vocab] = i
        self.vocab_dict = vocab_dict

        # save the dictionary and vocabulary
        with open(self.bpe_codes_path, 'wb') as f:
            pickle.dump(self.bpe_codes, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(self.bpe_codes_reverse_path, 'wb') as f:
            pickle.dump(self.bpe_codes_reverse, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(self.bpe_vocab_binary_path, 'wb') as f:
            pickle.dump(self.vocab_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.bpe_vocab_path, 'w', encoding='utf-8') as f:
            for k in self.vocab_dict:
                f.write(f'{k}:{self.vocab_dict[k]}\n')
        
    def transform(self, orig, max_seq_len=512):
        def get_pairs(word):
            """
            Return set of symbol pairs in a word.
            Word is represented as a tuple of symbols (symbols being variable-length strings)
            """
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char, char))
                prev_char = char
            return pairs

        """Encode word based on list of BPE merge operations, which are applied consecutively"""
        
        sentence = []

        for orig_word in orig.split(" "):
            word = tuple(orig_word) + ('</w>', )

            pairs = get_pairs(word)
            if not pairs:
                return orig
            
            iteration = 0
            while True:
                iteration += 1
                # TODO: logger로 변경하기.
                # print(f"bigrams in the word: {pairs}")
                bigram = min(pairs, key=lambda pair: self.bpe_codes.get(pair, float('inf')))
                if bigram not in self.bpe_codes:
                    # print("Candidate not in BPE merges, algorithm stops")
                    break
                first, second = bigram
                new_word = []
                i = 0
                while i<len(word):
                    try:
                        j=word.index(first, i)
                        new_word.extend(word[i:j])
                        i=j
                    except:
                        new_word.extend(word[i:])
                        break
                        
                    if word[i] == first and i<len(word)-1 and word[i+1]==second:
                        new_word.append(first+second)
                        i+=2
                    else:
                        new_word.append(word[i])
                        i+=1

                new_word = tuple(new_word)
                word = new_word
                # print(f"word after merging: {word}")
                if len(word)==1:
                    break
                else:
                    pairs = get_pairs(word)

            # 특별 토큰인 </w>는 출력하지 않음.
            # if word[-1]=="</w>":
            #     word = word[:-1]
            # elif word[-1].endswith('</w>'):
            #     word = word[:-1] + (word[-1].replace('</w>', ''), )
            sentence.append(word)

        ids = []
        for word in sentence:
            for w in word:
                try:
                    # ids.append(self.bpe_codes[self.bpe_codes_reverse[w]])
                    ids.append(self.vocab_dict[w])
                except:
                    ids.append(-1) # unknown
        
        # print("sentence: ", sentence)
        # print("ids: ", ids)
        return ids # word
    
    def fit(self):
        # Data loader
        with open(self.bpe_codes_path, 'rb') as f:
            self.bpe_codes = pickle.load(f)

        with open(self.bpe_codes_reverse_path, 'rb') as f:
            self.bpe_codes_reverse = pickle.load(f)
        
        with open(self.bpe_vocab_binary_path, 'rb') as f:
            self.vocab_dict = pickle.load(f)
        
        # print("fit tokenizer: ", len(self.bpe_codes_reverse.keys()))

        if len(self.bpe_codes) == 0:
            self.train()

def main():
    """
    Train BPE Model
    """
    args = get_args()
    tokenizer = BPETokenizer(
                    datapath=args.datapath, 
                    savepath=args.savepath, 
                    lang=args.lang, 
                    vocab_size=args.vocab_size, 
                    num_merges=args.num_merges
                )


    tokenizer.train()

    # USAGE: If you already have a trained tokenizer
    # tokenizer.fit()

if __name__ == "__main__":
    main()