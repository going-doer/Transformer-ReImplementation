import argparse
import logging
import re, collections, os
from bs4 import BeautifulSoup
import pickle
LOGGER = logging.getLogger()

def get_args():
    parser = argparse.ArgumentParser()

    # Run Settings
    parser.add_argument('--datapath',
                        default='./datasets/train/train.en')
    parser.add_argument('--lang',
                        default='en')
    parser.add_argument('--savepath',
                        default='./outputs')
    
    args = parser.parse_args()
    return args

class BPETokenizer(object):
    def __init__(self, datapath, savepath, lang="en"):
        logging.info(f"data path={datapath}")
        self.datapath = datapath

        bpe_path = f"{savepath}/bpe"
        if not os.path.exists(bpe_path):
            os.makedirs(bpe_path)

        self.bpe_codes_path = os.path.join(bpe_path, f'bpe_codes_{lang}.pkl')
        self.bpe_codes_reverse_path = os.path.join(bpe_path, f'bpe_codes_reverse_{lang}.pkl')

        self.bpe_codes={}
        self.bpe_codes_reverse={}

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
        for line in lines:
            #print(line.text)
            for word in line.rstrip().split(" "):
                dict_word = ""
                for character in word:
                    dict_word += character + " "
                dict_word += "</w>"
                
                dictionary[dict_word] += 1

        bpe_codes = {}
        bpe_codes_reverse = {}

        i=0
        while True:
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
            i += 1
        
        self.bpe_codes = bpe_codes
        self.bpe_codes_reverse = bpe_codes_reverse

        # save the dictionary
        with open(self.bpe_codes_path, 'wb') as f:
            pickle.dump(self.bpe_codes, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(self.bpe_codes_reverse_path, 'wb') as f:
            pickle.dump(self.bpe_codes_reverse, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        
    def transform(self, orig, max_length=0):
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
        word = tuple(orig) + ('</w>', )
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
        if word[-1]=="</w>":
            word = word[:-1]
        elif word[-1].endswith('</w>'):
            word = word[:-1] + (word[-1].replace('</w>', ''), )
        
        ids = []
        for w in word:
            try:
                ids.append(self.bpe_codes[self.bpe_codes_reverse[w]])
            except:
                ids.append(-1) # unknown
        
        if max_length>0:
            pad = [0]*max_length
            pad[:min(max_length, len(word))] = ids[:min(max_length, len(word))]
            ids = pad

        return ids # word
    
    def fit(self):
        # 데이터 로드
        with open(self.bpe_codes_path, 'rb') as f:
            self.bpe_codes = pickle.load(f)

        with open(self.bpe_codes_reverse_path, 'rb') as f:
            self.bpe_codes_reverse = pickle.load(f)
        
        # print("fit tokenizer: ", len(self.bpe_codes_reverse.keys()))

        if len(self.bpe_codes) == 0:
            self.train()

def main():
    """
    Train BPE Model
    """
    args = get_args()
    tokenizer = BPETokenizer(datapath=args.datapath, savepath=args.savepath, lang=args.lang)


    # TODO: train할 경우에만
    tokenizer.train()

    # TODO: 호출할 경우에만
    # tokenizer.fit()

if __name__ == "__main__":
    main()