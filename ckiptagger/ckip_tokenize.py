import os
import sys
import re
from ckiptagger import construct_dictionary, WS

def main():
    # Download data
    #data_utils.download_data("./")
    
    # Load model without GPU
    ws = WS("/raid/yihui/chinese/ckiptagger/data",disable_cuda=False)  

    word_sentence_list = []
    for sentence in sys.stdin.readlines():
        sent_splits = []
        sentence = sentence.strip()
        res = ws(re.findall(r'\S+', sentence))
        tokens = []
        for r in res:
            tokens.extend(r)
        print(' '.join([t.strip() for t in tokens if t.strip()]))
    
    # Release model
    del ws
    

    return
    
if __name__ == "__main__":
    main()
    sys.exit()
    

