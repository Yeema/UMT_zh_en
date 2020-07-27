import os
import sys
import re
# Suppress as many warnings as possible
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from ckiptagger import construct_dictionary, WS

def main():
    # Download data
    #data_utils.download_data("./")
    
    # Load model without GPU
    ws = WS("/raid/yihui/chinese/ckiptagger/data",disable_cuda=False)
    #pos = POS("./data")
    #ner = NER("./data")
    
    # Load model with GPU
    # ws = WS("./data", disable_cuda=False)
    # pos = POS("./data", disable_cuda=False)
    # ner = NER("./data", disable_cuda=False)
    
    # Create custom dictionary
    # word_to_weight = {
    #     "土地公": 1,
    #     "土地婆": 1,
    #     "公有": 2,
    #     "": 1,
    #     "來亂的": "啦",
    #     "緯來體育台": 1,
    # }
    # dictionary = construct_dictionary(word_to_weight)
    # print(dictionary)
    
    # Run WS-POS-NER pipeline
    sentence_list = """雷鳥 lt-234 日式木紋桌墊 45 x 60 cm / 片.
狂砂小子dvd vol-08,
anacomda 巨蟒 t1 泰坦系列 120gb ssd固態硬碟,
ifairies 單肩斜背包側背包肩背包【49146】,
大雪山農場 山苦瓜茶20包/盒,
高露潔 雙效潔淨牙刷單支,
apieu color lip pencil matt絲緞,
芮菲客米蘭玫瑰金鋼珠筆,
海底總動員2 玻璃磁鐵 多莉款_野獸國,
natural甘草瓜子160g 南瓜子150g shopping168,
oppo a75 a75s a73 手机壳 软壳 挂绳壳 大眼兔硅胶壳 """.split(',')
    
    word_sentence_list = []
    for sentence in sentence_list:
        sent_splits = []
        res = ws(re.findall(r'\S+', sentence))
        tokens = []
        for r in res:
            tokens.extend(r)
        word_sentence_list.append(' '.join([t for t in tokens if t.strip() is not None]))
    # word_sentence_list = ws(sentence_list, sentence_segmentation=True)
    # word_sentence_list = ws(sentence_list, recommend_dictionary=dictionary)
    # word_sentence_list = ws(sentence_list, coerce_dictionary=dictionary)
    #pos_sentence_list = pos(word_sentence_list)
    #entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    
    # Release model
    del ws
    #del pos
    #del ner
    
    # Show results
    '''
    def print_word_pos_sentence(word_sentence, pos_sentence):
        assert len(word_sentence) == len(pos_sentence)
        for word, pos in zip(word_sentence, pos_sentence):
            print(f"{word}({pos})", end="\u3000")
        print()
        return
    ''' 
    for i, sentence in enumerate(sentence_list):
        print()
        print(f"'{sentence}'")
        print(word_sentence_list[i])
        '''print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])
        for entity in sorted(entity_sentence_list[i]):
            print(entity)'''
    return
    
if __name__ == "__main__":
    main()
    sys.exit()
    

