# Tools

In `XLM/tools/`, you will need to install the following tools:

## Tokenizers

[Moses](https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer) tokenizer:
```
git clone https://github.com/moses-smt/mosesdecoder
```

Thai [PythaiNLP](https://github.com/PyThaiNLP/pythainlp) tokenizer:
```
pip install pythainlp
```

Japanese [KyTea](http://www.phontron.com/kytea) tokenizer:
```
wget http://www.phontron.com/kytea/download/kytea-0.4.7.tar.gz
tar -xzf kytea-0.4.7.tar.gz
cd kytea-0.4.7
./configure
make
make install
kytea --help
```

Chinese Stanford segmenter:
```
wget https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip
unzip stanford-segmenter-2018-10-16.zip
```
Chinese [ckiptagger](https://github.com/ckiplab/ckiptagger):
```
# if you have tensorflow
pip install -U ckiptagger[tf,gdown]

# (Complete installation) If you have just set up a clean virtual environment, and want everything, including GPU support
pip install -U ckiptagger[tfgpu,gdown]
```
CkipTagger is a Python library hosted on PyPI. Requirements:

- python>=3.6
- tensorflow>=1.13.1,<2 / tensorflow-gpu>=1.13.1,<2 (one of them)
- gdown (optional, for **downloading** model files from **google drive**)

## fastBPE

```
git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```
