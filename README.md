# XLM mainly focused on unsupervised machine translation

**Notice:** Read [original readme](https://github.com/Yeema/XLM/blob/master/ORIGINAL_README.md) for installation first.

## Main Concept
Training step is
1. training LM on monolingual/ multiple monolingual/ parallel monolingual corpus
  1. preparing training data in the format of one sentence one line. Avoid too long sentences or please setting ```--bptt``` when you run train.py.
  2. applying BPE (Byte Pair Encoding).
  3. binarizing BPE sentences.
  4. training your own bert model.
2. fintuning a specific case like UMT **baed on pretrained** LM
  1. training UMT model using pretrined [XLM](https://github.com/Yeema/XLM/blob/master/ORIGINAL_README.md#pretrained-cross-lingual-language-models) or your own LM
  2. [translating](https://github.com/Yeema/XLM/blob/master/train.py)

## Unsupervised Machine Translation without parallel data but only with  monolingual data (MLM)
In what follows we explain how you can train your own cross-lingual BERT model on multiple monolingual corpus or use [pretrained XLM](https://github.com/Yeema/XLM/blob/master/ORIGINAL_README.md#pretrained-cross-lingual-language-models) weights. In [ORIGINAL_README](https://github.com/Yeema/XLM/blob/master/ORIGINAL_README.md), it shows how to train monolingual BERT model.

### Train your own cross-lingual BERT model with multiple monolingual dataset
Now in what follows, we will explain how you can train an XLM model on your own data.

### 1. Preparing the data
First, get the multiple [monolingual corpus]().

Transform from csv to ***one sentence one line*** format.
```
import pandas as pd
import re
import regex
re.Pattern = regex.compile(r'\p{So}')
raw_data_dir = '/raid/yihui/titles/'
files = ['dev_en.csv','dev_tcn.csv','test_tcn.csv','train_en.csv','train_tcn.csv']
for filename in files:
    file_type, lng = filename[:-4].split('_')
    df = pd.read_csv(raw_data_dir+filename)
    if lng =='tcn':
        lng = 'zh'
    fd = open(raw_data_dir+'%s.%s'%(file_type,lng),'w')
    print(df.head())
    if 'translation_output' in df.columns:
        print( '\n'.join([re.Pattern.sub(' ',str(line).lower()) for line in df['translation_output'].tolist()]), file=fd)
    elif 'text' in df.columns:
        print( '\n'.join([re.Pattern.sub(' ',str(line).lower()) for line in df['text'].tolist()]), file=fd)
    else:
        print( '\n'.join([re.Pattern.sub(' ',str(line).lower()) for line in df['product_title'].tolist()]), file=fd)
    fd.close()
```

### 2.I. Train your own Cross-lingual language model (XLM)

1. Tokenize sentences and save tokenized data in one sentence one line format.
```
lg=en
RAW_DATA_PATH=/raid/yihui/titles
cat $RAW_DATA_PATH/dev.$lg | ./tools/tokenize.sh $lg > $RAW_DATA_PATH/tokenized_valid.$lg &
cat $RAW_DATA_PATH/train.$lg | ./tools/tokenize.sh $lg > $RAW_DATA_PATH/tokenized_train.$lg &

lg=zh
cat $RAW_DATA_PATH/dev.$lg | ./tools/tokenize.sh $lg > $RAW_DATA_PATH/tokenized_valid.$lg &
cat $RAW_DATA_PATH/test.$lg | ./tools/tokenize.sh $lg > $RAW_DATA_PATH/tokenized_test.$lg &
cat $RAW_DATA_PATH/train.$lg | ./tools/tokenize.sh $lg > $RAW_DATA_PATH/tokenized_train.$lg &
```

2. [Install fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) under the directory of XLM and **learn BPE** vocabulary (with 350000 codes here) of multiple monolingual sentences from training set: You can decide codes number on your own but larger codes longer producing time.
```
OUTPATH=/raid/yihui/titles/XLM_zhen
# create output path
mkdir -p $OUTPATH
FASTBPE=tools/fastBPE/fast  # path to the fastBPE tool

# learn 350000 bpe codes on the training set and output bpe codes to $OUTPATH/codes_zhen -> you can define your own code filename
$FASTBPE learnbpe 350000 $RAW_DATA_PATH/tokenized_train.zh $RAW_DATA_PATH/tokenized_train.en > $OUTPATH/codes_zhen
```

3. Now **apply BPE** encoding to all files (train/valid/test):
```
pair=zh-en
CODE=codes_zhen
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    FILE=$RAW_DATA_PATH/tokenized_$split.$lg
    if [ -f "$FILE" ]; then
        # $FASTBPE applybpe [output] [input] [codes]
        $FASTBPE applybpe $OUTPATH/$split.$lg $RAW_DATA_PATH/tokenized_$split.$lg $OUTPATH/$CODE &
    else
        echo "$FILE does not exist."
    fi
  done
done

'''
# equals to the following
$FASTBPE applybpe $OUTPATH/train.zh $RAW_DATA_PATH/tokenized_train.zh $OUTPATH/codes_zhen &
$FASTBPE applybpe $OUTPATH/test.zh $RAW_DATA_PATH/tokenized_test.zh $OUTPATH/codes_zhen &
$FASTBPE applybpe $OUTPATH/valid.zh $RAW_DATA_PATH/tokenized_valid.zh $OUTPATH/codes_zhen &

$FASTBPE applybpe $OUTPATH/train.en $RAW_DATA_PATH/tokenized_train.en $OUTPATH/codes_zhen &
$FASTBPE applybpe $OUTPATH/valid.en $RAW_DATA_PATH/tokenized_valid.en $OUTPATH/codes_zhen &
'''
```

**optional** ... this step is to produce test,valid file for training
```
mv $OUTPATH/test.zh $OUTPATH/pred_test.zh
# total number of test sentences in valid.{zh,en} is 1k respectively
tail -n 5 $OUTPATH/valid.zh > $OUTPATH/test.zh
tail -n 5 $OUTPATH/valid.en > $OUTPATH/test.en
head -n 995 $OUTPATH/valid.zh > $OUTPATH/valid1.zh
mv $OUTPATH/valid1.zh $OUTPATH/valid.zh
head -n 995 $OUTPATH/valid.en > $OUTPATH/valid1en
mv $OUTPATH/valid1.en $OUTPATH/valid.en
```

4. and **get** the post-BPE **vocabulary**:
```
cat $OUTPATH/train.zh $OUTPATH/train.en | $FASTBPE getvocab - > $OUTPATH/vocab &
```

5. **Binarize the data** to limit the size of the data we load in memory: input xxx output xxx.path
```
# This will create three files: $OUTPATH/{train,valid,test}.{zh,en}.pth
# After that we're all set
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    FILE=$RAW_DATA_PATH/tokenized_$split.$lg
    if [ -f "$FILE" ]; then
        python preprocess.py $OUTPATH/vocab $OUTPATH/$split.$lg &
    else
        echo "$FILE does not exist."
    fi
  done
done
''' 
# equals to the following
python preprocess.py $OUTPATH/vocab $OUTPATH/train.zh &
python preprocess.py $OUTPATH/vocab $OUTPATH/valid.zh &
python preprocess.py $OUTPATH/vocab $OUTPATH/test.zh &

python preprocess.py $OUTPATH/vocab $OUTPATH/train.en &
python preprocess.py $OUTPATH/vocab $OUTPATH/valid.en &
python preprocess.py $OUTPATH/vocab $OUTPATH/test.en &
'''
```

6. (optional) make sure you have 10 files with exactly same names in $OUTPATH 
```
Monolingual training data:
    zh: train.zh.pth
    en: train.en.pth
Monolingual validation data:
    zh: valid.zh.pth
    en: valid.en.pth
Monolingual test data:
    zh: test.zh.pth
    en: test.en.pth
Parallel validation data:
    zh: valid.zh-en.zh.pth
    en: valid.zh-en.en.pth
Parallel test data:
    zh: test.zh-en.zh.pth
    en: test.zh-en.en.pth
```

Here, I personally choose to set Monolingual test,validation data same as Parallel one but "Monolingual test,validation data" are indeed parallel corpus.

```
cp $OUTPATH/valid.zh.pth $OUTPATH/valid.en-zh.zh.pth
cp $OUTPATH/valid.en.pth $OUTPATH/valid.en-zh.en.pth
cp $OUTPATH/test.en.pth $OUTPATH/test.en-zh.en.pth
cp $OUTPATH/test.zh.pth $OUTPATH/test.en-zh.zh.pth
```

7. Train your XLM (MLM only) on the preprocessed data and please check the number of tokens in a training senece is smaller than default or you can set ```--bptt 256``` to meet your need:

```

python train.py

## main parameters
--exp_name test_zhen_mlm                   # experiment name
--dump_path /raid/yihui/titles/mlm         # where to store the experiment

## data location / training objective
--data_path /raid/yihui/titles/XLM_zhen/   # data location
--lgs 'zh-en'                              # considered languages
--clm_steps ''                             # CLM objective (for training GPT-2 models)
--mlm_steps 'zh,en'                        # MLM objective

## transformer parameters
--emb_dim 1024                             # embeddings / model dimension (2048 is big, reduce if only 16Gb of GPU memory)
--n_layers 6                               # number of layers
--n_heads 8                                # number of heads
--dropout 0.1                              # dropout
--attention_dropout 0.1                    # attention dropout
--gelu_activation true                     # GELU instead of ReLU

## optimization
--batch_size 32                            # sequences per batch
--bptt 256                                 # sequences length  (streams of 256 tokens)
--optimizer adam,lr=0.0001                 # optimizer (training is quite sensitive to this parameter)
--epoch_size 200000                        # number of sentences per epoch
--max_epoch 100000                         # max number of epochs (~infinite here)
--validation_metrics _valid_mlm_ppl        # validation metric (when to save the best model)
--stopping_criterion _valid_mlm_ppl,15     # stopping criterion (if criterion does not improve 15 times)

## There are other parameters that are not specified here (see [here](https://github.com/facebookresearch/XLM/blob/master/train.py#L24-L198)).
```

To [train with multiple GPUs](https://github.com/facebookresearch/XLM#how-can-i-run-experiments-on-multiple-gpus) use:
```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
```

**Tips**: Even when the validation perplexity plateaus, keep training your model. The larger the batch size the better (so using multiple GPUs will improve performance). Tuning the learning rate (e.g. [0.0001, 0.0002]) should help.
**Tips**: Avoid setting epoch_size too small!!! It means how many samples will be learned in an epoch.

### 2.II Using pretrained cross-lingual language models

[Facebook Research](https://github.com/Yeema/XLM/blob/master/ORIGINAL_README.md#pretrained-cross-lingual-language-models) provide large pretrained models for the 15 languages of [XNLI](https://github.com/facebookresearch/XNLI), and two other models in [17 and 100 languages](#the-17-and-100-languages).

|Languages|Pretraining|Tokenization                          |  Model                                                              | BPE codes                                                            | Vocabulary                                                            |
|---------|-----------|--------------------------------------| ------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------- |
|15       |    MLM    |tokenize + lowercase + no accent + BPE| [Model](https://dl.fbaipublicfiles.com/XLM/mlm_xnli15_1024.pth)     | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15) (80k)  | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15) (95k)  |
|15       | MLM + TLM |tokenize + lowercase + no accent + BPE| [Model](https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth) | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_15) (80k)  | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15) (95k)  |
|17       |    MLM    |tokenize + BPE                        | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_17_1280.pth)         | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_17) (175k) | [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_17) (200k) |
|100      |    MLM    |tokenize + BPE                        | [Model](https://dl.fbaipublicfiles.com/XLM/mlm_100_1280.pth)        | [BPE codes](https://dl.fbaipublicfiles.com/XLM/codes_xnli_100) (175k)| [Vocabulary](https://dl.fbaipublicfiles.com/XLM/vocab_xnli_100) (200k)|

If you want to play around with the model and its representations, just download the model and take a look at our [ipython notebook](https://github.com/facebookresearch/XLM/blob/master/generate-embeddings.ipynb) demo.

#### The 17 and 100 Languages

The XLM-17 model includes these languages: en-fr-es-de-it-pt-nl-sv-pl-ru-ar-tr-zh-ja-ko-hi-vi

The XLM-100 model includes these languages: en-es-fr-de-zh-ru-pt-it-ar-ja-id-tr-nl-pl-simple-fa-vi-sv-ko-he-ro-no-hi-uk-cs-fi-hu-th-da-ca-el-bg-sr-ms-bn-hr-sl-zh_yue-az-sk-eo-ta-sh-lt-et-ml-la-bs-sq-arz-af-ka-mr-eu-tl-ang-gl-nn-ur-kk-be-hy-te-lv-mk-zh_classical-als-is-wuu-my-sco-mn-ceb-ast-cy-kn-br-an-gu-bar-uz-lb-ne-si-war-jv-ga-zh_min_nan-oc-ku-sw-nds-ckb-ia-yi-fy-scn-gan-tt-am

1. Tokenize sentences and save tokenized data in one sentence one line format.
precessing same as [Train your Cross-lingual language model (XLM)](https://github.com/Yeema/XLM#train-your-cross-lingual-language-model-(XLM))

2. Download pretrained model weights, BPE codes and vocab: Here I download [mlm_17](https://github.com/Yeema/XLM#pretrained-cross-lingual-language-models) as example.
```
OUTPATH=/raid/yihui/titles/mlm17
cd $OUTPATH

# download model weights
wget -c https://dl.fbaipublicfiles.com/XLM/mlm_17_1280.pth
# download BPE codes
wget -c https://dl.fbaipublicfiles.com/XLM/codes_xnli_17
# download model weights
wget -c https://dl.fbaipublicfiles.com/XLM/vocab_xnli_17
```

3. Now **apply BPE** encoding to all files (train/valid/test):
```
cd ~/XLM #go to XLM directory
FASTBPE=tools/fastBPE/fast
pair=zh-en
CODE=codes_xnli_17
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    FILE=$RAW_DATA_PATH/tokenized_$split.$lg
    if [ -f "$FILE" ]; then
        $FASTBPE applybpe $OUTPATH/$split.$lg $RAW_DATA_PATH/tokenized_$split.$lg $OUTPATH/$CODE &
    else
        echo "$FILE does not exist."
    fi
  done
done
'''
# equals to the following
$FASTBPE applybpe $OUTPATH/train.zh $RAW_DATA_PATH/tokenized_train.zh $OUTPATH/codes_xnli_17 &
$FASTBPE applybpe $OUTPATH/test.zh $RAW_DATA_PATH/tokenized_test.zh $OUTPATH/codes_xnli_17 &
$FASTBPE applybpe $OUTPATH/valid.zh $RAW_DATA_PATH/tokenized_valid.zh $OUTPATH/codes_xnli_17 &

$FASTBPE applybpe $OUTPATH/train.en $RAW_DATA_PATH/tokenized_train.en $OUTPATH/codes_xnli_17 &
$FASTBPE applybpe $OUTPATH/valid.en $RAW_DATA_PATH/tokenized_valid.en $OUTPATH/codes_xnli_17 &
'''
```

**optional** ... this step is to produce test,valid file for training
```
mv $OUTPATH/test.zh $OUTPATH/pred_test.zh
# total number of test sentences in valid.{zh,en} is 1k respectively
tail -n 5 $OUTPATH/valid.zh > $OUTPATH/test.zh
tail -n 5 $OUTPATH/valid.en > $OUTPATH/test.en
head -n 995 $OUTPATH/valid.zh > $OUTPATH/valid1.zh
mv $OUTPATH/valid1.zh $OUTPATH/valid.zh
head -n 995 $OUTPATH/valid.en > $OUTPATH/valid1.en
mv $OUTPATH/valid1.en $OUTPATH/valid.en
```

4. **Binarize the data** to limit the size of the data we load in memory: input xxx output xxx.path
```
# This will create three files: $OUTPATH/{train,valid,test}.{zh,en}.pth
# After that we're all set
VOCAB=vocab_xnli_17
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    FILE=$OUTPATH/$split.$lg
    if [ -f "$FILE" ]; then
        python preprocess.py $OUTPATH/$VOCAB $FILE&
    else
        echo "$FILE does not exist."
    fi
  done
done
''' 
# equals to the following
python preprocess.py $OUTPATH/vocab_xnli_17 $OUTPATH/train.zh &
python preprocess.py $OUTPATH/vocab_xnli_17 $OUTPATH/valid.zh &
python preprocess.py $OUTPATH/vocab_xnli_17 $OUTPATH/test.zh &

python preprocess.py $OUTPATH/vocab_xnli_17 $OUTPATH/train.en &
python preprocess.py $OUTPATH/vocab_xnli_17 $OUTPATH/valid.en &
python preprocess.py $OUTPATH/vocab_xnli_17 $OUTPATH/test.en &
'''
```
5. (optional) make sure you have 10 files with exactly same name in $OUTPATH 
```
Monolingual training data:
    zh: train.zh.pth
    en: train.en.pth
Monolingual validation data:
    zh: valid.zh.pth
    en: valid.en.pth
Monolingual test data:
    zh: test.zh.pth
    en: test.en.pth
Parallel validation data:
    zh: valid.zh-en.zh.pth
    en: valid.zh-en.en.pth
Parallel test data:
    zh: test.zh-en.zh.pth
    en: test.zh-en.en.pth
```

Here, I personally choose to set Monolingual test,validation data same as Parallel one but "Monolingual test,validation data" are indeed parallel corpus.

```
cp $OUTPATH/valid.zh.pth $OUTPATH/valid.en-zh.zh.pth
cp $OUTPATH/valid.en.pth $OUTPATH/valid.en-zh.en.pth
cp $OUTPATH/test.en.pth $OUTPATH/test.en-zh.en.pth
cp $OUTPATH/test.zh.pth $OUTPATH/test.en-zh.zh.pth
```

### 3. Fine-tune XLM models (Applications, see below)

Cross-lingual language model (XLM) provides a strong pretraining method for cross-lingual understanding (XLU) tasks. In what follows, we present applications to machine translation (unsupervised and supervised) and cross-lingual classification (XNLI).


## 3. Applications: Unsupervised MT

XLMs can be used as a pretraining method for unsupervised or supervised neural machine translation. Here we provide the step to train from zh to en.


### Train on unsupervised MT **from a pretrained model**

Using your pretrained LM weight
```
python train.py 
--exp_name unsupMT_zhen 
--dump_path /raid/yihui/titles/xlm 
--reload_model '/raid/yihui/titles/mlm/test_zhen_mlm/3myx2ymkhz/best-valid_mlm_ppl.pth,/raid/yihui/titles/mlm/test_zhen_mlm/3myx2ymkhz/best-valid_mlm_ppl.pth' 
--data_path $OUTPATH 
--lgs 'zh-en' 
--ae_steps 'zh,en' 
--bt_steps 'zh-en-zh,en-zh-en' 
--word_shuffle 3 
--word_dropout 0.1 
--word_blank 0.1 
--lambda_ae '0:1,100000:0.1,300000:0' 
--encoder_only false 
--emb_dim 1024 
--n_layers 6 
--n_heads 8 
--dropout 0.1 
--attention_dropout 0.1 
--eval_bleu true 
--gelu_activation true 
--batch_size 8 
--max_epoch 100 
--bptt 256 
--optimizer adam_inverse_sqrt,lr=0.00010,warmup_updates=30000,beta1=0.9,beta2=0.999,weight_decay=0.01,eps=0.000001 
--epoch_size 200000 
--stopping_criterion 'valid_zh-en_mt_bleu,2' 
--validation_metrics 'valid_zh-en_mt_bleu'
```

Using downloaded pretrained weight
1. --emb_dim should be set the same as pretrained weights' original model
2. --max_vocab 200000 can gaurantee the number of unique words can be fed into the current model
3. --epoch_size is the number of sentences can be trained in an epoch so don't set it so small that the model couldn't learn well

```
python train.py

## main parameters
--exp_name unsupMT_zhen                                       # experiment name
--dump_path /raid/yihui/titles/mlm17                          # where to store the experiment
--reload_model '/raid/yihui/titles/mlm17/mlm_17_1280.pth,/raid/yihui/titles/mlm17/mlm_17_1280.pth'          # model to reload for encoder,decoder

## data location / training objective
--data_path $OUTPATH                                          # data location
--lgs 'zh-en'                                                 # considered languages
--ae_steps 'zh,en'                                            # denoising auto-encoder training steps
--bt_steps 'zh-en-zh,en-zh-en'                                # back-translation steps
--word_shuffle 3                                              # noise for auto-encoding loss
--word_dropout 0.1                                            # noise for auto-encoding loss
--word_blank 0.1                                              # noise for auto-encoding loss
--lambda_ae '0:1,100000:0.1,300000:0'                         # scheduling on the auto-encoding coefficient

## transformer parameters
--encoder_only false                                          # use a decoder for MT
--emb_dim 1208                                                # embeddings / model dimension
--n_layers 6                                                  # number of layers
--n_heads 8                                                   # number of heads
--dropout 0.1                                                 # dropout
--attention_dropout 0.1                                       # attention dropout
--gelu_activation true                                        # GELU instead of ReLU

## optimization
--beam_size 3
--max_vocab 200000
--max_epoch 100
--batch_size 8                                                # batch size (for back-translation)
--bptt 256                                                    # sequence length
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001  # optimizer
--epoch_size 200000                                           # number of sentences per epoch
--eval_bleu true                                              # also evaluate the BLEU score
--stopping_criterion 'valid_zh-en_mt_bleu,10'                 # validation metric (when to save the best model)
--validation_metrics 'valid_zh-en_mt_bleu'                    # end experiment if stopping criterion does not improve
```

The parameters of your Transformer model have to be identical to the ones used for pretraining (or you will have to slightly modify the code to only reload existing parameters). After 8 epochs on 1 GPU, the above command should give you something like this:

```
epoch               ->     7
valid_fr-en_mt_bleu -> 28.36
valid_en-fr_mt_bleu -> 30.50
test_fr-en_mt_bleu  -> 34.02
test_en-fr_mt_bleu  -> 36.62
```

## Frequently Asked Questions

### How can I run experiments on multiple GPUs?

XLM supports both multi-GPU and multi-node training, and was tested with up to 128 GPUs. To run an experiment with multiple GPUs on a single machine, simply replace `python train.py` in the commands above with:

```
export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU train.py
```

The multi-node is automatically handled by SLURM.

## References

Please cite [[1]](https://arxiv.org/abs/1901.07291) if you found the resources in this repository useful.

### Cross-lingual Language Model Pretraining

[1] G. Lample *, A. Conneau * [*Cross-lingual Language Model Pretraining*](https://arxiv.org/abs/1901.07291)

\* Equal contribution. Order has been determined with a coin flip.

```
@article{lample2019cross,
  title={Cross-lingual Language Model Pretraining},
  author={Lample, Guillaume and Conneau, Alexis},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

### XNLI: Evaluating Cross-lingual Sentence Representations

[2] A. Conneau, G. Lample, R. Rinott, A. Williams, S. R. Bowman, H. Schwenk, V. Stoyanov [*XNLI: Evaluating Cross-lingual Sentence Representations*](https://arxiv.org/abs/1809.05053)

```
@inproceedings{conneau2018xnli,
  title={XNLI: Evaluating Cross-lingual Sentence Representations},
  author={Conneau, Alexis and Lample, Guillaume and Rinott, Ruty and Williams, Adina and Bowman, Samuel R and Schwenk, Holger and Stoyanov, Veselin},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2018}
}
```

### Phrase-Based \& Neural Unsupervised Machine Translation

[3] G. Lample, M. Ott, A. Conneau, L. Denoyer, MA. Ranzato [*Phrase-Based & Neural Unsupervised Machine Translation*](https://arxiv.org/abs/1804.07755)

```
@inproceedings{lample2018phrase,
  title={Phrase-Based \& Neural Unsupervised Machine Translation},
  author={Lample, Guillaume and Ott, Myle and Conneau, Alexis and Denoyer, Ludovic and Ranzato, Marc'Aurelio},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2018}
}
```

### Large Memory Layers with Product Keys

[4] G. Lample, A. Sablayrolles, MA. Ranzato, L. Denoyer, H. Jégou [*Large Memory Layers with Product Keys*](https://arxiv.org/abs/1907.05242)

```
@article{lample2019large,
  title={Large Memory Layers with Product Keys},
  author={Lample, Guillaume and Sablayrolles, Alexandre and Ranzato, Marc'Aurelio and Denoyer, Ludovic and J{\'e}gou, Herv{\'e}},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

### Unsupervised Cross-lingual Representation Learning at Scale

[5] A. Conneau *, K. Khandelwal *, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzman, E. Grave, M. Ott, L. Zettlemoyer, V. Stoyanov [*Unsupervised Cross-lingual Representation Learning at Scale*](https://arxiv.org/abs/1911.02116)

\* Equal contribution

```
@article{conneau2019unsupervised,
  title={Unsupervised Cross-lingual Representation Learning at Scale},
  author={Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1911.02116},
  year={2019}
}
```

## License

See the [LICENSE](LICENSE) file for more details.

