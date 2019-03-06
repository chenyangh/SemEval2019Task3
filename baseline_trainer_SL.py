import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.early_stopping import EarlyStopping
import numpy as np
import copy
from tqdm import tqdm
from tasks.Task3.baseline_model_SL import HierarchicalPredictor, NUM_EMO
from sklearn.metrics import classification_report
from tasks.Task3.data.evaluate import load_dev_labels
import pickle as pkl
import emoji
import nltk
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import sys
from allennlp.modules.elmo import Elmo, batch_to_ids
from copy import deepcopy
import argparse
import random
from utils.focalloss import FocalLoss
import json
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from emoji import UNICODE_EMOJI
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('-folds', default=9, type=int,
                    help="num of folds")
parser.add_argument('-bs', default=128, type=int,
                    help="batch size")
parser.add_argument('-postname', default='', type=str,
                    help="post name")
parser.add_argument('-gamma', default=0.2, type=float,
                    help="post name")
parser.add_argument('-lr', default=5e-4, type=float,
                    help="post name")
parser.add_argument('-lbd1', default=0, type=float,
                    help="lambda1 is for MTL")
parser.add_argument('-lbd2', default=0, type=float,
                    help="lambda2 is for optimizing only the emotional labels")
parser.add_argument('-patience', default=1, type=int,
                    help="patience of early stopping")
parser.add_argument('-flat', default=1, type=float,
                    help="flatten para")
parser.add_argument('-focal', default=2, type=int,
                    help="patience ")
parser.add_argument('-w', default=10, type=int,
                    help="patience ")
parser.add_argument('-loss', default='ce', type=str,
                    help="ce or focal ")
parser.add_argument('-dim', default=1500, type=int,
                    help="post name")
opt = parser.parse_args()


NUM_OF_FOLD = opt.folds
learning_rate = opt.lr
MAX_EPOCH = 200
SENT_PAD_LEN = 30
EMOJ_SENT_PAD_LEN = 30
CONV_PAD_LEN = 3
FILL_VOCAB = True
BATCH_SIZE = opt.bs
SENT_EMB_DIM = 300
SENT_HIDDEN_SIZE = opt.dim
CLIP = 0.888
EARLY_STOP_PATIENCE = opt.patience
LAMBDA1 = opt.lbd1
LAMBDA2 = opt.lbd2
FLAT = opt.flat
EMOS = ['happy', 'angry', 'sad', 'others']
EMOS_DIC = {'happy': 0,
            'angry': 1,
            'sad': 2,
            'others': 3}

# DEV_DIST = [0.05154264973, 0.05444646098, 0.04537205082, 0.8486388385]
# fix random seeds to ensure replicability
RANDOM_SEED = 0

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# FAST_EMB_PATH = '/home/chenyang/data/feature/wiki-news-300d-1M.vec'
# FAST_EMB_PATH = '/remote/eureka1/chuang8/wiki-news-300d-1M.vec'
# GLOVE_EMB_PATH = '/home/chenyang/PycharmProjects/InsincereQuestions/' \
#                  'input/embeddings/glove.840B.300d/glove.840B.300d.txt'
GLOVE_EMB_PATH = '/remote/eureka1/chuang8/glove.840B.300d.txt'

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
# weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'

elmo = Elmo(options_file, weight_file, 2, dropout=0).cuda()
elmo.eval()

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
emoji_st = SentenceTokenizer(vocabulary, EMOJ_SENT_PAD_LEN)

# '/remote/eureka1/chuang8/wiki-news-300d-1M.vec'

# NUM_EMO = 7 defined in sa_lstm. # TODO: need to refactor

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", #  "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def remove_dup_emoji(sent):
    ret = []
    for word in sent.split():
        emo_found = [char for char in word if char in UNICODE_EMOJI]
        if len(emo_found) > 1:
            word = emo_found[0]
        ret.append(word)
    return ' '.join(ret)


def remove_underscope_for_emoji(text):
    tokens = text.split()
    ret_list = []
    for token in tokens:
        if len(token) > 3 and '_' in token:
            token = token.replace('_', ' ')

        if token[0] == '<' and token[-1] == '>':
            token = token[1:-1]

        ret_list.append(token)
    return ' '.join(ret_list)


def processing_pipelie(text):
    text = text.lower().strip()
    # text = remove_dup_emoji(text)
    text = ' '.join(text_processor.pre_process_doc(text))
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = remove_underscope_for_emoji(text)
    return text


def load_data_context(data_path='data/train.txt', is_train=True):
    # data_path = 'data/train.txt'

    data_list = []
    target_list = []
    f_data = open(data_path, 'r')
    data_lines = f_data.readlines()
    f_data.close()

    for i, text in enumerate(data_lines):
        # skip the first line
        if i == 0:
            continue

        tokens = text.split('\t')

        convers = tokens[1:CONV_PAD_LEN+1]

        # normal preprocessing
        raw_a = convers[0]
        raw_b = convers[1]
        raw_c = convers[2]

        a = processing_pipelie(raw_a)
        b = processing_pipelie(raw_b)
        c = processing_pipelie(raw_c)

        data_list.append(a + ' ' + b + ' ' + c)
        if is_train:
            emo = tokens[CONV_PAD_LEN + 1].strip()
            target_list.append(EMOS_DIC[emo])

    if is_train:
        return data_list, target_list
    else:
        return data_list


def build_vocab(data_list_list, vocab_size, fill_vocab=False):

    all_str_list = []
    for data_list in data_list_list:
        all_str_list.extend(data_list)

    word_count = {}
    word2id = {}
    id2word = {}
    for tokens in all_str_list:
        for word in tokens.split():
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    word_list = [x for x, _ in sorted(word_count.items(), key=lambda v: v[1], reverse=True)]
    print('found', len(word_count), 'words')

    if len(word_count) < vocab_size:
        raise Exception('Vocab less than requested!!!')

    # add <pad> first
    word2id['<pad>'] = 0
    id2word[0] = '<pad>'

    word2id['<unk>'] = 1
    id2word[1] = '<unk>'
    word2id['<empty>'] = 2
    id2word[2] = '<empty>'

    n = len(word2id)
    if not fill_vocab:
        word_list = word_list[:vocab_size - n]

    for word in word_list:
        word2id[word] = n
        id2word[n] = word
        n += 1

    if fill_vocab:
        print('filling vocab to', len(id2word))
        return word2id, id2word, len(id2word)
    return word2id, id2word, len(word2id)


class TrainDataSet(Dataset):
    def __init__(self, data_list, target_list, conv_pad_len, sent_pad_len, word2id, max_size=None, use_unk=False):

        self.sent_pad_len = sent_pad_len
        self.conv_pad_len = conv_pad_len
        self.word2id = word2id
        self.pad_int = word2id['<pad>']

        self.use_unk = use_unk

        # set max size for the purpose of testing
        if max_size is not None:
            self.data = self.data[:max_size]
            self.target = self.target[:max_size]

        # internal data
        self.a = []
        self.a_len = []
        self.emoji_a = []

        self.e_c = []
        self.e_c_binary = []
        self.e_c_emo = []
        self.num_empty_lines = 0

        self.weights = []
        # prepare dataset
        self.read_data(data_list, target_list)

    def sent_to_ids(self, text):
        tokens = text.split()
        if self.use_unk:
            tmp = [self.word2id[x] if x in self.word2id else self.word2id['<unk>'] for x in tokens]
        else:
            tmp = [self.word2id[x] for x in tokens if x in self.word2id]
        if len(tmp) == 0:
            tmp = [self.word2id['<empty>']]
            self.num_empty_lines += 1

        # PADDING
        if len(tmp) > self.sent_pad_len:
            tmp = tmp[: self.sent_pad_len]
        text_len = len(tmp)

        tmp = tmp + [self.pad_int] * (self.sent_pad_len - len(tmp))

        return tmp, text_len

    def read_data(self, data_list, target_list):
        assert len(data_list) == len(target_list)

        for X, y in zip(data_list, target_list):
            clean_a = X

            a, a_len = self.sent_to_ids(clean_a)

            self.a.append(a)

            self.a_len.append(a_len)

            self.emoji_a.append(emoji_st.tokenize_sentences([clean_a])[0].reshape((-1)).astype(np.int64))

            self.e_c.append(int(y))
            self.e_c_binary.append(1 if int(y) == len(EMOS) - 1 else 0)

            e_c_emo = [0] * (len(EMOS) - 1)
            if int(y) < len(EMOS) - 1:  # i.e. only first three emotions
                e_c_emo[int(y)] = 1
            self.e_c_emo.append(e_c_emo)

        print('num of empty lines,', self.num_empty_lines)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return torch.LongTensor(self.a[idx]), torch.LongTensor([self.a_len[idx]]), \
               torch.LongTensor(self.emoji_a[idx]), \
               torch.LongTensor([self.e_c[idx]]), torch.LongTensor([self.e_c_binary[idx]]), \
               torch.FloatTensor(self.e_c_emo[idx])


class TestDataSet(Dataset):
    def __init__(self, data_list, conv_pad_len, sent_pad_len, word2id, id2word, use_unk=False):

        self.sent_pad_len = sent_pad_len
        self.conv_pad_len = conv_pad_len
        self.word2id = word2id
        self.pad_int = word2id['<pad>']

        self.use_unk = use_unk

        # internal data
        self.a = []
        self.a_len = []
        self.emoji_a = []

        self.num_empty_lines = 0
        # prepare dataset
        self.ex_word2id = copy.deepcopy(word2id)
        self.ex_id2word = copy.deepcopy(id2word)
        self.unk_words_idx = set()
        self.read_data(data_list)

    def sent_to_ids(self, text):
        tokens = text.split()
        if self.use_unk:
            tmp = [self.word2id[x] if x in self.word2id else self.word2id['<unk>'] for x in tokens]
        else:
            tmp = [self.word2id[x] for x in tokens if x in self.word2id]
        if len(tmp) == 0:
            tmp = [self.word2id['<empty>']]
            self.num_empty_lines += 1

        # PADDING
        if len(tmp) > self.sent_pad_len:
            tmp = tmp[: self.sent_pad_len]
        text_len = len(tmp)

        tmp = tmp + [self.pad_int] * (self.sent_pad_len - len(tmp))

        return tmp, text_len

    def read_data(self, data_list):
        for X in data_list:
            clean_a= X

            a, a_len = self.sent_to_ids(clean_a)

            self.a.append(a)

            self.a_len.append(a_len)

            self.emoji_a.append(emoji_st.tokenize_sentences([clean_a])[0].reshape((-1)).astype(np.int64))


        print('num of empty lines,', self.num_empty_lines)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return torch.LongTensor(self.a[idx]), torch.LongTensor([self.a_len[idx]]), \
               torch.LongTensor(self.emoji_a[idx])


def to_categorical(vec):
    to_ret = np.zeros((vec.shape[0], NUM_EMO))
    for idx, val in enumerate(vec):
        to_ret[idx, val] = 1
    return to_ret


def get_metrics(ground, predictions):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref -
        https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions)
    ground = to_categorical(ground)
    truePositives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    #  Macro level calculation
    macroPrecision = 0
    macroRecall = 0
    f1_list = []
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(NUM_EMO-1):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(f1)
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (EMOS[c], precision, recall, f1))

    print('Direct average of macro F1s are :------> ', (f1_list[0] + f1_list[1] + f1_list[2]) / 3)
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision + macroRecall) \
        if (macroPrecision + macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
    macroPrecision, macroRecall, macroF1))

    # Micro level calculation
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d"
          % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall)\
        if (microPrecision + microRecall) > 0 else 0

    # predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (
    accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def build_embedding(id2word, fname, num_of_vocab):
    """
    :param id2word, fname:
    :return:
    """
    import io

    def load_vectors(fname):
        print("Loading Glove Model")
        f = open(fname, 'r', encoding='utf8')
        model = {}
        for line in tqdm(f.readlines(), total=2196017):
            values = line.split(' ')
            word = values[0]
            try:
                embedding = np.array(values[1:], dtype=np.float32)
                model[word] = embedding
            except ValueError:
                print(len(values), values[0])

        print("Done.", len(model), " words loaded!")
        f.close()
        return model

    def get_emb(emb_dict, vocab_size, embedding_dim):
        # emb_dict = load_vectors(fname)
        all_embs = np.stack(emb_dict.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()

        emb = np.random.normal(emb_mean, emb_std, (vocab_size, embedding_dim))

        # emb = np.zeros((vocab_size, embedding_dim))
        num_found = 0
        print('loading glove')
        for idx in tqdm(range(vocab_size)):
            word = id2word[idx]
            if word == '<pad>' or word == '<unk>':
                emb[idx] = np.zeros([embedding_dim])
            elif word in emb_dict:
                emb[idx] = emb_dict[word]
                num_found += 1

        return emb, num_found

    pkl_path = fname + '.pkl'
    if not os.path.isfile(pkl_path):
        print('creating pkl file for the emb text file')
        emb_dict = load_vectors(fname)
        with open(pkl_path, 'wb') as f:
            pkl.dump(emb_dict, f)
    else:
        print('loading pkl file')
        with open(pkl_path, 'rb') as f:
            emb_dict = pkl.load(f)
        print('loading finished')

    emb, num_found = get_emb(emb_dict, num_of_vocab, SENT_EMB_DIM)

    print(num_found, 'of', num_of_vocab, 'found', 'coverage', num_found/num_of_vocab)

    return emb


def main():
    num_of_vocab = 10000

    # load data
    train_file = 'data/train.txt'
    data_list, target_list = load_data_context(data_path=train_file)

    # dev set
    dev_file = 'data/dev.txt'
    dev_data_list, dev_target_list = load_data_context(data_path=dev_file)

    # test set
    test_file = 'data/test.txt'
    test_data_list, test_target_list = load_data_context(data_path=test_file)

    # load final test data
    final_test_file = 'data/testwithoutlabels.txt'
    final_test_data_list = load_data_context(data_path=final_test_file, is_train=False)

    # build vocab
    word2id, id2word, num_of_vocab = build_vocab([data_list, dev_data_list, test_data_list], num_of_vocab,
                                                 FILL_VOCAB)
    emb = build_embedding(id2word, GLOVE_EMB_PATH, num_of_vocab)

    gold_dev_data_set = TestDataSet(dev_data_list, CONV_PAD_LEN, SENT_PAD_LEN, word2id, id2word, use_unk=False)
    gold_dev_data_loader = DataLoader(gold_dev_data_set, batch_size=BATCH_SIZE, shuffle=False)
    print("Size of test data", len(gold_dev_data_set))

    test_data_set = TestDataSet(test_data_list, CONV_PAD_LEN, SENT_PAD_LEN, word2id, id2word, use_unk=False)
    test_data_loader = DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=False)
    print("Size of test data", len(test_data_set))
    # ex_id2word, unk_words_idx = test_data_set.get_ex_id2word_unk_words()

    final_test_data_set = TestDataSet(final_test_data_list, CONV_PAD_LEN, SENT_PAD_LEN, word2id, id2word, use_unk=False)
    final_test_data_loader = DataLoader(final_test_data_set, batch_size=BATCH_SIZE, shuffle=False)
    print("Size of final test data", len(final_test_data_set))

    # final_ex_id2word, _ = final_test_data_set.get_ex_id2word_unk_words()
    def glove_tokenizer(ids, __id2word):
        return [__id2word[int(x)] for x in ids if x != 0]

    def elmo_encode(data, __id2word=id2word):
        data_text = [glove_tokenizer(x, __id2word) for x in data]
        with torch.no_grad():
            character_ids = batch_to_ids(data_text).cuda()
            elmo_emb = elmo(character_ids)['elmo_representations']
            elmo_emb = (elmo_emb[0] + elmo_emb[1]) / 2  # avg of two layers
        return elmo_emb.cuda()

    X = data_list
    y = target_list
    y = np.array(y)

    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)

    # train dev split
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=NUM_OF_FOLD, random_state=0)
    all_fold_results = []
    real_test_results = []

    def one_fold(num_fold, train_index, dev_index):
        print("Training on fold:", num_fold)
        X_train, X_dev = [X[i] for i in train_index], [X[i] for i in dev_index]
        y_train, y_dev = y[train_index], y[dev_index]

        # construct data loader
        train_data_set = TrainDataSet(X_train, y_train, CONV_PAD_LEN, SENT_PAD_LEN, word2id, use_unk=True)

        dev_data_set = TrainDataSet(X_dev, y_dev, CONV_PAD_LEN, SENT_PAD_LEN, word2id, use_unk=True)
        dev_data_loader = DataLoader(dev_data_set, batch_size=BATCH_SIZE, shuffle=False)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        pred_list_test_best = None
        final_pred_best = None
        # This is to prevent model diverge, once happen, retrain
        while True:
            is_diverged = False
            model = HierarchicalPredictor(SENT_EMB_DIM, SENT_HIDDEN_SIZE, num_of_vocab, USE_ELMO=True, ADD_LINEAR=False)
            model.load_embedding(emb)
            model.cuda()
            # model = nn.DataParallel(model)
            # model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True) #
            # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)

            if opt.w == 1:
                weight_list = [0.24, 0.24, 0.24, 1.76]
                weight_list_binary = [0.24, 1.76]
            elif opt.w == 2:
                weight_list = [0.3, 0.3, 0.3, 1.7]
                weight_list_binary = [0.3, 1.7]
            elif opt.w == 3:
                weight_list = [0.27, 0.27, 0.27, 1.73]
                weight_list_binary = [0.27, 1.73]
            elif opt.w == 4:
                weight_list = [0.2, 0.2, 0.2, 1.8]
                weight_list_binary = [0.2, 1.8]
            elif opt.w == 5:
                weight_list = [0.35, 0.35, 0.35, 1.65]
                weight_list_binary = [0.35, 1.65]
            elif opt.w == 6:
                weight_list = [0.4, 0.4, 0.4, 1.6]
                weight_list_binary = [0.4, 1.6]
            elif opt.w == 7:
                weight_list =[0.5, 0.5, 0.5, 1.5]
                weight_list_binary = [0.5, 1.5]
            elif opt.w == 8:
                weight_list = [1, 1, 1, 1]
                weight_list_binary = [1, 1]
            elif opt.w == 9:
                weight_list = [0.16, 0.16, 0.16, 1.84]
                weight_list_binary = [0.16, 1.84]
            elif opt.w == 10:
                weight_list = [0.3554089088, 0.2738830367, 0.2760388065, 1.715012042]
                weight_list_binary = [2 - weight_list[-1], weight_list[-1]]
            elif opt.w == 11:
                weight_list = [0.3198680179, 0.246494733, 0.2484349259, 1.74527696]
                weight_list_binary = [2 - weight_list[-1], weight_list[-1]]
            weight_list = [x**FLAT for x in weight_list]
            weight_label = torch.Tensor(weight_list).cuda()

            weight_list_binary = [x**FLAT for x in weight_list_binary]
            weight_binary = torch.Tensor(weight_list_binary).cuda()
            print('classification reweight: ', weight_list)
            print('binary loss reweight = weight_list_binary', weight_list_binary)
            # loss_criterion_binary = nn.CrossEntropyLoss(weight=weight_list_binary)  #
            if opt.loss == 'focal':
                loss_criterion = FocalLoss(gamma=opt.focal, reduce=False)
                loss_criterion_binary = FocalLoss(gamma=opt.focal, reduce=False)  #
            elif opt.loss == 'ce':
                loss_criterion = nn.CrossEntropyLoss(reduce=False)
                loss_criterion_binary = nn.CrossEntropyLoss(reduce=False)  #

            loss_criterion_emo_only = nn.MSELoss()

            # es = EarlyStopping(min_delta=0.005, patience=EARLY_STOP_PATIENCE)
            es = EarlyStopping(patience=EARLY_STOP_PATIENCE)
            # best_model = None
            final_pred_list_test = None
            pred_list_test = None
            for num_epoch in range(MAX_EPOCH):
                # to ensure shuffle at ever epoch
                train_data_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)

                print('Begin training epoch:', num_epoch, end='...\t')
                sys.stdout.flush()

                # stepping scheduler
                scheduler.step(num_epoch)
                print('Current learning rate', scheduler.get_lr())

                train_loss = 0
                model.train()
                for i, (a, a_len, emoji_a, e_c, e_c_binary, e_c_emo) \
                        in tqdm(enumerate(train_data_loader), total=len(train_data_set)/BATCH_SIZE):
                    optimizer.zero_grad()
                    elmo_a = elmo_encode(a)

                    pred, pred2, pred3 = model(a.cuda(), a_len, emoji_a.cuda(), elmo_a)

                    loss_label = loss_criterion(pred, e_c.view(-1).cuda()).cuda()
                    loss_label = torch.matmul(torch.gather(weight_label, 0, e_c.view(-1).cuda()), loss_label) / \
                                 e_c.view(-1).shape[0]

                    loss_binary = loss_criterion_binary(pred2, e_c_binary.view(-1).cuda()).cuda()
                    loss_binary = torch.matmul(torch.gather(weight_binary, 0, e_c_binary.view(-1).cuda()),
                                               loss_binary) / e_c.view(-1).shape[0]

                    loss_emo = loss_criterion_emo_only(pred3, e_c_emo.cuda())

                    loss = (loss_label + LAMBDA1 * loss_binary + LAMBDA2 * loss_emo) / float(1 + LAMBDA1 + LAMBDA2)

                    # loss = torch.matmul(torch.gather(weight, 0, trg.view(-1).cuda()), loss) / trg.view(-1).shape[0]

                    # training trilogy
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                    optimizer.step()

                    train_loss += loss.data.cpu().numpy() * a.shape[0]
                    del pred, loss, elmo_a, e_c_emo, loss_binary, loss_label, loss_emo

                # Evaluate
                model.eval()
                dev_loss = 0
                # pred_list = []
                # gold_list = []
                for i, (a, a_len, emoji_a, e_c, e_c_binary, e_c_emo) \
                        in enumerate(dev_data_loader):
                    with torch.no_grad():
                        elmo_a = elmo_encode(a)

                        pred, pred2, pred3 = model(a.cuda(), a_len, emoji_a.cuda(), elmo_a)

                        loss_label = loss_criterion(pred, e_c.view(-1).cuda()).cuda()
                        loss_label = torch.matmul(torch.gather(weight_label, 0, e_c.view(-1).cuda()), loss_label) / e_c.view(-1).shape[0]

                        loss_binary = loss_criterion_binary(pred2, e_c_binary.view(-1).cuda()).cuda()
                        loss_binary = torch.matmul(torch.gather(weight_binary, 0, e_c_binary.view(-1).cuda()), loss_binary) / e_c.view(-1).shape[0]

                        loss_emo = loss_criterion_emo_only(pred3, e_c_emo.cuda())

                        loss = (loss_label + LAMBDA1 * loss_binary + LAMBDA2 * loss_emo) / float(1 + LAMBDA1 + LAMBDA2)

                        dev_loss += loss.data.cpu().numpy() * a.shape[0]

                        # pred_list.append(pred.data.cpu().numpy())
                        # gold_list.append(e_c.numpy())
                        del pred, loss, elmo_a,  e_c_emo, loss_binary, loss_label, loss_emo

                print('Training loss:', train_loss / len(train_data_set), end='\t')
                print('Dev loss:', dev_loss / len(dev_data_set))
                # print(classification_report(gold_list, pred_list, target_names=EMOS))
                # get_metrics(pred_list, gold_list)
                if dev_loss/len(dev_data_set) > 1.3 and num_epoch > 4:
                    print("Model diverged, retry")
                    is_diverged = True
                    break

                if es.step(dev_loss):  # overfitting
                    print('overfitting, loading best model ...')
                    break
                else:
                    if es.is_best():
                        print('saving best model ...')
                        if final_pred_best is not None:
                            del final_pred_best
                        final_pred_best = deepcopy(final_pred_list_test)
                        if pred_list_test_best is not None:
                            del pred_list_test_best
                        pred_list_test_best = deepcopy(pred_list_test)
                    else:
                        print('not best model, ignoring ...')
                        if final_pred_best is None:
                            final_pred_best = deepcopy(final_pred_list_test)
                        if pred_list_test_best is None:
                            pred_list_test_best = deepcopy(pred_list_test)

                # Gold Dev testing...
                print('Gold Dev testing....')
                pred_list_test = []
                model.eval()
                for i, (a, a_len, emoji_a) in enumerate(gold_dev_data_loader):
                    with torch.no_grad():
                        elmo_a = elmo_encode(a)  # , __id2word=ex_id2word

                        pred, _, _ = model(a.cuda(), a_len, emoji_a.cuda(), elmo_a)

                        pred_list_test.append(pred.data.cpu().numpy())
                    del elmo_a, a, pred
                pred_list_test = np.argmax(np.concatenate(pred_list_test, axis=0), axis=1)
                get_metrics(load_dev_labels('data/dev.txt'), pred_list_test)

                # Testing
                print('Gold test testing...')
                final_pred_list_test = []
                model.eval()
                for i, (a, a_len, emoji_a) in enumerate(test_data_loader):
                    with torch.no_grad():
                        elmo_a = elmo_encode(a)  # , __id2word=ex_id2word

                        pred, _, _ = model(a.cuda(), a_len, emoji_a.cuda(), elmo_a)

                        final_pred_list_test.append(pred.data.cpu().numpy())
                    del elmo_a, a, pred
                final_pred_list_test = np.argmax(np.concatenate(final_pred_list_test, axis=0), axis=1)
                get_metrics(load_dev_labels('data/test.txt'), final_pred_list_test)

            if is_diverged:
                print("Reinitialize model ...")
                del model
                continue

            all_fold_results.append(pred_list_test_best)
            real_test_results.append(final_pred_best)
            del model
            break

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # Training the folds
    for idx, (_train_index, _dev_index) in enumerate(skf.split(X, y)):
        print('Train size:', len(_train_index), 'Dev size:', len(_dev_index))
        one_fold(idx, _train_index, _dev_index)

    # Function of majority voting
    def find_majority(k):
        myMap = {}
        maximum = ('', 0)  # (occurring element, occurrences)
        for n in k:
            if n in myMap:
                myMap[n] += 1
            else:
                myMap[n] = 1

            # Keep track of maximum on the go
            if myMap[n] > maximum[1]: maximum = (n, myMap[n])

        return maximum

    all_fold_results = np.asarray(all_fold_results)

    mj_dev = []
    for col_num in range(all_fold_results.shape[1]):
        a_mj = find_majority(all_fold_results[:, col_num])
        mj_dev.append(a_mj[0])

    print('FINAL gold DEV RESULTS')
    get_metrics(load_dev_labels('data/dev.txt'), np.asarray(mj_dev))

    real_test_results = np.asarray(real_test_results)
    mj = []
    for col_num in range(real_test_results.shape[1]):
        a_mj = find_majority(real_test_results[:, col_num])
        mj.append(a_mj[0])

    print('FINAL TESTING RESULTS')
    get_metrics(load_dev_labels('data/test.txt'), np.asarray(mj))
    # MAKE SUBMISSION
    # WRITE TO FILE
    test_file = 'data/testwithoutlabels.txt'
    f_in = open(test_file, 'r')
    f_out = open('test_SL' + opt.postname + '.txt', 'w')

    data_lines = f_in.readlines()
    for idx, text in enumerate(data_lines):
        if idx == 0:
            f_out.write(text.strip() + '\tlabel\n')
        else:
            f_out.write(text.strip() + '\t' + EMOS[mj[idx-1]] + '\n')

    f_in.close()
    f_out.close()

    print('I am SL :) Final testing')


main()
