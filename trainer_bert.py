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
from model.bert import BERT_classifer
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from data.evaluate import load_dev_labels, get_metrics
import sys
import argparse
import random
from utils.focalloss import FocalLoss
from utils.tweet_processor import processing_pipeline
from copy import deepcopy

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('-folds', default=9, type=int,
                    help="num of folds")
parser.add_argument('-bs', default=128, type=int,
                    help="batch size")
parser.add_argument('-postname', default='', type=str,
                    help="name that will be added at the end of generated file")
parser.add_argument('-gamma', default=0.2, type=float,
                    help="the decay of the ")
parser.add_argument('-lr', default=5e-4, type=float,
                    help="learning rate")
parser.add_argument('-lbd1', default=0, type=float,
                    help="lambda1 is for MTL")
parser.add_argument('-lbd2', default=0, type=float,
                    help="lambda2 is for optimizing only the emotional labels")
parser.add_argument('-patience', default=1, type=int,
                    help="patience of early stopping")
parser.add_argument('-flat', default=1, type=float,
                    help="flatten para")
parser.add_argument('-focal', default=2, type=int,
                    help="gamma value for focal loss, default 2")
parser.add_argument('-w', default=2, type=int,
                    help="patience ")
parser.add_argument('-loss', default='ce', type=str,
                    help="ce or focal ")
parser.add_argument('-tokentype', default='True', type=str,
                    help="post name")
opt = parser.parse_args()

if opt.size == 'large':
    BERT_MODEL = 'bert-large-uncased'
elif opt.size == 'base':
    BERT_MODEL = 'bert-base-uncased'
else:
    raise ValueError

NUM_OF_FOLD = opt.folds
NUM_EMO = 4
learning_rate = opt.lr
MAX_EPOCH = 300
CONV_PAD_LEN = 3
SENT_PAD_LEN = opt.padlen
BATCH_SIZE = opt.bs
SENT_EMB_DIM = 300
CLIP = 0.888
FLAT = opt.flat
EARLY_STOP_PATIENCE = 1
LAMBDA1 = opt.lbd1
LAMBDA2 = opt.lbd2
EMOS = ['happy', 'angry', 'sad', 'others']
EMOS_DIC = {'happy': 0,
            'angry': 1,
            'sad': 2,
            'others': 3}
USE_TOKEN_TYPE = None
if opt.tokentype == 'False':
    USE_TOKEN_TYPE = False
elif opt.tokentype == 'True':
    USE_TOKEN_TYPE = True
else:
    raise ValueError('Token type is not recognised')

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)


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

        a = convers[0]
        b = convers[1]
        c = convers[2]

        a = processing_pipeline(a)
        b = processing_pipeline(b)
        c = processing_pipeline(c)

        a_len = len(a.split())
        b_len = len(b.split())
        c_len = len(c.split())

        data_list.append((a, a_len, b, b_len, c, c_len))
        if is_train:
            emo = tokens[CONV_PAD_LEN + 1].strip()
            target_list.append(EMOS_DIC[emo])

    if is_train:
        return data_list, target_list
    else:
        return data_list


class DataSet(Dataset):
    def __init__(self, data_list, target_list, sent_pad_len):

        self.sent_pad_len = sent_pad_len
        self.word2id = 0
        self.pad_int = 0

        # set max size for the purpose of testing

        # internal data
        self.tokens = []
        self.token_masks = []
        self.token_segments = []
        self.e_c = []
        self.e_c_binary = []
        self.e_c_emo = []

        self.num_empty_lines = 0
        # prepare dataset
        self.read_data(data_list, target_list)

    def read_data(self, data_list, target_list):

        assert len(data_list) == len(target_list)

        for X, y in zip(data_list, target_list):
            a, _, b, _, c, _ = X

            a = tokenizer.tokenize(a)
            b = tokenizer.tokenize(b)
            c = tokenizer.tokenize(c)

            a = ['[CLS]'] + a + ['[SEP]']
            b = b + ['[SEP]']
            c = c + ['[SEP]']

            a = tokenizer.convert_tokens_to_ids(a)
            b = tokenizer.convert_tokens_to_ids(b)
            c = tokenizer.convert_tokens_to_ids(c)

            a_len = len(a)
            b_len = len(b)
            c_len = len(c)

            combined_tokens = a + b + c

            token_seg = [0] * (a_len + b_len - 1) + [1] * (c_len+1)

            if len(combined_tokens) > self.sent_pad_len:
                combined_tokens = combined_tokens[:self.sent_pad_len]
                mask = [1] * self.sent_pad_len
                token_seg = token_seg[:self.sent_pad_len]
            else:
                combined_tokens = combined_tokens + [self.pad_int] * (self.sent_pad_len - len(combined_tokens))
                mask = [1] * (a_len + b_len + c_len) + [0] * (self.sent_pad_len - (a_len + b_len + c_len))
                token_seg = token_seg + [self.pad_int] * (self.sent_pad_len - len(token_seg))

            self.tokens.append(combined_tokens)

            self.token_masks.append(mask)

            self.token_segments.append(token_seg)

            self.e_c.append(int(y))

            self.e_c_binary.append(1 if int(y) == len(EMOS) - 1 else 0)

            e_c_emo = [0] * (len(EMOS) - 1)
            if int(y) < len(EMOS) - 1:  # i.e. only first three emotions
                e_c_emo[int(y)] = 1
            self.e_c_emo.append(e_c_emo)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return torch.LongTensor(self.tokens[idx]),\
               torch.LongTensor(self.token_masks[idx]),\
               torch.LongTensor(self.token_segments[idx]), \
               torch.LongTensor([self.e_c[idx]]), \
               torch.LongTensor([self.e_c_binary[idx]]), \
               torch.FloatTensor(self.e_c_emo[idx])


class TestDataSet(Dataset):
    def __init__(self, data_list, sent_pad_len):

        self.sent_pad_len = sent_pad_len
        self.word2id = 0
        self.pad_int = 0

        # internal data
        self.tokens = []
        self.token_masks = []
        self.token_segments = []
        self.e_c = []

        self.num_empty_lines = 0
        # prepare dataset

        self.read_data(data_list)

    def read_data(self, data_list):

        for X in data_list:
            a, _, b, _, c, _ = X

            a = tokenizer.tokenize(a)
            b = tokenizer.tokenize(b)
            c = tokenizer.tokenize(c)

            a = ['[CLS]'] + a
            b = b + ['[SEP]']
            c = c + ['[SEP]']

            a = tokenizer.convert_tokens_to_ids(a)
            b = tokenizer.convert_tokens_to_ids(b)
            c = tokenizer.convert_tokens_to_ids(c)

            a_len = len(a)
            b_len = len(b)
            c_len = len(c)

            combined_tokens = a + b + c

            token_seg = [0] * (a_len + b_len - 1) + [1] * (c_len+1)

            if len(combined_tokens) > self.sent_pad_len:
                combined_tokens = combined_tokens[:self.sent_pad_len]
                mask = [1] * self.sent_pad_len
                token_seg = token_seg[:self.sent_pad_len]
            else:
                combined_tokens = combined_tokens + [self.pad_int] * (self.sent_pad_len - len(combined_tokens))
                mask = [1] * (a_len + b_len + c_len) + [0] * (self.sent_pad_len - (a_len + b_len + c_len))
                token_seg = token_seg + [self.pad_int] * (self.sent_pad_len - len(token_seg))

            self.tokens.append(combined_tokens)

            self.token_masks.append(mask)

            self.token_segments.append(token_seg)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return torch.LongTensor(self.tokens[idx]), \
               torch.LongTensor(self.token_masks[idx]), \
               torch.LongTensor(self.token_segments[idx])


def main():
    # load data
    path = 'data/train.txt'
    data_list, target_list = load_data_context(path)

    # build vocab

    X = data_list
    y = target_list
    y = np.array(y)

    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)

    # skf.get_n_splits(X, y)
    # train dev split
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=NUM_OF_FOLD, random_state=0)
    all_fold_results = []
    real_test_results = []
    # dev set
    dev_file = 'data/dev.txt'
    dev_data_list, dev_target_list = load_data_context(data_path=dev_file)

    # test set
    gold_test_file = 'data/test.txt'
    gold_test_data_list, gold_test_target_list = load_data_context(data_path=gold_test_file)

    gold_dev_data_set = DataSet(dev_data_list, dev_target_list, SENT_PAD_LEN)
    gold_dev_data_loader = DataLoader(gold_dev_data_set, batch_size=BATCH_SIZE, shuffle=False)
    print("Size of test data", len(gold_dev_data_set))

    gold_test_data_set = DataSet(gold_test_data_list, gold_test_target_list, SENT_PAD_LEN)
    gold_test_data_loader = DataLoader(gold_test_data_set, batch_size=BATCH_SIZE, shuffle=False)
    print("Size of test data", len(gold_test_data_set))

    test_file = 'data/testwithoutlabels.txt'
    test_data_list = load_data_context(data_path=test_file, is_train=False)
    test_data_set = TestDataSet(test_data_list, SENT_PAD_LEN)
    test_data_loader = DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=False)

    def one_fold(num_fold, train_index, dev_index):
        print("Training on fold:", num_fold)
        X_train, X_dev = [X[i] for i in train_index], [X[i] for i in dev_index]
        y_train, y_dev = y[train_index], y[dev_index]

        # construct data loader
        train_data_set = DataSet(X_train, y_train, SENT_PAD_LEN)
        train_data_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)

        dev_data_set = DataSet(X_dev, y_dev, SENT_PAD_LEN)
        dev_data_loader = DataLoader(dev_data_set, batch_size=BATCH_SIZE, shuffle=False)
        gradient_accumulation_steps = 1
        num_train_steps = int(
            len(train_data_set) / BATCH_SIZE / gradient_accumulation_steps * MAX_EPOCH)

        pred_list_test_best = None
        final_pred_best = None
        # This is to prevent model diverge, once happen, retrain
        while True:
            is_diverged = False
            model = BERT_classifer.from_pretrained(BERT_MODEL)
            model.add_output_layer(BERT_MODEL, NUM_EMO)
            model = nn.DataParallel(model)
            model.cuda()

            # BERT optimizer
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]

            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=learning_rate,
                                 warmup=0.1,
                                 t_total=num_train_steps)

            if opt.w == 1:
                weight_list = [0.3, 0.3, 0.3, 1.7]
                weight_list_binary = [2 - weight_list[-1], weight_list[-1]]
            elif opt.w == 2:
                weight_list = [0.3198680179, 0.246494733, 0.2484349259, 1.74527696]
                weight_list_binary = [2 - weight_list[-1], weight_list[-1]]

            weight_list = [x**FLAT for x in weight_list]
            weight_label = torch.Tensor(weight_list).cuda()

            weight_list_binary = [x**FLAT for x in weight_list_binary]
            weight_binary = torch.Tensor(weight_list_binary).cuda()
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
            final_pred_best = None
            final_pred_list_test = None
            pred_list_test = None
            for num_epoch in range(MAX_EPOCH):
                print('Begin training epoch:', num_epoch)
                sys.stdout.flush()
                train_loss = 0
                model.train()
                for i, (tokens, masks, segments, e_c, e_c_binary, e_c_emo) in tqdm(enumerate(train_data_loader),
                                                              total=len(train_data_set)/BATCH_SIZE):
                    optimizer.zero_grad()

                    if USE_TOKEN_TYPE:
                        pred, pred2, pred3 = model(tokens.cuda(), masks.cuda(), segments.cuda())
                    else:
                        pred, pred2, pred3 = model(tokens.cuda(), masks.cuda())

                    loss_label = loss_criterion(pred, e_c.view(-1).cuda()).cuda()
                    loss_label = torch.matmul(torch.gather(weight_label, 0, e_c.view(-1).cuda()), loss_label) / \
                                 e_c.view(-1).shape[0]

                    loss_binary = loss_criterion_binary(pred2, e_c_binary.view(-1).cuda()).cuda()
                    loss_binary = torch.matmul(torch.gather(weight_binary, 0, e_c_binary.view(-1).cuda()),
                                               loss_binary) / e_c.view(-1).shape[0]

                    loss_emo = loss_criterion_emo_only(pred3, e_c_emo.cuda())

                    loss = (loss_label + LAMBDA1 * loss_binary + LAMBDA2 * loss_emo) / float(1 + LAMBDA1 + LAMBDA2)

                    # training trilogy
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                    optimizer.step()

                    train_loss += loss.data.cpu().numpy() * tokens.shape[0]

                    del loss, pred

                # Evaluate
                model.eval()
                dev_loss = 0
                # pred_list = []
                # gold_list = []
                for i, (tokens, masks, segments, e_c, e_c_binary, e_c_emo) in enumerate(dev_data_loader):
                    with torch.no_grad():
                        if USE_TOKEN_TYPE:
                            pred, pred2, pred3 = model(tokens.cuda(), masks.cuda(), segments.cuda())
                        else:
                            pred, pred2, pred3 = model(tokens.cuda(), masks.cuda())

                        loss_label = loss_criterion(pred, e_c.view(-1).cuda()).cuda()
                        loss_label = torch.matmul(torch.gather(weight_label, 0, e_c.view(-1).cuda()), loss_label) / \
                                     e_c.view(-1).shape[0]

                        loss_binary = loss_criterion_binary(pred2, e_c_binary.view(-1).cuda()).cuda()
                        loss_binary = torch.matmul(torch.gather(weight_binary, 0, e_c_binary.view(-1).cuda()),
                                                   loss_binary) / e_c.view(-1).shape[0]

                        loss_emo = loss_criterion_emo_only(pred3, e_c_emo.cuda())

                        loss = (loss_label + LAMBDA1 * loss_binary + LAMBDA2 * loss_emo) / float(1 + LAMBDA1 + LAMBDA2)

                        dev_loss += loss.data.cpu().numpy() * tokens.shape[0]

                        # pred_list.append(pred.data.cpu().numpy())
                        # gold_list.append(e_c.numpy())
                        del pred, loss

                # pred_list = np.argmax(np.concatenate(pred_list, axis=0), axis=1)
                # gold_list = np.concatenate(gold_list, axis=0)
                print('Training loss:', train_loss / len(train_data_set), end='\t')
                print('Dev loss:', dev_loss / len(dev_data_set))
                # print(classification_report(gold_list, pred_list, target_names=EMOS))
                # get_metrics(pred_list, gold_list)
                # checking diverge
                if dev_loss/len(dev_data_set) > 1.3 and num_epoch > 4:
                    print("Model diverged, retry")
                    is_diverged = True
                    break

                if es.step(dev_loss):  # overfitting
                    print('overfitting, loading best model ...')
                    if num_epoch == 1:
                        is_diverged = True
                        final_pred_best = deepcopy(final_pred_list_test)
                        pred_list_test_best = deepcopy(pred_list_test)
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

                print('Gold Dev ...')
                pred_list_test = []
                model.eval()
                for i, (tokens, masks, segments, e_c, e_c_binary, e_c_emo) in enumerate(gold_dev_data_loader):
                    with torch.no_grad():
                        if USE_TOKEN_TYPE:
                            pred, _, _ = model(tokens.cuda(), masks.cuda(), segments.cuda())
                        else:
                            pred, _, _ = model(tokens.cuda(), masks.cuda())
                        pred_list_test.append(pred.data.cpu().numpy())

                pred_list_test = np.argmax(np.concatenate(pred_list_test, axis=0), axis=1)
                # get_metrics(load_dev_labels('data/dev.txt'), pred_list_test)

                print('Gold Test ...')
                final_pred_list_test = []
                model.eval()
                for i, (tokens, masks, segments, e_c, e_c_binary, e_c_emo) in enumerate(gold_test_data_loader):
                    with torch.no_grad():
                        if USE_TOKEN_TYPE:
                            pred, _, _ = model(tokens.cuda(), masks.cuda(), segments.cuda())
                        else:
                            pred, _, _ = model(tokens.cuda(), masks.cuda())
                        final_pred_list_test.append(pred.data.cpu().numpy())

                final_pred_list_test = np.argmax(np.concatenate(final_pred_list_test, axis=0), axis=1)
                # get_metrics(load_dev_labels('data/test.txt'), final_pred_list_test)

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
    f_out = open('test_bert_mtl' + opt.postname + '.txt', 'w')

    data_lines = f_in.readlines()
    for idx, text in enumerate(data_lines):
        if idx == 0:
            f_out.write(text.strip() + '\tlabel\n')
        else:
            f_out.write(text.strip() + '\t' + EMOS[mj[idx - 1]] + '\n')

    f_in.close()
    f_out.close()
    print('Final testing')

main()
