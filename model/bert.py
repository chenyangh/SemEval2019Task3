import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class BERT_classifer(PreTrainedBertModel):
    """
    A Hierarchical LSTM with for 3 turns dialogue
    """
    def __init__(self, config):
        super(BERT_classifer, self).__init__(config)

        self.num_labels = NUM_EMO
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.apply(self.init_bert_weights)
        self.bert_out_dim = None
        self.out2label = None
        self.out2binary = None
        self.out2emo = None

    def add_output_layer(self, BERT_MODEL, NUM_EMO):
        if BERT_MODEL == 'bert-large-uncased':
            self.bert_out_dim = 1024
        elif BERT_MODEL == 'bert-base-uncased':
            self.bert_out_dim = 768
        self.out2label = nn.Linear(self.bert_out_dim, NUM_EMO)
        self.out2binary = nn.Linear(self.bert_out_dim, 2)
        self.out2emo = nn.Linear(self.bert_out_dim, NUM_EMO - 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        _, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)

        out1 = self.out2label(pooled_output)
        out2 = self.out2binary(pooled_output)
        out3 = F.sigmoid(self.out2emo(pooled_output))

        return out1, out2, out3
