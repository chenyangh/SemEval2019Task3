import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from model.self_attention import SelfAttentive, AttentionOneParaPerChan
from model.torch_moji import TorchMoji

import pickle as pkl
import os
from tqdm import tqdm

NUM_EMO = 4


class HierarchicalPredictor(nn.Module):
    """
    A Hierarchical LSTM with for 3 turns dialogue
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, USE_ELMO, ADD_LINEAR):
        super(HierarchicalPredictor, self).__init__()
        self.SENT_LSTM_DIM = hidden_dim
        self.bidirectional = True
        self.add_linear = ADD_LINEAR

        self.sent_lstm_directions = 2 if self.bidirectional else 1
        self.cent_lstm_att_fn = AttentionOneParaPerChan
        self.ctx_lstm__att_fn = AttentionOneParaPerChan

        self.deepmoji_model = TorchMoji(nb_classes=None,
                                        nb_tokens=50000,
                                        embed_dropout_rate=0.2,
                                        final_dropout_rate=0.2)
        self.deepmoji_dim = 2304
        self.deepmoji_out = 300
        self.deepmoji2linear = nn.Linear(self.deepmoji_dim, self.deepmoji_out)

        self.elmo_dim = 1024

        self.num_layers = 2
        self.use_elmo = USE_ELMO
        if not self.use_elmo:
            self.elmo_dim = 0

        self.a_lstm = nn.LSTM(embedding_dim + self.elmo_dim, hidden_dim, num_layers=self.num_layers, batch_first=True,
                            bidirectional=self.bidirectional, dropout=0.2)
        self.a_self_attention = self.cent_lstm_att_fn(self.sent_lstm_directions*hidden_dim)
        # self.a_layer_norm = BertLayerNorm(hidden_dim*self.sent_lstm_directions)

        self.b_lstm = nn.LSTM(embedding_dim + self.elmo_dim, hidden_dim, num_layers=self.num_layers, batch_first=True,
                            bidirectional=self.bidirectional, dropout=0.2)
        self.b_self_attention = self.cent_lstm_att_fn(self.sent_lstm_directions*hidden_dim)
        # self.b_layer_norm = BertLayerNorm(hidden_dim*self.sent_lstm_directions)

        self.ctx_bidirectional = True
        self.ctx_lstm_dim = 800
        self.ctx_lstm_directions = 2 if self.ctx_bidirectional else 1

        if not self.add_linear:
            self.deepmoji_out = self.deepmoji_dim

        self.context_lstm = nn.LSTM(self.SENT_LSTM_DIM * self.sent_lstm_directions + self.deepmoji_out, self.ctx_lstm_dim,
                                    num_layers=2, batch_first=True, dropout=0.2, bidirectional=self.ctx_bidirectional)
        self.ctx_self_attention = self.ctx_lstm__att_fn(self.ctx_lstm_directions * self.ctx_lstm_dim)
        # self.ctx_layer_norm = BertLayerNorm(self.ctx_lstm_dim*self.ctx_lstm_directions)
        # self.ctx_pooler = BertPooler(self.ctx_lstm_dim*self.ctx_lstm_directions)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.context_to_emo = nn.Linear(self.ctx_lstm_dim, NUM_EMO)
        self.drop_out = nn.Dropout(0.2)
        self.out2label = nn.Linear(self.ctx_lstm_dim * self.ctx_lstm_directions, NUM_EMO)
        self.out2binary = nn.Linear(self.ctx_lstm_dim * self.ctx_lstm_directions, 2)
        self.out2emo = nn.Linear(self.ctx_lstm_dim * self.ctx_lstm_directions, NUM_EMO-1)

    def init_hidden(self, x):
        batch_size = x.size(0)
        if self.bidirectional:
            h0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
        else:
            h0 = Variable(torch.zeros(1*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(1*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
        return (h0, c0)

    @staticmethod
    def sort_batch(batch, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        rever_sort = np.zeros(len(seq_lengths))
        for i, l in enumerate(perm_idx):
            rever_sort[l] = i
        return seq_tensor, seq_lengths, rever_sort.astype(int), perm_idx

    def lstm_forward(self, x, x_len, elmo_x, lstm, hidden=None, attention_layer=None):
        x, x_len_sorted, reverse_idx, perm_idx = self.sort_batch(x, x_len.view(-1))
        max_len = int(x_len_sorted[0])

        emb_x = self.embeddings(x)
        emb_x = self.drop_out(emb_x)
        emb_x = emb_x[:, :max_len, :]

        if self.use_elmo:
            elmo_x = elmo_x[perm_idx]
            elmo_x = self.drop_out(elmo_x)
            emb_x = torch.cat((emb_x, elmo_x), dim=2)

        packed_input = nn.utils.rnn.pack_padded_sequence(emb_x, x_len_sorted.cpu().numpy(), batch_first=True)
        if hidden is None:
            hidden = self.init_hidden(x)
        packed_output, hidden = lstm(packed_input, hidden)
        output, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        attention_layer = None  # testing
        if attention_layer is None:
            seq_len = torch.LongTensor(unpacked_len).view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
            seq_len = Variable(seq_len - 1).cuda()
            output = torch.gather(output, 1, seq_len).squeeze(1)
        else:
            if isinstance(attention_layer, AttentionOneParaPerChan):
                output, alpha = attention_layer(output, unpacked_len)
            else:
                unpacked_len = [int(x.data) for x in unpacked_len]
                # print(unpacked_len)
                max_len = max(unpacked_len)
                mask = [[1] * l + [0] * (max_len - l) for l in unpacked_len]
                mask = torch.FloatTensor(np.asarray(mask)).cuda()
                attention_mask = torch.ones_like(mask)
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # extended_attention_mask = extended_attention_mask.to(
                #       type=next(self.parameters()).dtype)  # fp16 compatibility
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                output, alpha = attention_layer(output, extended_attention_mask)
                # out, att = self.attention_layer(lstm_out[:, -1:].squeeze(1), lstm_out)
                output = output[:, 0, :]

        return output[reverse_idx], (hidden[0][:, reverse_idx, :], hidden[1][:, reverse_idx, :])

    def forward(self, a, a_len, b, b_len, c, c_len, a_emoji, b_emoji, c_emoji, elmo_a=None, elmo_b=None, elmo_c=None):
        # Sentence LSTM A
        a_out, a_hidden = self.lstm_forward(a, a_len, elmo_a, self.a_lstm,
                                            attention_layer=self.a_self_attention)
        # a_out = a_out[:, 0, :]
        if self.add_linear:
            a_emoji = self.deepmoji_model(a_emoji)
            a_emoji = self.deepmoji2linear(a_emoji)
            a_emoji = F.relu(a_emoji)
            a_emoji = self.drop_out(a_emoji)
        else:
            a_emoji = self.deepmoji_model(a_emoji)
            a_emoji = F.relu(a_emoji)

        # a_out = torch.cat((F.relu(a_out), a_emoji), dim=1)
        a_out = torch.cat((a_out, a_emoji), dim=1)
        # Sentence LSTM B
        b_out, b_hidden = self.lstm_forward(b, b_len, elmo_b, self.a_lstm,
                                            attention_layer=self.b_self_attention)
        # b_out = b_out[:, 0, :]
        if self.add_linear:
            b_emoji = self.deepmoji_model(b_emoji)
            b_emoji = self.deepmoji2linear(b_emoji)
            b_emoji = F.relu(b_emoji)
            b_emoji = self.drop_out(b_emoji)
        else:
            b_emoji = self.deepmoji_model(b_emoji)
            b_emoji = F.relu(b_emoji)

        # b_out = torch.cat((F.relu(b_out), b_emoji), dim=1)
        b_out = torch.cat((b_out, b_emoji), dim=1)
        # Sentence LSTM A
        c_out, c_hidden = self.lstm_forward(c, c_len, elmo_c, self.a_lstm,  # a_hidden,
                                            attention_layer=self.a_self_attention)
        # c_out = c_out[:, 0, :]
        if self.add_linear:
            c_emoji = self.deepmoji_model(c_emoji)
            c_emoji = self.deepmoji2linear(c_emoji)
            c_emoji = F.relu(c_emoji)
            c_emoji = self.drop_out(c_emoji)
        else:
            c_emoji = self.deepmoji_model(c_emoji)
            c_emoji = F.relu(c_emoji)

        # c_out = torch.cat((F.relu(c_out), c_emoji), dim=1)
        c_out = torch.cat((c_out, c_emoji), dim=1)
        # Context LSTM
        context_in = torch.cat((a_out.unsqueeze(1), b_out.unsqueeze(1), c_out.unsqueeze(1)), dim=1)
        ctx_out, _ = self.context_lstm(context_in)
        if isinstance(self.ctx_self_attention, AttentionOneParaPerChan):
            ctx_out, _ = self.ctx_self_attention(ctx_out, torch.LongTensor([3 for _ in range(a.size(0))]))
        else:
            ctx_out, _ = self.ctx_self_attention(ctx_out)
            ctx_out = ctx_out[:, 0, :]

        # ctx_out = self.ctx_layer_norm(ctx_out)
        # ctx_out = self.ctx_pooler(ctx_out)

        # multi-task learning
        out1 = self.out2label(ctx_out)
        out2 = self.out2binary(ctx_out)
        out3 = F.sigmoid(self.out2emo(ctx_out))
        return out1, out2, out3

    def load_embedding(self, emb):
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(emb))
