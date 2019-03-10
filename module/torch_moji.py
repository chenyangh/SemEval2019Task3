import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from module.lstm_hard_sigmoid import LSTMHardSigmoid
from module.self_attention import AttentionOneParaPerChan
from os.path import exists

NB_TOKENS = 50000


class TorchMoji(nn.Module):
    def __init__(self, nb_classes, nb_tokens, output_logits=False,
                 embed_dropout_rate=0, final_dropout_rate=0, return_attention=False, IS_HALF=False):
        """
        torchMoji model.
        IMPORTANT: The model is loaded in evaluation mode by default (self.eval())

        # Arguments:
            nb_classes: Number of classes in the dataset.
            nb_tokens: Number of tokens in the dataset (i.e. vocabulary size).
            feature_output: If True the model returns the penultimate
                            feature vector rather than Softmax probabilities
                            (defaults to False).
            output_logits:  If True the model returns logits rather than probabilities
                            (defaults to False).
            embed_dropout_rate: Dropout rate for the embedding layer.
            final_dropout_rate: Dropout rate for the final Softmax layer.
            return_attention: If True the model also returns attention weights over the sentence
                              (defaults to False).
        """
        super(TorchMoji, self).__init__()

        embedding_dim = 256
        hidden_size = 512
        attention_size = 4 * hidden_size + embedding_dim

        self.embed_dropout_rate = embed_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.return_attention = return_attention
        self.hidden_size = hidden_size
        self.output_logits = output_logits
        self.nb_classes = nb_classes

        self.add_module('embed', nn.Embedding(nb_tokens, embedding_dim))
        # dropout2D: embedding channels are dropped out instead of words
        # many exampels in the datasets contain few words that losing one or more words can alter the emotions completely
        self.add_module('embed_dropout', nn.Dropout2d(embed_dropout_rate))
        self.add_module('lstm_0', LSTMHardSigmoid(embedding_dim, hidden_size, batch_first=True, bidirectional=True))
        self.add_module('lstm_1', LSTMHardSigmoid(hidden_size*2, hidden_size, batch_first=True, bidirectional=True))
        self.add_module('attention_layer', AttentionOneParaPerChan(attention_size=attention_size, IS_HALF=IS_HALF))
        self.add_module('final_dropout', nn.Dropout2d(final_dropout_rate))
        if self.nb_classes is not None:
            self.add_module('output_layer', nn.Sequential(nn.Linear(attention_size, nb_classes if self.nb_classes > 2 else 1)))

        self.init_weights()
        # Put model in evaluation mode by default
        # self.eval()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        nn.init.uniform(self.embed.weight.data, a=-0.5, b=0.5)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)
        if self.nb_classes is not None:
            nn.init.xavier_uniform(self.output_layer[0].weight.data)

    def forward(self, input_seqs):
        """ Forward pass.

        # Arguments:
            input_seqs: Can be one of Numpy array, Torch.LongTensor, Torch.Variable, Torch.PackedSequence.

        # Return:
            Same format as input format (except for PackedSequence returned as Variable).
        """

        ho = self.lstm_0.weight_hh_l0.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()
        co = self.lstm_0.weight_hh_l0.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()

        # Reorder batch by sequence length
        input_lengths = torch.LongTensor([torch.max(input_seqs[i, :].data.nonzero()) + 1 for i in range(input_seqs.size()[0])])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        input_seqs = input_seqs[perm_idx][:, :input_lengths.max()]

        # Pack sequence and work on data tensor to reduce embeddings/dropout computations
        packed_input = pack_padded_sequence(input_seqs, input_lengths.cpu().numpy(), batch_first=True)

        hidden = (Variable(ho, requires_grad=False), Variable(co, requires_grad=False))

        # Embed with an activation function to bound the values of the embeddings
        x = self.embed(packed_input.data)
        x = nn.Tanh()(x)

        # pyTorch 2D dropout2d operate on axis 1 which is fine for us
        x = self.embed_dropout(x)

        # Update packed sequence data for RNN
        packed_input = PackedSequence(x, packed_input.batch_sizes)

        # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
        # ordering of the way the merge is done is important for consistency with the pretrained model
        lstm_0_output, _ = self.lstm_0(packed_input, hidden)
        lstm_1_output, _ = self.lstm_1(lstm_0_output, hidden)

        # Update packed sequence data for attention layer
        packed_input = PackedSequence(torch.cat((lstm_1_output.data,
                                                 lstm_0_output.data,
                                                 packed_input.data), dim=1),
                                      packed_input.batch_sizes)

        input_seqs, _ = pad_packed_sequence(packed_input, batch_first=True)

        x, att_weights = self.attention_layer(input_seqs, input_lengths)

        if self.nb_classes is None:
            reorered = Variable(x.data.new(x.size()))
            reorered[perm_idx] = x
            return reorered

        # output class probabilities or penultimate feature vector
        x = self.final_dropout(x)
        outputs = self.output_layer(x)

        # Reorder output if needed
        reorered = Variable(outputs.data.new(outputs.size()))
        reorered[perm_idx] = outputs
        outputs = reorered
        return outputs

    def load_specific_weights(self, weight_path, exclude_names=[], extend_embedding=0, verbose=True):
        """ Loads model weights from the given file path, excluding any
            given layers.

        # Arguments:
            model: Model whose weights should be loaded.
            weight_path: Path to file containing model weights.
            exclude_names: List of layer names whose weights should not be loaded.
            extend_embedding: Number of new words being added to vocabulary.
            verbose: Verbosity flag.

        # Raises:
            ValueError if the file at weight_path does not exist.
        """
        if not exists(weight_path):
            raise ValueError('ERROR (load_weights): The weights file at {} does '
                             'not exist. Refer to the README for instructions.'
                             .format(weight_path))

        if extend_embedding and 'embed' in exclude_names:
            raise ValueError('ERROR (load_weights): Cannot extend a vocabulary '
                             'without loading the embedding weights.')

        # Copy only weights from the temporary model that are wanted
        # for the specific task (e.g. the Softmax is often ignored)
        weights = torch.load(weight_path)
        for key, weight in weights.items():
            if any(excluded in key for excluded in exclude_names):
                if verbose:
                    print('Ignoring weights for {}'.format(key))
                continue

            try:
                model_w = self.state_dict()[key]
            except KeyError:
                raise KeyError("Weights had parameters {},".format(key)
                               + " but could not find this parameters in model.")

            if verbose:
                print('Loading weights for {}'.format(key))

            # extend embedding layer to allow new randomly initialized words
            # if requested. Otherwise, just load the weights for the layer.
            if 'embed' in key and extend_embedding > 0:
                weight = torch.cat((weight, model_w[NB_TOKENS:, :]), dim=0)
                if verbose:
                    print('Extended vocabulary for embedding layer ' +
                          'from {} to {} tokens.'.format(
                            NB_TOKENS, NB_TOKENS + extend_embedding))
            try:
                model_w.copy_(weight)
            except:
                print('While copying the weigths named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the saved file are {}, ...'.format(
                            key, model_w.size(), weight.size()))
                raise


