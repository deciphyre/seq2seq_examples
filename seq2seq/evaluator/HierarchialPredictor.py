import torch
from torch.autograd import Variable

class HierarchialPredictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab


    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """

        seq = [x.split('|') for x in src_seq]
        max_len = max(len(x) for x in seq)
        padded_seq = [x + ['<cpad>']*(max_len - len(x)) for x in seq]
        chunk_lengths = torch.LongTensor([len(x) for x in seq])
        src_id_seq = Variable(torch.LongTensor([[self.src_vocab.stoi[tok] for tok in x] for x in padded_seq]),
                              volatile=True).view(1, len(padded_seq), -1)

        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()
            chunk_lengths = chunk_lengths.cuda()
        softmax_list, _, other = self.model(src_id_seq, [len(padded_seq)], chunk_lengths)
        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq
