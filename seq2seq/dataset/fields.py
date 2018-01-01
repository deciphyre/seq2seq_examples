import logging
import torch
from torch.autograd import Variable

import torchtext
from torchtext.data.dataset import Dataset

from collections import Counter, OrderedDict

class SourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True

        super(SourceField, self).__init__(**kwargs)



class HierarchialSourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)
        field_sep = kwargs.get('field_sep')
        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True

        self.chunk_pad_token = '<cpad>'

        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: [x.split(field_sep) for x in seq]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: func([x.split(field_sep) for x in seq])

#        self.postprocessing = self.prenumericalize
        
        super(HierarchialSourceField, self).__init__(**kwargs)


#    def prenumericalize(self, arr, vocab, train):
#        print(arr)

    def numericalize(self, arr, device=None, train=True):

        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)

        if self.use_vocab:
            if self.sequential:
                arr = [[[self.vocab.stoi[y] for y in x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab, train)

        else:
            if self.tensor_type not in self.tensor_types:
                raise ValueError(
                    "Specified Field tensor_type {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.tensor_type))
            numericalization_func = self.tensor_types[self.tensor_type]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None, train)

#        print(arr)

        arr = self.tensor_type(arr)
#        print(arr.shape)
        if self.sequential and not self.batch_first:
            arr.t_()
        if device == -1:
            if self.sequential:
                arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
            if self.include_lengths:
                lengths = lengths.cuda(device)
        if self.include_lengths:
            return Variable(arr, volatile=not train), lengths
        return Variable(arr, volatile=not train)        

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch

        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []


        field_max_len = max([len(y) for x in minibatch for y in x])
        
#        print(max_len, field_max_len)
        for x in minibatch:
        #    print(x)

            ypadded = []
            for y in x:

                if self.pad_first:
                    ypadded.append(
                        [self.chunk_pad_token] * max(0, field_max_len - len(y)) +
                        ([] if self.init_token is None else [self.init_token]) +
                        list(y[:field_max_len]) +
                        ([] if self.eos_token is None else [self.eos_token]))
                else:
                    ypadded.append(
                        ([] if self.init_token is None else [self.init_token]) +
                        list(y[:field_max_len]) +
                        ([] if self.eos_token is None else [self.eos_token]) +
                        [self.chunk_pad_token] * max(0, field_max_len - len(y)))

            
            
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(ypadded)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(ypadded[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(ypadded[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(ypadded)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(ypadded)))

 #       print(minibatch)
 #       print(padded)
        if self.include_lengths:
            return (padded, lengths)
        return padded        

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                for y in x:
                    counter.update(y)

        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)



        
#        print(args)
#        super(TargetField, self).build_vocab(*args, **kwargs)
#        self.sos_id = self.vocab.stoi[self.SYM_SOS]
#        self.eos_id = self.vocab.stoi[self.SYM_EOS]

        


class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]
