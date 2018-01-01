import torch.nn as nn
import torch.nn.functional as F


class HSeq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio, volatile
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, hrnn, decoder, decode_function=F.log_softmax):
        super(HSeq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hrnn = hrnn
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.hrnn.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
#        print(input_variable.size())
#        print(input_variable.data)
        transformed_input = input_variable.view(-1, input_variable.size()[-1])
        #print(input_variable.size(), "Transformed into", transformed_input.size())
        batch_size = input_variable.size()[0]
        sequence_length = input_variable.size()[1] # [Batch Size, Seq Length, Chunk Length]
        chunk_length = input_variable.size()[2]
        transformed_input_lengths = [chunk_length] * transformed_input.size()[0] # New Batch size = batch_size * seq_len
#        print(input_lengths)
#        print(transformed.data)
#        encoder_outputs, encode_hidden = self.encoder(input_variable, input_lengths)
        encoder_outputs, encoder_hidden = self.encoder(transformed_input, transformed_input_lengths)
        #print("Outputs of Encoder", encoder_outputs.size(), encoder_hidden.size())

        last_outputs = encoder_outputs[:, -1, :] # From [Batch Size, Seq Length, Chunk Length]
        #print("Last outputs", last_outputs.size())
        reshaped_encoder_outputs = last_outputs.view(batch_size, sequence_length, -1)
        #print("Reshaped Outputs", reshaped_encoder_outputs.size())
        sequence_input_lengths = [sequence_length] * batch_size

        hrnn_outputs, hrnn_hidden = self.hrnn(reshaped_encoder_outputs, sequence_input_lengths)
        #print("HRNN Outputs", hrnn_outputs.size())

        result = self.decoder(inputs=target_variable,
                              encoder_hidden=hrnn_hidden,
                              encoder_outputs=hrnn_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
