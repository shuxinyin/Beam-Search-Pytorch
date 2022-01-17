import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        '''
        Illustrative decoder
        '''
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embedding_size,
                                      )
        self.rnn = nn.GRU(embedding_size,
                          hidden_size,
                          num_layers,
                          bidirectional=True,
                          dropout=dropout,
                          batch_first=False)
        self.dropout_rate = dropout
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, hidden, not_used=None):
        embedded = self.embedding(input).transpose(0, 1)  # [B,1] -> [ 1, B, D]
        embedded = F.dropout(embedded, self.dropout_rate)

        # output: [sequence_length, batch_size, input_size]
        # hidden: [D*num_layer, batch_size, hidden_size]
        output = embedded
        print(output.shape, hidden.shape)

        # output: [sequence_length, batch_size, D*hidden_size]
        # hidden: [D*num_layer, N, hidden_size]
        output, hidden = self.rnn(output, hidden)
        print(f"output.shape {output.shape}, hidden.shape： {hidden.shape}")

        out = self.linear(output.squeeze(0))
        output = F.log_softmax(out, dim=1)
        return output, hidden


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(target_tensor, decoder_hiddens, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]], device=device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch


def greedy_decode(decoder_hidden, target_tensor, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
    batch_size, seq_len = target_tensor.size()
    decoded_batch = torch.zeros((batch_size, MAX_LENGTH))
    decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=device)

    for t in range(MAX_LENGTH):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        print(f"decoder_output, decoder_hidden {decoder_output.shape}， {decoder_hidden.shape}")

        topv, topi = decoder_output.data.topk(1)  # get candidates
        print(topi)
        topi = topi.view(-1)
        print(topi)
        decoded_batch[:, t] = topi

        decoder_input = topi.detach().view(-1, 1)

    return decoded_batch


def test_greedy_decode():
    target_tensor = torch.from_numpy(np.random.randint(100, size=[2, 16]))  # (B, seq_len)
    # hidden: [D*num_layer, batch_size, hidden_size]
    decoder_hidden = torch.randn(2, 2, 312)
    # out, hidden = decoder(input_tensor, hidden_state)

    decoded_batch = greedy_decode(decoder_hidden, target_tensor)
    print(f"decoded_batch: {decoded_batch.shape}")
    print(f"decoded_batch: {decoded_batch}")



def test_rnn():
    output = torch.randn(16, 2, 768)  # (B, seq_len)
    hidden = torch.randn(2, 2, 312)

    rnn = nn.GRU(768,
                 312,
                 1,
                 bidirectional=True,
                 dropout=0.1,
                 batch_first=False)

    output, hidden = rnn(output, hidden)
    print(output.shape, hidden.shape)


if __name__ == '__main__':
    import numpy as np

    # device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    SOS_token = 0
    EOS_token = 1
    MAX_LENGTH = 50
    decoder = DecoderRNN(embedding_size=768, hidden_size=312, output_size=256)

    # input_tensor = torch.from_numpy(np.random.randint(100, size=[2, 16]))  # (B, seq_len)
    # # hidden: [D*num_layer, batch_size, hidden_size]
    # hidden_state = torch.randn(2, 2, 312)
    # out, hidden = decoder(input_tensor, hidden_state)

    test_greedy_decode()
