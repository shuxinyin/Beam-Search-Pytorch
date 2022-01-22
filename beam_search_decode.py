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
        print(output.shape, hidden.shape)  # (2, 1, 312)

        # output: [sequence_length, batch_size, D*hidden_size]
        # hidden: [D*num_layer, N, hidden_size]
        output, hidden = self.rnn(output, hidden)
        print(f"output.shape {output.shape}, hidden.shape： {hidden.shape}")

        out = self.linear(output.squeeze(0))
        output = F.log_softmax(out, dim=1)
        return output, hidden


def beam_search_decoder(post, top_k):
    """
    Parameters:
        post(Tensor) – the output probability of decoder. shape = (batch_size, seq_length, vocab_size).
        top_k(int) – beam size of decoder. shape
    return:
        indices(Tensor) – a beam of index sequence. shape = (batch_size, beam_size, seq_length).
        log_prob(Tensor) – a beam of log likelihood of sequence. shape = (batch_size, beam_size).
    """

    batch_size, seq_length, vocab_size = post.shape
    log_post = post.log()
    print("--", log_post[:, 0, :].shape)
    log_prob, indices = log_post[:, 0, :].topk(top_k, sorted=True)  # first word top-k candidates
    print(log_prob.shape, indices.shape)
    indices = indices.unsqueeze(-1)
    print(indices.shape)
    for i in range(1, seq_length):
        # log_post here should be computed again from rnn decoder in fact
        log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, top_k, 1)  # word by word
        log_prob, index = log_prob.view(batch_size, -1).topk(top_k, sorted=True)
        indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)
    return indices, log_prob


if __name__ == '__main__':
    post = torch.softmax(torch.randn([32, 20, 1000]), -1)
    print(post.shape)
    indices, log_prob = beam_search_decoder(post, top_k=3)
    print(indices.shape, log_prob.shape)
    # print(log_prob)
