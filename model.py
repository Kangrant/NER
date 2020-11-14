from torch.nn import LayerNorm
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from crf import CRF

class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NERModel(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,
                 label2id,device,drop_p = 0.1):
        super(NERModel, self).__init__()
        self.emebdding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,
                              batch_first=True,num_layers=2,dropout=drop_p,
                              bidirectional=True)
        self.dropout = SpatialDropout(drop_p)
        self.layer_norm = LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2,len(label2id))
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device)

    def forward(self, inputs_ids, input_mask):
        embs = self.embedding(inputs_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)

        seq_length = input_mask.sum(1).cpu()

        # sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        # embs = embs[perm_idx, :]

        pack_sequence = pack_padded_sequence(embs, lengths=seq_length, batch_first=True)
        seqence_output, _ = self.bilstm(pack_sequence)
        seqence_output, _ = pad_packed_sequence(seqence_output, batch_first=True)

        # _, unperm_idx = perm_idx.sort()
        # seqence_output = seqence_output[unperm_idx, :]

        #seqence_output, _ = self.bilstm(embs)
        #seqence_output= self.layer_norm(seqence_output)
        features = self.classifier(seqence_output)
        return features

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags=None):
        features = self.forward(input_ids, input_mask)
        if input_tags is not None:
            return features, self.crf.calculate_loss(features, tag_list=input_tags, lengths=input_lens)
        else:
            return features

#test
#test123
#test12345
#test2
#test4