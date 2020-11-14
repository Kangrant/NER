import torch.nn as nn
from crf import CRF
from transformers import BertConfig, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class BERT_NERModel(nn.Module):
    def __init__(self, label2id, device, need_birnn=True):
        super(BERT_NERModel, self).__init__()
        # BERT模型
        self.config = BertConfig.from_pretrained('BERT_model/bert_pretrain/bert_config.json')
        self.bert = BertModel.from_pretrained('BERT_model/bert_pretrain/pytorch_model.bin',config=self.config)

        self.dropout = SpatialDropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size * 2, len(label2id))
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device)

        self.need_birnn = need_birnn
        self.bilstm = nn.LSTM(input_size=self.config.hidden_size, hidden_size=self.config.hidden_size,
                              batch_first=True, num_layers=2, dropout=self.config.hidden_dropout_prob,
                              bidirectional=True)

    def forward(self, inputs_ids, input_mask):
        outputs = self.bert(input_ids=inputs_ids, attention_mask=input_mask)   # [batch_size, seq_len, hidden_size]
        sequence_output = outputs[0]  # B, L, H
        sequence_output = self.dropout(sequence_output)
        sequence_output = sequence_output * input_mask.float().unsqueeze(2)

        # pad_pack
        seq_length = input_mask.sum(1).cpu()
        pack_sequence = pack_padded_sequence(sequence_output, lengths=seq_length, batch_first=True)
        seqence_output, _ = self.bilstm(pack_sequence)
        seqence_output, _ = pad_packed_sequence(seqence_output, batch_first=True)

        features = self.classifier(seqence_output)
        return features

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags=None):
        features = self.forward(input_ids, input_mask)
        if input_tags is not None:
            return features, self.crf.calculate_loss(features, tag_list=input_tags, lengths=input_lens)
        else:
            return features
