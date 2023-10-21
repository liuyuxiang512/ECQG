from transformers import T5ForConditionalGeneration, BartForConditionalGeneration, BertForTokenClassification
from transformers.file_utils import ModelOutput
import torch
from torch import nn


class GenCONE(nn.Module):
    def __init__(self, model_checkpoint, gamma, max_gen_len, supervision):
        super(GenCONE, self).__init__()
        self.gamma = gamma
        self.max_gen_len = max_gen_len
        self.supervision = supervision

        if model_checkpoint in ["t5-base", "bart-base"]:
            bert_checkpoint = "bert-base-uncased"
        else:  # "t5-large", "bart-large"
            bert_checkpoint = "bert-large-uncased"

        self.QG_model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
        self.hidden_dim = self.QG_model.config.d_model

        self.CF_model = BertForTokenClassification.from_pretrained(bert_checkpoint)
        self.QV_model = BertForTokenClassification.from_pretrained(bert_checkpoint)

        self.linear_CF = nn.Linear(self.hidden_dim + 2, self.hidden_dim)
        self.linear_sim = nn.Linear(self.hidden_dim * 3, 1)
        self.linear_QV = nn.Linear(self.hidden_dim * 4, self.hidden_dim)

        self.CELoss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, questions_labels, token_tags, mode):
        # ... Encoder ...
        encoder_hc = self.QG_model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # ... Content Focusing ...
        if self.supervision in ["Both", "CF"]:
            cf_output = self.CF_model(inputs_embeds=encoder_hc, attention_mask=attention_mask)[0]
            cf_scores = nn.functional.softmax(cf_output, dim=2)
            encoder_hcf = self.linear_CF(torch.cat([encoder_hc, cf_output], dim=2))
        else:  # QV, No
            encoder_hcf = encoder_hc
            cf_scores = torch.zeros((encoder_hc.size(0), encoder_hc.size(1), 2), device=encoder_hc.device)

        # ... Question Generation ...
        qg_output = self.QG_model.generate(encoder_outputs=ModelOutput(last_hidden_state=encoder_hcf),
                                           attention_mask=attention_mask,
                                           max_new_tokens=self.max_gen_len,
                                           do_sample=False,
                                           output_hidden_states=True,
                                           return_dict_in_generate=True)
        qg_sequences = qg_output.sequences[:, 1:].contiguous()
        decoder_hq = qg_output.decoder_hidden_states
        decoder_hq = torch.cat([item[-1] for item in decoder_hq], dim=1)

        if mode == "eval":
            return qg_sequences

        loss_qg = self.QG_model(encoder_outputs=ModelOutput(last_hidden_state=encoder_hcf),
                                attention_mask=attention_mask,
                                labels=questions_labels).loss

        if self.supervision == "CF":
            loss_cf = self.CELoss(cf_scores.reshape(-1, 2), token_tags.reshape(-1, ))
            loss = self.gamma * loss_cf + loss_qg
            return qg_sequences, loss

        # ... Question Verification ...
        # DualAttention, combine encoder_hc & decoder_hq: batch_size, max_source_len, dim & batch_size, max_gen_len, dim
        Sim = torch.zeros((encoder_hc.size(0), encoder_hc.size(1), decoder_hq.size(1)), device=encoder_hc.device)
        for i in range(encoder_hc.size(1)):  # context word
            for j in range(decoder_hq.size(1)):  # query word
                hc_i = encoder_hc[:, i, :]
                hq_j = decoder_hq[:, j, :]
                hij = torch.cat([hc_i, hq_j, torch.mul(hc_i, hq_j)], dim=1)
                Sim[:, i, j] = self.linear_sim(hij).squeeze(1)

        A = nn.functional.softmax(Sim, dim=2)
        hq_hat = torch.matmul(A, decoder_hq)

        B = nn.functional.softmax(torch.max(Sim, dim=2).values, dim=1).unsqueeze(1)
        hc_hat = torch.tile(torch.matmul(B, encoder_hc), (1, encoder_hc.size(1), 1))

        hcqhat = torch.mul(encoder_hc, hq_hat)
        hcchat = torch.mul(encoder_hc, hc_hat)

        hcq = self.linear_QV(torch.cat([encoder_hc, hq_hat, hcqhat, hcchat], dim=2))

        # AnswerInferring
        qv_output = self.QV_model(inputs_embeds=hcq, attention_mask=attention_mask)[0]
        qv_scores = nn.functional.softmax(qv_output, dim=2)

        if self.supervision == "Both":
            loss_cf = self.CELoss(cf_scores.reshape(-1, 2), token_tags.reshape(-1,))
            loss_qv = self.CELoss(qv_scores.reshape(-1, 2), token_tags.reshape(-1,))
            loss = self.gamma * (loss_cf + loss_qv) / 2 + loss_qg
        elif self.supervision == "QV":
            loss_qv = self.CELoss(qv_scores.reshape(-1, 2), token_tags.reshape(-1, ))
            loss = self.gamma * loss_qv + loss_qg
        else:  # Not in paper yet
            loss_match = self.CELoss(cf_scores.reshape(-1, 2), qv_scores.reshape(-1, 2))
            loss = self.gamma * loss_match + (1 - self.gamma) * loss_qg

        return qg_sequences, loss
