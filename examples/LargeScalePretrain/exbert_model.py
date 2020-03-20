import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# pylint:disable=no-member,invalid-unary-operand-type


# @torch.jit.script
def h_swish(x):
    """Hard Swish: MobileNetV3 https://arxiv.org/pdf/1905.02244.pdf
    """
    return x * F.relu6(x + 3) / 6


def swish(x):
    return x * torch.sigmoid_(x)


ACT2FN = {"gelu": F.gelu, "h_swish": h_swish, "swish": swish}


# class ScaleNorm(nn.Module):
#     """ScaleNorm"""
#     def __init__(self, hidden_size, scale, eps=1e-5):
#         super(ScaleNorm, self).__init__()
#         self.scale = nn.Parameter(torch.ones(hidden_size) * scale, requires_grad=True)
#         self.eps = eps

#     def forward(self, x):
#         norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
#         return x * norm

class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, hidden_size, scale, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * x


class ExBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

    def init_weights(self):
        self.word_embeddings.data.uniform_(-self.config.embedding_std, self.config.embedding_std)
        self.position_embeddings.data.uniform_(-self.config.embedding_std, self.config.embedding_std)


class ExBertSelfAttention(nn.Module):
    def __init__(self, config):
        """
        This version is modified to support batch training
        mask needs to be precomputed
        """
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.split_size = config.hidden_size
        self.scale = np.sqrt(self.attention_head_size)

        self.persist_key = nn.Parameter(
            torch.empty(1, config.num_attention_heads, self.attention_head_size, config.persistent_mem_size)
        )
        self.persist_value = nn.Parameter(
            torch.empty(1, config.num_attention_heads, config.persistent_mem_size, self.attention_head_size)
        )

        self.qkv_project = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        nn.init.xavier_normal_(self.persist_key)
        nn.init.xavier_normal_(self.persist_value)

    def _compute_attention(self, query, key, value, mask):
        attention = torch.matmul(query, key)
        # scale
        attention = attention / self.scale
        # shape: (batch, num_head, query_len, key_len)
        mask = mask[:, :, -attention.size(-2):, :]
        attention.masked_fill_(~mask, -5e4)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        return torch.matmul(attention, value)

    def merge_heads(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_x_shape = hidden_states.size()[:-2] + (hidden_states.size(-2) * hidden_states.size(-1), )
        return hidden_states.view(*new_x_shape)

    def split_heads(self, hidden_states, key=False):
        new_x_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(*new_x_shape)

        if key:
            # (batch, head, head_features, seq_length)
            return hidden_states.permute(0, 2, 3, 1)
        else:
            # (batch, head, seq_length, head_features)
            return hidden_states.permute(0, 2, 1, 3)

    def _compute_q_k_v(self, hidden_states):
        hidden_states = self.qkv_project(hidden_states)
        query, key, value = hidden_states.split(self.split_size, dim=2)
        query = self.split_heads(query)
        value = self.split_heads(value)
        key = self.split_heads(key, key=True)
        return query, key, value

    def forward(self, hidden_states, attention_mask=None):
        query, key, value = self._compute_q_k_v(hidden_states)
        # concatenate persistent memory
        key = torch.cat([self.persist_key.expand(key.size(0), -1, -1, -1), key], dim=-1)
        value = torch.cat([self.persist_value.expand(value.size(0), -1, -1, -1), value], dim=-2)
        hidden_states = self._compute_attention(query, key, value, attention_mask)
        hidden_states = self.merge_heads(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ExBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ExBertSelfAttention(config)
        self.mlp = FeedForward(config)
        self.feature_norm1 = ScaleNorm(config.hidden_size, config.hidden_size**0.5, eps=config.layer_norm_eps)
        self.feature_norm2 = ScaleNorm(config.hidden_size, config.hidden_size**0.5, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        attention_outputs = self.attention(self.feature_norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + attention_outputs
        mlp_outputs = self.mlp(self.feature_norm2(hidden_states))
        hidden_states = hidden_states + mlp_outputs

        return hidden_states


class ExBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([ExBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):

        for _, layer_module in enumerate(self.layer):

            if self.gradient_checkpointing:
                layer_outputs = torch.utils.checkpoint.checkpoint(layer_module, hidden_states, attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask)

            hidden_states = layer_outputs

        # Add last layer
        # if self.output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states, )
        # outputs = (hidden_states, )
        # if self.output_hidden_states:
        #     outputs = outputs + (all_hidden_states, )
        # if self.output_attentions:
        #     outputs = outputs + (all_attentions, )

        outputs = hidden_states
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class ExBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.num_attention_heads = config.num_attention_heads
        self.persistent_mem_size = config.persistent_mem_size

        self.embeddings = ExBertEmbeddings(config)
        self.encoder = ExBertEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor = None,
        position_ids: torch.LongTensor = None,
    ):
        if attention_mask is None:
            attention_mask = input_ids != self.pad_token_id

        attention_mask = torch.cat(
            [torch.ones(input_ids.size(0), self.persistent_mem_size, device=input_ids.device).bool(), attention_mask],
            dim=1
        )

        extended_attention_mask = attention_mask.view(input_ids.shape[0], 1, 1, attention_mask.shape[1]).repeat(
            1, self.num_attention_heads, attention_mask.shape[1], 1
        )
        extended_attention_mask = extended_attention_mask & extended_attention_mask.permute(0, 1, 3, 2)

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        hidden_states = self.encoder(hidden_states=embedding_output, attention_mask=extended_attention_mask)

        return hidden_states


class ExBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = ExBertModel(config)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)  # -100 index = padding token

        self.init_weights()
        self.set_embeddings_weights(self.bert.embeddings.word_embeddings.weight)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        masked_lm_labels=None,
    ):
        prediction_scores = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        prediction_scores = self.decoder(prediction_scores)

        results = {
            "outputs": prediction_scores,
        }

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_func(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)
            )
            results["loss"] = masked_lm_loss

        return results

    def init_weights(self):
        def _init_weights(module):
            """ Initialize the weights """
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_normal_(module.weight.data)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(_init_weights)