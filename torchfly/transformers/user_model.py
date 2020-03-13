import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math

from ..utils.file_utils import gdrive_download
# from ..cuda import gpt_gelu as gelu
# assert installed
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .gpt_model import gelu, Conv1D, Attention, MLP, Block, GPT2LMHead, GPT2SmallConfig, GPT2Model

# pylint:disable=no-member
class GPT2SimpleLMwithCopyEncoder(nn.Module):
    """OpenAI GPT-2 model with a Language Modeling head ("Language Models are Unsupervised Multitask Learners").
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self.init_weights)
        
    @classmethod
    def from_pretrained(cls, modelname):
        if modelname == "unified-gpt2-small":
            model = cls(GPT2SmallConfig)
            url = "https://drive.google.com/uc?id=1C5uuC2RNMwIjLC5UInmoEVXbX-U1OEvF"
            filepath = gdrive_download(url, "models", "unified-gpt2-small.pth")
            states_dict = torch.load(filepath)
            model.load_state_dict(states_dict)
            return model

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, past=None, mask=None):

        if past is None:
            past_length = input_ids.shape[1]
        else:
            # count self
            past_length = past[0].shape[3] + input_ids.shape[1]

        if mask is None:
            # print("mask is not provided")
            mask = torch.ones(
                input_ids.shape[0],
                past_length,
                dtype=torch.bool,
                device=input_ids.device
            )

        # Fast way to compute lower triangle attention mask
        # shape: (batch, num_head, key_length, query_length/seq_length)
        mask = mask.view(input_ids.shape[0], 1, 1, mask.shape[1]).repeat(
            1, self.config.n_head, mask.shape[1], 1
        )
        mask = mask & mask.permute(0, 1, 3, 2)
        mask = torch.tril(mask.byte())
        mask = mask.bool()
        mask = mask[:, :, -input_ids.shape[1]:, :]

        hidden_states, presents = self.transformer(
            input_ids, position_ids, past, mask
        )
        # lm_logits = self.lm_head(hidden_states)
        # probability_copy = torch.sigmoid(linear_copy(hidden_states))
        # return lm_logits, presents
        return hidden_states, presents

class GPT2SimpleLMwithCopyDecoder(nn.Module):
    """OpenAI GPT-2 model with a Language Modeling head ("Language Models are Unsupervised Multitask Learners").
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self.init_weights)
        # todo: use config to set the input and out put dimension of linear transformation
        self.linear_copy = nn.Linear(config.n_embd*2, 1)
        self.copy_project = nn.Linear(config.n_embd*2, config.n_embd)
        self.copy_head = GPT2LMHead(self.transformer.wte.weight, config)

        # initialize modules that are related to copy mechanism
        torch.nn.init.xavier_uniform_(self.linear_copy.weight)
        torch.nn.init.xavier_uniform_(self.copy_project.weight)
        self.linear_copy.bias.data.zero_()
        self.copy_project.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, modelname):
        if modelname == "unified-gpt2-small":
            model = cls(GPT2SmallConfig)
            url = "https://drive.google.com/uc?id=1C5uuC2RNMwIjLC5UInmoEVXbX-U1OEvF"
            filepath = gdrive_download(url, "models", "unified-gpt2-small.pth")
            states_dict = torch.load(filepath)
            model.load_state_dict(states_dict, strict=False)
            return model

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, past=None, mask=None, lastEncoderHidden = None):

        if past is None:
            past_length = input_ids.shape[1]
        else:
            # count self
            past_length = past[0].shape[3] + input_ids.shape[1]

        if mask is None:
            # print("mask is not provided")
            mask = torch.ones(
                input_ids.shape[0],
                past_length,
                dtype=torch.bool,
                device=input_ids.device
            )

        # Fast way to compute lower triangle attention mask
        # shape: (batch, num_head, key_length, query_length/seq_length)
        mask = mask.view(input_ids.shape[0], 1, 1, mask.shape[1]).repeat(
            1, self.config.n_head, mask.shape[1], 1
        )
        mask = mask & mask.permute(0, 1, 3, 2)
        mask = torch.tril(mask.byte())
        mask = mask.bool()
        mask = mask[:, :, -input_ids.shape[1]:, :]

        hidden_states, presents = self.transformer(
            input_ids, position_ids, past, mask
        )

        # get last hidden_states from decoder
        # concatenate hidden_states from encoder and decoder
        # calculate prob_copy
        # copy_prob = torch.sigmoid(linear_copy(hidden_states))
        # update logits as the fusion of generation distribution and copy distribution

        if type(lastEncoderHidden) != None:
            lastEncoderHiddens = lastEncoderHidden.unsqueeze(1).repeat(1,hidden_states.shape[-2],1)
            copyHiddens = torch.cat([lastEncoderHiddens, hidden_states], dim = 2)
            tmpCopyHiddens = self.linear_copy(copyHiddens)
            copy_prob = torch.sigmoid(tmpCopyHiddens)

            projectedCopyHiddens = self.copy_project(copyHiddens)
            copy_logits = self.copy_head(projectedCopyHiddens)

            lm_logits = self.lm_head(hidden_states)
            lm_logits = (1-copy_prob)*lm_logits + copy_prob*copy_logits

        else:
            lm_logits = self.lm_head(hidden_states)
        
        return lm_logits, presents
