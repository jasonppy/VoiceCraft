import random

import numpy as np
import logging
import argparse, copy
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from .modules.utils import make_pad_mask

from .modules.embedding import SinePositionalEmbedding, TokenEmbedding
from .modules.transformer import (
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .codebooks_patterns import DelayedPatternProvider

from argparse import Namespace
from huggingface_hub import PyTorchModelHubMixin


def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token



class VoiceCraft(
        nn.Module,
        PyTorchModelHubMixin,
        library_name="voicecraft",
        repo_url="https://github.com/jasonppy/VoiceCraft",
        tags=["text-to-speech"],
    ):
    def __new__(cls, args: Optional[Namespace] = None, config: Optional[Dict] = None, **kwargs) -> "VoiceCraft":
        # If initialized from Namespace args => convert to dict config for 'PyTorchModelHubMixin' to serialize it as config.json
        # Won't affect instance initialization
        if args is not None:
            if config is not None:
                raise ValueError("Cannot provide both `args` and `config`.")
            config = vars(args)
        return super().__new__(cls, args=args, config=config, **kwargs)

    def __init__(self, args: Optional[Namespace] = None, config: Optional[Dict] = None):
        super().__init__()

        # If loaded from HF Hub => convert config.json to Namespace args before initializing
        if args is None:
            if config is None:
                raise ValueError("Either `args` or `config` must be provided.")
            args = Namespace(**config)

        self.args = copy.copy(args)
        self.pattern = DelayedPatternProvider(n_q=self.args.n_codebooks)
        if not getattr(self.args, "special_first", False):
            self.args.special_first = 0
        if not getattr(self.args, "n_special", False):
            self.args.n_special = 3
        self.args.eos = getattr(self.args, "eos", -1)
        self.eog = nn.Parameter(torch.full((self.args.n_codebooks, 1), self.args.eog, dtype=torch.long), requires_grad=False) # [K 1]
        if self.args.eos > 0:
            assert self.args.eos != self.args.audio_pad_token and self.args.eos != self.args.empty_token, self.args.eos
            self.eos = nn.Parameter(torch.full((self.args.n_codebooks, 1), self.args.eos, dtype=torch.long), requires_grad=False) # [K 1]
        if isinstance(self.args.audio_vocab_size, str):
            self.args.audio_vocab_size = eval(self.args.audio_vocab_size)

        self.n_text_tokens = self.args.text_vocab_size + 1
        assert self.args.text_pad_token == self.args.text_vocab_size, f"self.args.text_vocab_size: {self.args.text_vocab_size}, self.args.text_pad_token: {self.args.text_pad_token}"

        self.n_audio_tokens = [self.args.audio_vocab_size + self.args.n_special] * self.args.n_codebooks # special tokens: empty token, EOG token, audio pad token
        assert self.args.audio_vocab_size == self.args.empty_token, self.args.empty_token
        assert self.args.eog == self.args.audio_vocab_size + 1, self.args.eog
        assert self.args.audio_pad_token == self.args.audio_vocab_size + 2, self.args.audio_pad_token

        self.text_embedding = TokenEmbedding(
            dim_model=self.args.d_model,
            vocab_size=self.n_text_tokens, 
            dropout=self.args.text_embedding_dropout
        )

        self.audio_embedding = nn.ModuleList(
            [
                TokenEmbedding(
                dim_model=self.args.audio_embedding_dim, 
                vocab_size=self.n_audio_tokens[k], 
                dropout=self.args.audio_embedding_dropout
            ) for k in range(self.args.n_codebooks)
            ]
        )
        self.mask_embedding = nn.Parameter(torch.randn(self.args.max_n_spans, self.args.d_model), requires_grad=True)
        self.text_positional_embedding = SinePositionalEmbedding(
            self.args.d_model,
            dropout=self.args.text_positional_embedding_dropout,
            scale=False,
            alpha=True, # learnable scaler, scale the volume of positional embedding
        )
        self.audio_positional_embedding = SinePositionalEmbedding(
            self.args.d_model,
            dropout=self.args.audio_positional_embedding_dropout,
            scale=False,
            alpha=True, # learnable scaler, scale the volume of positional embedding
        )

        dec_layer = TransformerEncoderLayer(
            self.args.d_model,
            self.args.nhead,
            dim_feedforward=self.args.d_model * 4,
            dropout=self.args.trm_dropout,
            batch_first=True,
            norm_first=True,
            layer_norm_cls=LayerNorm
        )
        self.decoder = TransformerEncoder(
            dec_layer,
            num_layers=self.args.num_decoder_layers,
            norm=LayerNorm(self.args.d_model),
        )
        
        self.predict_layer = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.args.d_model, self.args.audio_vocab_size//2), nn.GELU(), nn.Linear(self.args.audio_vocab_size//2, self.n_audio_tokens[k])) for k in range(self.args.n_codebooks)
            ]
        )
        
        self.accuracy_metrics = nn.ModuleList(
            [MulticlassAccuracy(
                self.n_audio_tokens[k],
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=None,
            ) for k in range(self.args.n_codebooks)]
        )

    
    def prepare_mask_intervals(self, y_lens):
        mask_intervals = []
        non_mask_intervals = []

        for i, y_len in enumerate(y_lens):
            if self.args.mask_sample_dist == "uniform":
                n_spans = random.choice(range(1, self.args.max_n_spans+1))
            elif "poisson" in self.args.mask_sample_dist.lower():
                param = float(self.args.mask_sample_dist[len("poisson"):])
                poisson_sample = torch.poisson(torch.tensor([param]))
                n_spans = int(poisson_sample.clamp(1, self.args.max_n_spans).item())

            starts = random.sample(range(1, y_len-1-self.args.mask_len_min), n_spans)
            starts = sorted(starts)

            for j in range(len(starts)-1, 0, -1):
                if starts[j] - starts[j-1] < self.args.min_gap:
                    del starts[j] # If elements are too close, delete the later one
            assert len(starts) > 0, f"there is no masked span left, y_len: {y_len}, sampled n_spans: {n_spans}"

            temp_starts =  starts + [y_len]
            gaps = [temp_starts[j+1] - temp_starts[j] for j in range(len(temp_starts)-1)]

            ends = []

            for j, (start, gap) in enumerate(zip(starts, gaps)):
                mask_len = random.randint(self.args.mask_len_min, self.args.mask_len_max)
                # if mask_len > gap * self.args.max_mask_portion: # make sure the masks are not overlapping with each other
                if mask_len > gap - 1: # make sure the masks are not overlapping with each other
                    # temp_mask_start = int(0.6*gap*self.args.max_mask_portion)
                    # temp_mask_end = int(gap*self.args.max_mask_portion)
                    temp_mask_start = 1
                    temp_mask_end = gap - 1
                    mask_len = random.randint(temp_mask_start, temp_mask_end)
                ends.append(start + mask_len)
            
            mask_intervals.append([(s,e) for s,e in zip(starts, ends)])
            non_mask_intervals.append([(ns,ne) for ns, ne in zip([0]+ends, starts+[y_len])])

        return mask_intervals, non_mask_intervals
    
    def rearrange(self, y, non_mask_intervals, mask_intervals):
        reduced_eog = getattr(self.args, "reduced_eog", 0)
        rearranged_y = []
        for i in range(len(y)):
            if self.args.eos > 0:
                assert reduced_eog
                cur_y = [y[i, :, item[0]: item[1]] for item in non_mask_intervals[i][:-1]] + [torch.cat([y[i, :, non_mask_intervals[i][-1][0]: non_mask_intervals[i][-1][1]], self.eos], dim=-1)] + [torch.cat([y[i, :, item[0]: item[1]], self.eog], dim=-1) for item in mask_intervals[i]] # only insert eog to the last non-mask-interval, which is when the utterance actual ends
            else:
                if reduced_eog:
                    cur_y = [y[i, :, item[0]: item[1]] for item in non_mask_intervals[i][:-1]] + [torch.cat([y[i, :, non_mask_intervals[i][-1][0]: non_mask_intervals[i][-1][1]], self.eog], dim=-1)] + [torch.cat([y[i, :, item[0]: item[1]], self.eog], dim=-1) for item in mask_intervals[i]] # only insert eog to the last non-mask-interval, which is when the utterance actual ends
                else:
                    cur_y = [torch.cat([y[i, :, item[0]: item[1]], self.eog], dim=-1) for item in non_mask_intervals[i]] + [torch.cat([y[i, :, item[0]: item[1]], self.eog], dim=-1) for item in mask_intervals[i]] # eog is added to each section TODO this is not correct, I should add eog to non_mask_intervals if that segment is not the ending segment (as there is no way for the model to predict eog for those segments, and this will do harm to tts experiment, where the model randomly output eog for the first segment)
            rearranged_y.append(cur_y)
        return rearranged_y
        
    def shift(self, rearranged_y):
        shifted_y = []
        patterns = []
        for i in range(len(rearranged_y)):
            cur_patterns = [self.pattern.get_pattern(cur_y.shape[1]) for cur_y in rearranged_y[i]]
            out = [cur_pattern.build_pattern_sequence(z=cur_y.unsqueeze(0).contiguous(), special_token=self.args.empty_token, keep_only_valid_steps=False) for cur_pattern, cur_y in zip(cur_patterns, rearranged_y[i])]
            shifted_y.append([item[0].squeeze(0) for item in out]) # the first item is values, later two are indexes and mask
            patterns.append(cur_patterns)
        return shifted_y, patterns
    
    def insert_mask(self, shifted_y):
        inserted_y = []
        mask_position = []
        mask_value = []
        for i in range(len(shifted_y)):
            num_masks = (len(shifted_y[i]) - 1) // 2
            assert num_masks == (len(shifted_y[i]) - 1) / 2, len(shifted_y[i])
            emb_inds = list(range(self.args.max_n_spans))
            if self.args.shuffle_mask_embedding:
                random.shuffle(emb_inds)
            emb_inds_use = emb_inds[:num_masks]
            emb_inds_use = emb_inds_use + emb_inds_use
            mask_value.append(emb_inds_use)
            cur_inserted_y = []
            cur_mask_position = []
            for j in range(len(shifted_y[i])-1):
                cur_inserted_y.append(shifted_y[i][j])
                cur_mask_position.append(sum([item.shape[1] for item in cur_inserted_y])) # each item is of shape [K S], so take shape[1]
                cur_inserted_y.append(self.eog) # insert mask token of shape [K, 1], BUT we are actually using the eog token as a place holder here, as the real mask will be inserted in embed_y function

            cur_inserted_y.append(shifted_y[i][-1])

            inserted_y.append(cur_inserted_y)
            mask_position.append(cur_mask_position)
        return inserted_y, mask_position, mask_value
    
    def cat_y(self, inserted_y, mask_position, y_lens):
        reduced_eog = getattr(self.args, "reduced_eog", 0)
        cated_y = []
        new_y_lens = []
        for i in range(len(inserted_y)):
            cur_cated_y = torch.cat(inserted_y[i], dim=1) #[K S]
            cur_cated_y = cur_cated_y.transpose(1,0) # [S K]
            cur_cated_y_len = cur_cated_y.shape[0]
            if reduced_eog:
                assert cur_cated_y_len == y_lens[i] + len(mask_position[i]) + (len(mask_position[i]) + 1) * self.args.n_codebooks + (len(mask_position[i])/2 + 1), f"cur_cated_y_len == {cur_cated_y_len}, but it should be y_lens[i] ({y_lens[i]}) + len(mask_position[i]) ({len(mask_position[i])}) + (len(mask_position[i]) + 1) * self.args.n_codebooks ({(len(mask_position[i]) + 1) * self.args.n_codebooks}) + (len(mask_position[i])/2 + 1) ({len(mask_position[i])/2 + 1})={y_lens[i] + len(mask_position[i]) + (len(mask_position[i]) + 1) * self.args.n_codebooks + (len(mask_position[i])/2 + 1)}"
            else:
                assert cur_cated_y_len == y_lens[i] + len(mask_position[i]) + (len(mask_position[i]) + 1) * self.args.n_codebooks + (len(mask_position[i]) + 1), f"cur_cated_y_len == {cur_cated_y_len}, but it should be y_lens[i] ({y_lens[i]}) + len(mask_position[i]) ({len(mask_position[i])}) + (len(mask_position[i]) + 1) * self.args.n_codebooks ({(len(mask_position[i]) + 1) * self.args.n_codebooks}) + (len(mask_position[i]) + 1) ({len(mask_position[i]) + 1})" # the last term represent the inserted eog token, originally it's inserted at the end of every token, but this is wrong
            new_y_lens.append(cur_cated_y_len)
            cated_y.append(cur_cated_y)

        cated_y = torch.nn.utils.rnn.pad_sequence(cated_y, batch_first=False, padding_value=self.args.audio_pad_token)
        assert cated_y.shape == torch.Size([max(new_y_lens),len(inserted_y), self.args.n_codebooks]), f"cated_y.shape: {cated_y.shape}, but it should be {torch.Size([max(new_y_lens,len(inserted_y), self.args.n_codebooks)])}"
        cated_y = cated_y.permute(2,0,1) # [T,B,K]->[K,T,B]
        assert cated_y.shape[0] == self.args.n_codebooks, cated_y.shape
        return cated_y, torch.LongTensor(new_y_lens).to(cated_y.device)

    def embed_y(self, cated_y, mask_position, mask_value):
        embedded_y = torch.stack([self.audio_embedding[k](cated_y[k]) for k in range(self.args.n_codebooks)], dim=0) # [K, T, B, D]
        assert embedded_y.shape[0] == self.args.n_codebooks, embedded_y.shape
        assert embedded_y.shape[-1] == self.args.d_model, embedded_y.shape
        embedded_y = embedded_y.sum(dim=0) # [K,T,B,D]->[T,B,D]
        embedded_y = embedded_y.transpose(1,0) # [T,B,D]->[B,T,D]
        for i in range(len(embedded_y)):
            if len(mask_position[i]) > 0:
                embedded_y[i, mask_position[i]] = self.mask_embedding[mask_value[i]] 
        return embedded_y
    
    def prepare_input_target(self, y, y_lens):
        # rearrange y
        # assume y shape: [B T K], K is n_codebooks
        assert y.shape[1] == self.args.n_codebooks, y.shape
        # sample mask_intervals
        mask_intervals, non_mask_intervals = self.prepare_mask_intervals(y_lens)

        # need to have EOG in each section (SOG will be generated by the pattern class)
        # but mask can be inserted later after we have shifted the input
        # y could be rearranged in this way:
        # [
        # [tensor[4, 12], tensor[4, 45], tensor[4, 102], tensor[4, 32]], tensor[4, 22]],
        # [tensor[4, 44], tensor[4, 56], tensor[4, 19]],
        # ...
        # ]
        # for the first list of tensors (4 tensors), first 3 tensors are non_masked part, last 2 are masked part.
        # NOTE #non_masked_part = #masked_part + 1
        # NOTE *these are also the targets*
        # added eog at the end of each segment (masked segment and unmasked segment)
        rearranged_y = self.rearrange(y, non_mask_intervals, mask_intervals)
        targets = rearranged_y # each element in each sample is of shape [K T]
        assert targets[0][0].shape[0] == self.args.n_codebooks, targets[0][0].shape

        # next we need to apply pattern shifting to each tensor, after which, we'll replace the starting tokens of each section with a token that's different from the special padding token
        #  [[5, 1, 2, 3, 4, 5, 5],
        #  [5, 5, 1, 2, 3, 4, 5],
        #  [5, 5, 5, 1, 2, 3, 4]]
        shifted_y, patterns = self.shift(rearranged_y) # each element [K S]
        assert shifted_y[0][0].shape[0] == self.args.n_codebooks, shifted_y[0][0].shape[0]


        # then, insert mask token at the intersection of each tensor (we want to decide the arrangement of the mask (shuffle or not)), we better have a separate nn.embedding for it
        # we also need to record the position of the inserted mask
        inserted_y, mask_position, mask_value = self.insert_mask(shifted_y)
        assert inserted_y[0][0].shape[0] == self.args.n_codebooks, inserted_y[0][0].shape[0]
        assert inserted_y[0][1].shape == torch.Size((self.args.n_codebooks, 1)), f"this should be a mask, so should have shape {(self.args.n_codebooks, 1)}, but it's {inserted_y[0][1].shape}"
        
        # then concat tensors that belong to the same sample (in order) then get the length of each sample, and then stack them in batch dimension, pad them with pad_token
        cated_y, new_y_lens = self.cat_y(inserted_y, mask_position, y_lens) # KTB
        assert cated_y.shape == torch.Size((self.args.n_codebooks, cated_y.shape[1], len(inserted_y)))
        

        # embed remember to separately embed the mask tokens
        embedded_y = self.embed_y(cated_y, mask_position, mask_value) #BTD
        assert embedded_y.shape[1:] == torch.Size((max(new_y_lens), self.args.d_model)), embedded_y.shape
        
        # positional embedding
        y_input = self.audio_positional_embedding(embedded_y)

        # make attention mask and padding mask
        y_padding_mask = make_pad_mask(new_y_lens).to(y.device)
        y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y_padding_mask.device)
        return y_input, new_y_lens, targets, y_padding_mask, y_attention_mask, mask_position, patterns

    def remove_mask(self, logits, mask_position, new_y_lens):
        # logits: [B K S card]
        logits_use = []
        for i in range(len(logits)):
            non_mask_positions = [-1] + mask_position[i] + [new_y_lens[i]]
            non_mask_intervals = [[non_mask_positions[i]+1, non_mask_positions[i+1]] for i in range(len(non_mask_positions)-1)]
            cur_logits_use = [logits[i, :, l:r] for l,r in non_mask_intervals]
            logits_use.append(cur_logits_use)
        
        return logits_use
    
    def revert_pattern(self, patterns, logits_use):
        logits_final = []
        logit_masks = []
        for i in range(len(logits_use)):
            cur_logits = [
                item.unsqueeze(0).permute(0, 3, 1, 2).contiguous() for item in logits_use[i]
            ] # each item is of shape [1 K S card] [1 card K S]
            cur_logits_final = [
                cur_pattern.revert_pattern_logits(
                item, 0, keep_only_valid_steps=False
                )
                for cur_pattern, item in zip(patterns[i], cur_logits)
            ] # if input output order doesn't match, this step will give an error
            cur_logits_final_ret = [item[0].permute(0,2,3,1).squeeze(0) for item in cur_logits_final] # each element is of shape [K,T,card]
            logits_final.append(cur_logits_final_ret)
            logit_masks.append([item[2] for item in cur_logits_final])

        return logits_final, logit_masks

    def dec_forward(
            self, 
            x_input, 
            x_lens,
            x_attention_mask,
            x_padding_mask,
            y_input,
            new_y_lens,
            y_attention_mask,
            y_padding_mask,
            past=None,
            last_3_tokens=False
        ):
            x_attn_mask = F.pad(
                x_attention_mask,
                (0, new_y_lens.max()),
                value=True,
            ) # x attn to all x, doesn't attn to any y, this follow figure 3 of the valle paper
            y_attn_mask = F.pad(
                y_attention_mask,
                (x_lens.max(), 0), # y is padded at the front
                value=False,
            ) # y attn to all x, for y itself use lower triangle mask to ensure autoregressive
            xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)

            # merge key padding and attention masks
            bsz, src_len = x_input.shape[0], x_lens.max() + new_y_lens.max()
            xy_padding_mask = torch.concat([x_padding_mask, y_padding_mask], dim=1)
            _xy_padding_mask = (
                xy_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.args.nhead, -1, -1)
                .reshape(bsz * self.args.nhead, 1, src_len)
            )
            # Check shapes and resize+broadcast as necessary
            if xy_attn_mask.shape != _xy_padding_mask.shape:
                assert xy_attn_mask.ndim + 1 == _xy_padding_mask.ndim, f"xy_attn_mask.shape: {xy_attn_mask.shape}, _xy_padding_mask: {_xy_padding_mask.shape}"
                xy_attn_mask = xy_attn_mask.unsqueeze(0).repeat(_xy_padding_mask.shape[0], 1, 1)  # Example approach
            xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)

            new_attn_mask = torch.zeros_like(xy_attn_mask)
            new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
            xy_attn_mask = new_attn_mask

            xy_input = torch.cat([x_input, y_input], dim=1)

            if past == None: # do not use kvcache
                out, _ =  self.decoder((xy_input, None), mask=xy_attn_mask)
                return out[:, x_lens.max():], None
            else: # use kvcache
                if past.ndim > 3: # uses kvcache, only need to pass the last tokens, this doesn't work with multi-span speech editing yet
                    if last_3_tokens:
                        xy_input = xy_input[:, -3:]
                        xy_attn_mask = xy_attn_mask[:, -3:]
                    else:
                        xy_input = xy_input[:, -1:]
                        xy_attn_mask = xy_attn_mask[:, -1:]

                out, present =  self.decoder((xy_input, None), mask=xy_attn_mask, past=past)
                if isinstance(out, tuple): # get rid of stage_embedding
                    out = out[0]

                if out.shape[1] > x_lens.max(): # the first pass, not kvcache yet
                    return out[:, x_lens.max():], present
                else: # used kvcache
                    return out, present

    def forward(self, batch):
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, K, T).
            where K is the number of codebooks
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
        """
        x, x_lens, y, y_lens = batch["x"], batch["x_lens"], batch["y"], batch["y_lens"]
        if len(x) == 0:
            return None
        x = x[:, :x_lens.max()] # this deal with gradient accumulation, where x_lens.max() might not be longer than the length of the current slice of x
        y = y[:, :, :y_lens.max()]
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3 and y.shape[1] == self.args.n_codebooks, y.shape
        assert y_lens.ndim == 1, y_lens.shape
        # makes attention mask and padding mask for x
        x_padding_mask = make_pad_mask(x_lens).to(x.device)
        x_attention_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x_padding_mask.device)
        x_input = self.text_embedding(x)
        x_input = self.text_positional_embedding(x_input)
        y_input, new_y_lens, targets, y_padding_mask, y_attention_mask, mask_position, patterns = self.prepare_input_target(y, y_lens)
        y_out = self.dec_forward(
                    x_input, 
                    x_lens,
                    x_attention_mask,
                    x_padding_mask,
                    y_input,
                    new_y_lens,
                    y_attention_mask,
                    y_padding_mask
                )
        y_out = y_out[0] # no kv-caching during training
        assert y_out.shape == y_input.shape, f"y_out.shape: {y_out.shape}, y_input.shape: {y_input.shape}" # [B S D]
        
        logits = torch.stack([self.predict_layer[i](y_out) for i in range(self.args.n_codebooks)], dim=1) # [B K S card]
        # take out the mask token (using mask_position and new_y_lens) and revert (using function provided by self.pattern)
        assert logits.shape[1] == self.args.n_codebooks and logits.shape[3] == self.n_audio_tokens[0], logits.shape

        logits_use = self.remove_mask(logits, mask_position, new_y_lens)
        
        # revert the pattern shift for each logits section in each sample
        logits_final, logit_masks = self.revert_pattern(patterns, logits_use)
        assert logits_final[0][0].shape[0] == self.args.n_codebooks and logits_final[0][0].shape[2] == self.n_audio_tokens[0], f"it is: {logits_final[0][0].shape}, but should be [K, T, card]"
        # testing
        sample_to_test = 0
        assert len(logits_final[sample_to_test]) == len(targets[sample_to_test]), f"{len(logits_final[sample_to_test])}, {len(targets[sample_to_test])}"
        temp = sum([logits_final[sample_to_test][i].shape[:-1] != targets[sample_to_test][i].shape for i in range(len(targets[sample_to_test]))])
        assert temp == 0, f"none equal positions: {temp}, total number of elements: {len(targets[sample_to_test])}"

        logit_masked = sum([(item==False).any() for cur_mask in logit_masks for item in cur_mask])
        assert logit_masked == 0, logit_masks

        logits = torch.cat([torch.cat(item, dim=1) for item in logits_final], dim=1) # [K, T1+T2+T3+..., card]
        targets = torch.cat([torch.cat(item, dim=1) for item in targets], dim=1) # [K, T1+T2+T3+...]
        assert targets.shape[0] == logits.shape[0], f"{targets.shape}, {logits.shape}"
        loss = []
        ntokens = []
        top10acc = []
        for k, (logit, target) in enumerate(zip(logits, targets)):
            loss.append(F.cross_entropy(logit, target, reduction='mean'))
            top10acc.append(self.accuracy_metrics[k](logit.detach(), target))
            ntokens.append(len(logit))
        
        all_ntokens = sum(ntokens)
        if self.args.codebook_weight != None:
            codebook_weight = eval(self.args.codebook_weight)
        else:
            codebook_weight = [1.] * self.args.n_codebooks
        loss = sum([l*nt*cw for l, nt, cw in zip(loss, ntokens, codebook_weight)])
        top10acc_by_codebook = [t10a*nt for t10a, nt in zip(top10acc, ntokens)]
        top10acc = sum(top10acc_by_codebook)
        ntokens = torch.tensor(all_ntokens).to(logits.device)

        return {
            "loss": loss,
            "top10acc": top10acc,
            "top10acc_by_codebook": top10acc_by_codebook,
            "effective_ntoken": ntokens,
        }
    
    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        mask_interval: list[torch.Tensor],
        top_k: int=-100,
        top_p: float=1.0,
        temperature: float=1.0,
        stop_repetition: int=-1,
        kvcache: int=1,
        silence_tokens: list[int]=[1388,1898,131],
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, L).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, K).
          mask_interval:
            a list of tensors of shape (M, 2). contains M mask_start and mask_end. list length is actually 1, because we only support single sample inference for now
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          top_p: (`optional`) float
            For Neucleus sampling
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
          eog_coef: (`optional`) float
            if 0, no change to eog token logits, otherwise, will adjust eog token logit based on the difference between acoustic token and phn token length
          stop_repetition (`optional`) int
            if not -1, will set the logits of a token that repeated this many times to be -100000, to avoid generating it again. This only apply to tokens from the first codebook
          allowed_repeat_tokens (`optional`) list of ints
            by inspecting the validation set, get a few tokens that indeed repeat a significant amount of time, and exclude those tokens from prevent repetition
          ultimate_stop_repetition (`optional`) int
            no matter that token it is, stop repetition once after this number
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        if self.args.special_first:
            y = y + int(self.args.n_special)
        y = y.transpose(2,1) # [1,T,K] -> [1,K,T]
        assert y.shape[0] == 1 and y.shape[1] == self.args.n_codebooks, y.shape # there is no padding
        assert mask_interval.shape == torch.Size((1, mask_interval.shape[1], 2)), mask_interval

        # make x attention mask and x_input
        x_attention_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x.device)
        # x_attention_mask = torch.zeros(x.shape[1], x.shape[1]).bool().to(x.device)
        x_input = self.text_embedding(x)
        x_input = self.text_positional_embedding(x_input)

        # make initial y_input

        # make mask_interval and non_mask_interval
        y_len = y.shape[2]
        y_lens = torch.LongTensor([y_len]).to(y.device)
        mask_interval = mask_interval[0]
        starts =  [item[0].item() for item in mask_interval] + [y_len]
        ends = [0] + [item[1].item() for item in mask_interval]
        mask_intervals = [[
            (item[0].item(), item[1].item()) for item in mask_interval
        ]] # a werid name change, mask_interval is input, now is mask_intervals, with one more dimension
        non_mask_intervals = [[
            (ns, ne) for ns, ne in zip(ends, starts)
        ]]
        
        # rearrange y
        # will add have EOG in each section (SOG will be generated by the pattern class)
        # but mask can be inserted later after we have shifted the input
        # y could be rearranged in this way:
        # [
        # [tensor[4, 12], tensor[4, 45], tensor[4, 102], tensor[4, 32]], tensor[4, 22]],
        # [tensor[4, 44], tensor[4, 56], tensor[4, 19]],
        # ...
        # ]
        # for the first list of tensors (4 tensors), first 3 tensors are non_masked part, last 2 are masked part.
        # NOTE #non_masked_part = #masked_part + 1
        rearranged_y = self.rearrange(y, non_mask_intervals, mask_intervals)
        assert rearranged_y[0][0].shape[0] == self.args.n_codebooks, rearranged_y[0][0].shape

        # shift each element of y
        # next we need to apply pattern shifting to each tensor, after which, we'll replace the starting tokens of each section with a token that's different from the special padding token
        #  [
        #  [empty, 1, 2, 3, eog, empty, empty, empty],
        #  [empty, empty, 1, 2, 3, eog, empty, empty],
        #  [empty, empty, empty, 1, 2, 3, eog, empty],
        #  [empty, empty, empty, empty, 1, 2, 3, eog]
        # ]
        shifted_y, patterns = self.shift(rearranged_y) # each element [K S], patterns is not used, as we directly use the original input y
        assert shifted_y[0][0].shape[0] == self.args.n_codebooks, shifted_y[0][0].shape

        # insert mask token at the intersction of each tensor, but *actually inserted eog as place holder*
        # the position of inserted mask is also recorded
        # and the mask_value, the index of the mask emb is recorded
        inserted_y, mask_position, mask_value = self.insert_mask(shifted_y)
        assert inserted_y[0][0].shape[0] == self.args.n_codebooks, inserted_y[0][0].shape[0]
        assert inserted_y[0][1].shape == torch.Size((self.args.n_codebooks, 1)), f"this should be a mask, so should have shape {(self.args.n_codebooks, 1)}, but it's {inserted_y[0][1].shape}"
        
        # then concat tensors that belong to the same sample (in order) then get the length of each sample, and then stack them in batch dimension, pad them with pad_token
        cated_y, new_y_lens = self.cat_y(inserted_y, mask_position, y_lens) # KTB
        assert cated_y.shape == torch.Size((self.args.n_codebooks, cated_y.shape[1], len(inserted_y)))
        assert not (cated_y == self.args.audio_pad_token).any(), cated_y

        ### NOTE this is different from forward, as we will remove the masked tokens
        ### say there are two masked region
        ### the cated_y should be like
        ### [empty a a a a mask0 empty b b b mask1 empty c c mask0 empty]
        ### which means we need to take the part after the last empty out
        num_mask = len(mask_position[0])//2
        assert num_mask == len(mask_position[0])/2, mask_position
        cated_y = cated_y[:, :mask_position[0][num_mask]+2] # of shape [K,T,B]
        # logging.info(f"mask_position[0][num_mask]+2: {mask_position[0][num_mask]+2}")
        more_mask_value = mask_value[0][num_mask+1:] # NOTE this will be used in the generation loop for reference for inserting mask embedding
        new_y_lens[0] = mask_position[0][num_mask]+2
        mask_position[0] = mask_position[0][:num_mask+1]
        assert mask_position[0][num_mask]+2 == cated_y.shape[1], f"num_mask: {num_mask}, mask_position: {mask_position}, cated_y.shape: {cated_y.shape}"

        # embed: remember to separately embed the mask tokens
        embedded_y = self.embed_y(cated_y, mask_position, [mask_value[0][:num_mask+1]]) #BTD
        # assert embedded_y.shape == torch.Size((y.shape[0], max(new_y_lens), self.args.d_model)), embedded_y.shape

        # positional embedding
        y_input = self.audio_positional_embedding(embedded_y)

        # make attention mask and padding mask
        y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y.device)
        # y_lens = torch.LongTensor([y_input.shape[1]]).to(y.device)

        x_padding_mask = torch.full((1,x_lens[0]), False).to(x.device)
        y_padding_mask = torch.full((1,new_y_lens[0]), False).to(y.device)


        codebook_eog = [False] * self.args.n_codebooks
        generated = [] # doesn't contain any empty_token, contains eog
        cur_generated = []
        # say 0 is empty, 4 is eog
        # tensor([[ 1,  2,  3,  4,  0,  0],
        #         [ 0,  1,  2,  3,  4,  0],
        #         [ 0,  0,  1,  2,  3,  4]])
        num_gen = []
        cur_num_gen = 0
        ##################### silence repetition handling #####################
        ##################### silence repetition handling #####################
        logging.info(f"silence tokens: {silence_tokens}, note that if you are not using the pretrained encodec 6f79c6a8, make sure you specified it yourself, rather than using the default")
        consec_silence_count = 0
        prev_token = None
        ##################### silence repetition handling #####################
        ##################### silence repetition handling #####################
        # prepare the cache placeholder
        # n_layers, 2, bsz, num_heads, src_len, head_dim
        past = torch.ones([self.args.num_decoder_layers, 2, x.shape[0]], device=x.device, dtype=torch.float32) if kvcache else None
        # handle multi-span kv-cache
        new_masked_span = False
        
        def sample_helper(n_eog, logits, codebook_eog, top_k, top_p, temperature, prev_token, consec_silence_count, stop_repetition, silence_tokens, cur_num_gen):
            if n_eog == 0:
                logits_adjust = logits
                for jj in range(1,self.args.n_codebooks):
                    logits_adjust[jj][self.args.eog] = -10000
                    logits_adjust[jj][self.args.empty_token] = -10000
                ##################### silence repetition handling #####################
                if stop_repetition > 0 and prev_token in silence_tokens and consec_silence_count > stop_repetition:
                    if logits_adjust[0, prev_token] < 0:
                        logits_adjust[0, prev_token] = logits_adjust[0, prev_token] * (consec_silence_count - (stop_repetition-1))
                    else:
                        logits_adjust[0, prev_token] = logits_adjust[0, prev_token] / (consec_silence_count - (stop_repetition-1))
                ##################### silence repetition handling #####################
                if type(logits_adjust) == list:
                    samples_list= []
                    for logit in logits_adjust:
                        # print(logit)
                        # print(logit.shape)
                        cur_sample = topk_sampling(
                            logit.unsqueeze(0), top_k=top_k, top_p=top_p, temperature=temperature
                        ) # [1, 1]
                        samples_list.append(cur_sample)
                    samples = torch.cat(samples_list, dim=0) # [K, 1]
                else:
                    samples = topk_sampling(
                            logits_adjust, top_k=top_k, top_p=top_p, temperature=temperature
                        ) # [K, 1]
                assert samples.shape == torch.Size((self.args.n_codebooks, 1)), f"samples.shape: {samples.shape}"
                if cur_num_gen < self.args.n_codebooks-1:
                    for jj in range(1, self.args.n_codebooks - cur_num_gen):
                        samples[-jj, 0] = self.args.empty_token

                if (
                    samples[0,0] == self.args.eog or torch.argmax(logits[0], dim=-1) == self.args.eog or y_input.shape[1] > x_lens[0] * 10
                ): # last one means y is already too long, shouldn't happen, but put it here
                    samples[0,0] = self.args.eog
                    codebook_eog[0] = True
                ##################### silence repetition handling #####################
                ##################### silence repetition handling #####################
                if samples[0,0] in silence_tokens and samples[0,0] == prev_token:
                    consec_silence_count += 1
                else:
                    consec_silence_count = 0
                prev_token = samples[0,0]
                ##################### silence repetition handling #####################
                ##################### silence repetition handling #####################
                return samples, codebook_eog, prev_token, consec_silence_count
            else:
                assert sum(codebook_eog[i] for i in range(n_eog)) == n_eog, f"codebook_eog: {codebook_eog}, but n_eog: {n_eog}"
                logits_adjust = logits
                for jj in range(n_eog+1,self.args.n_codebooks):
                    logits_adjust[jj][self.args.eog] = -10000
                    logits_adjust[jj][self.args.empty_token] = -10000
                if type(logits_adjust) == list:
                    samples_list= []
                    for logit in logits_adjust:
                        cur_sample = topk_sampling(
                            logit.unsqueeze(0), top_k=top_k, top_p=top_p, temperature=temperature
                        ) # [1, 1]
                        samples_list.append(cur_sample)
                    samples = torch.cat(samples_list, dim=0) # [K, 1]
                else:
                    samples = topk_sampling(
                            logits_adjust, top_k=top_k, top_p=top_p, temperature=temperature
                        ) # [K, 1]
                for jj in range(n_eog):
                    samples[jj, 0] = self.args.empty_token
                samples[n_eog, 0] = self.args.eog
                codebook_eog[n_eog] = True
                return samples, codebook_eog, prev_token, consec_silence_count

        while True:
            y_out, present = self.dec_forward(
                                    x_input, 
                                    x_lens,
                                    x_attention_mask,
                                    x_padding_mask,
                                    y_input,
                                    new_y_lens,
                                    y_attention_mask,
                                    y_padding_mask,
                                    past=past,
                                    last_3_tokens = new_masked_span
                                    )
            if new_masked_span:
                new_masked_span = False

            if past != None:
                past = torch.cat([past, present.to(past.dtype)], dim=-2) if past.ndim > 3 else present.to(past.dtype)

            y_out = y_out[:, -1:] # only take the last one

            logits = torch.stack([self.predict_layer[i](y_out) for i in range(self.args.n_codebooks)], dim=1) # [B K S card], B==S==1, so [1 K 1 card]
            logits = logits.squeeze(0).squeeze(1) # [K card]
            assert logits.shape == torch.Size((self.args.n_codebooks, self.n_audio_tokens[0])), f"{logits.shape}"

            n_eog = sum(codebook_eog)
            assert n_eog < self.args.n_codebooks
            if self.args.eos > 0: # eos stands for end-of-sentence, which shouldn't be used as we are doing speech editing
                for jj in range(self.args.n_codebooks):
                    logits[jj][self.args.eos] = -10000.
            # need to use a helper function to hand different n_eog cases
            samples, codebook_eog, prev_token, consec_silence_count = sample_helper(n_eog, logits, codebook_eog, top_k, top_p, temperature, prev_token, consec_silence_count, stop_repetition, silence_tokens, cur_num_gen)
            cur_num_gen += 1
            cur_generated.append(samples.squeeze(-1)) # [K,1] -> [K]
            # get samples_emb
            samples_emb = torch.stack([self.audio_embedding[k](samples[k]) for k in range(self.args.n_codebooks)], dim=0) # [K,1,D]
            samples_emb = samples_emb.sum(dim=0,keepdim=True) # [1,1,D]

            if sum(codebook_eog) == self.args.n_codebooks: # generation for the current span is done
                # re-init
                codebook_eog = [False] * self.args.n_codebooks
                num_gen.append(cur_num_gen)
                cur_num_gen = 0
                generated.append(cur_generated)
                cur_generated = []

                # if the current mask span is the last span, then all done
                # else
                # append the next mask token and the four empty tokens to start the next generation 
                if len(more_mask_value) > 0:
                    next_mask_ind = more_mask_value.pop(0)
                    mask_emb = self.mask_embedding[next_mask_ind].unsqueeze(0).unsqueeze(0) # [1,1,D]
                    assert mask_emb.shape == torch.Size((1,1,self.args.d_model)), mask_emb.shape
                    empty_token = torch.LongTensor([self.args.empty_token]).to(y.device)
                    empty_emb = torch.stack([
                        self.audio_embedding[k](empty_token) for k in range(self.args.n_codebooks)], dim=0
                    ).sum(dim=0, keepdim=True) # [1,1,D]
                    assert empty_emb.shape == torch.Size((1,1,self.args.d_model)), empty_emb.shape
                    extra_emb = torch.cat([mask_emb, empty_emb], dim=1) # [1,2,D]
                    samples_emb = torch.cat([samples_emb, extra_emb], dim=1) # [1,3,D] # prev_last_token, mask_token, empty token
                    assert samples_emb.shape == torch.Size((1,3,self.args.d_model)), f"samples_emb.shape: {samples_emb.shape}"
                    ##################### silence repetition handling #####################
                    ##################### silence repetition handling #####################
                    consec_silence_count = 0
                    prev_token = None
                    ##################### silence repetition handling #####################
                    ##################### silence repetition handling #####################

                    # handling kv-caching for multi-span editing
                    new_masked_span = True
                else:
                    break
            else:
                assert samples_emb.shape == torch.Size((1,1,self.args.d_model)), f"samples_emb.shape: {samples_emb.shape}"

            embedded_y = torch.cat([embedded_y, samples_emb], dim=1)
            # positional embedding
            y_input = self.audio_positional_embedding(embedded_y) # [B T D]
            # make attention mask and padding mask
            y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y.device)
            new_y_lens = torch.LongTensor([y_input.shape[1]]).to(y.device)
            y_padding_mask = torch.full((1,new_y_lens[0]), False).to(y.device)
        
        assert len(generated) == num_mask, f"len(generated): {len(generated)}, num_mask: {num_mask}"

        # # combine non_masked_span with generated spans
        # first need to shift the generated part back
        flatten_gen = []
        for l, orig_span in enumerate(generated):
            span = torch.stack(orig_span, dim=0) # [T K]
            span = span.transpose(1,0) # [K, T]
            assert span.shape[0] == self.args.n_codebooks, span.shape
            unshifted_span = []
            for j, s in enumerate(span):
                start_from = j
                end_at = - (self.args.n_codebooks - start_from)
                unshifted_span.append(s[start_from:end_at])
            unshifted_span = torch.stack(unshifted_span, dim=0)

            assert unshifted_span.shape[1] == num_gen[l] - self.args.n_codebooks, f"len(unshifted_spans[0]): {len(unshifted_span[0])}, num_gen[l]: {num_gen[l]}"
            flatten_gen.append(unshifted_span)
        # logging.info(f"unshfited_span: {unshifted_span.shape}")
        # raise
        assert len(non_mask_intervals[0]) - 1 == len(flatten_gen), f"len(non_mask_intervals[0]): {len(non_mask_intervals[0])}, len(flatten_gen): {len(flatten_gen)}"
        res = []
        for orig_interval, gen in zip(non_mask_intervals[0], flatten_gen):
            res.append(y[0, :, orig_interval[0]:orig_interval[1]])
            res.append(gen)
        res.append(y[0, :, non_mask_intervals[0][-1][0]:non_mask_intervals[0][-1][1]])
        res = torch.cat(res, dim=1).unsqueeze(0) # [K,new_T] -> [1, K, new_T]

        expected_y_len = y_len - sum([item[1] - item[0] for item in mask_intervals[0]]) + sum([item - self.args.n_codebooks for item in num_gen])
        assert res.shape == torch.Size((1, self.args.n_codebooks, expected_y_len)), f"res.shape: {res.shape}, expected_y_len: {expected_y_len}. y_len - sum([item[1] - item[0] for item in mask_interval]) + sum([item - self.args.n_codebooks for item in num_gen]): {y_len}-{sum([item[1] - item[0] for item in mask_interval])} + {sum([item - self.args.n_codebooks for item in num_gen])}"
        
        if self.args.special_first:
            res = res - int(self.args.n_special)

        return res

    def inference_tts(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        top_k: int=-100,
        top_p: float=1.0,
        temperature: float=1.0,
        stop_repetition: int=3,
        kvcache: int=1,
        silence_tokens: list[int]=[1388,1898,131],
        *kargs
    ) -> torch.Tensor:
        """
        different from inference_tts, this implementation uses kvcache, which should have significant speed up
        Args:
          x:
            A 2-D tensor of shape (1, L).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, K).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          top_p: (`optional`) float
            For Neucleus sampling
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        """
        eog_inference = self.args.eos if self.args.eos>0 else self.args.eog
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        if self.args.special_first:
            y = y + int(self.args.n_special)
        y = y.transpose(2,1) # [1,T,K] -> [1,K,T]
        assert y.shape[0] == 1 and y.shape[1] == self.args.n_codebooks, y.shape # there is no padding

        # make x attention mask and x_input
        x_attention_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x.device)
        # x_attention_mask = torch.zeros(x.shape[1], x.shape[1]).bool().to(x.device)
        x_input = self.text_embedding(x)
        x_input = self.text_positional_embedding(x_input)

        y_len = y.shape[2]
        y_lens = torch.LongTensor([y_len]).to(y.device)

        # rearrange y, we don't add eog to the end, this doesn't actually do anything in the tts scenario
        rearranged_y = [[y[0]]]
        assert rearranged_y[0][0].shape[0] == self.args.n_codebooks, rearranged_y[0][0].shape

        # shift y to create the delayed pattern
        shifted_y, patterns = self.shift(rearranged_y) # each element [K S], patterns is not used, as we directly use the original input y
        assert shifted_y[0][0].shape[0] == self.args.n_codebooks, shifted_y[0][0].shape
        assert len(shifted_y[0]) == 1, len(shifted_y[0])

        # below is different from forward or inference
        # where we cut this shifted part 
        shifted_y[0][0] = shifted_y[0][0][:, :-(self.args.n_codebooks-1)]
        assert not (shifted_y[0][0][self.args.n_codebooks:] == self.args.empty_token).any() and not (shifted_y[0][0][self.args.n_codebooks:] == self.args.eog).any(), shifted_y[0][0]

        # next section in inference is insert mask at the intersection of each tensor in a sample, but we don't need to do that 
        # next section is concate tensors of each sample to one tensor, which we also don't need
        cated_y = shifted_y[0][0].unsqueeze(-1) #[K,S]->[K,S,B]
        new_y_lens = torch.LongTensor([cated_y.shape[1]]).to(cated_y.device)
        assert cated_y.shape == torch.Size((self.args.n_codebooks, cated_y.shape[1], 1))
        assert not (cated_y == self.args.audio_pad_token).any(), cated_y

        # replace tokens in y with the embeddings, add sum codebooks up
        embedded_y = torch.stack([self.audio_embedding[k](cated_y[k]) for k in range(self.args.n_codebooks)], dim=0) # [K, S, B, D]
        assert embedded_y.shape[0] == self.args.n_codebooks, embedded_y.shape
        assert embedded_y.shape[-1] == self.args.d_model, embedded_y.shape
        embedded_y = embedded_y.sum(dim=0) # [K,S,B,D]->[S,B,D]
        embedded_y = embedded_y.transpose(1,0) # [S,B,D]->[B,S,D]
        
        # positional embedding
        y_input = self.audio_positional_embedding(embedded_y)

        # make attention mask and padding mask
        y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y.device)

        x_padding_mask = torch.full((1,x_lens[0]), False).to(x.device)
        y_padding_mask = torch.full((1,new_y_lens[0]), False).to(y.device)

        # entering the generation stage
        # starting from line 708
        codebook_eog = [False] * self.args.n_codebooks
        generated = [] # doesn't contain any empty token, contain eog
        cur_generated = []
        # say 0 is empty, 4 is eog
        # tensor([[ 1,  2,  3,  4,  0,  0],
        #         [ 0,  1,  2,  3,  4,  0],
        #         [ 0,  0,  1,  2,  3,  4]])
        num_gen = []
        cur_num_gen = 0
        ##################### silence repetition handling #####################
        ##################### silence repetition handling #####################
        logging.info(f"silence tokens: {silence_tokens}, note that if you are not using the pretrained encodec 6f79c6a8, make sure you specified it yourself, rather than using the default")
        consec_silence_count = 0
        prev_token = None
        ##################### silence repetition handling #####################
        ##################### silence repetition handling #####################

        # prepare the cache placeholder
        # n_layers, 2, bsz, num_heads, src_len, head_dim
        past = torch.ones([self.args.num_decoder_layers, 2, x.shape[0]], device=x.device, dtype=torch.float32) if kvcache else None
        # logging.info(f"number of decoder layers: {self.args.num_decoder_layers}")
        # logging.info(f"number of decoder layers: {self.args.num_decoder_layers}")
        # logging.info(f"number of decoder layers: {self.args.num_decoder_layers}")
        def sample_helper(n_eog, logits, codebook_eog, top_k, top_p, temperature, prev_token, consec_silence_count, stop_repetition, silence_tokens, cur_num_gen):
            if n_eog == 0:
                logits_adjust = logits
                for jj in range(1,self.args.n_codebooks):
                    logits_adjust[jj][eog_inference] = -10000
                    logits_adjust[jj][self.args.empty_token] = -10000
                if cur_num_gen <= self.args.encodec_sr // 5: # this shouldn't happen, but just in case the model stopped too early
                    logits_adjust[0][eog_inference] = -10000
                ##################### silence repetition handling #####################
                if stop_repetition > 0 and prev_token in silence_tokens and consec_silence_count > stop_repetition:
                    if logits_adjust[0, prev_token] < 0:
                        logits_adjust[0, prev_token] = logits_adjust[0, prev_token] * (consec_silence_count - (stop_repetition-1))
                    else:
                        logits_adjust[0, prev_token] = logits_adjust[0, prev_token] / (consec_silence_count - (stop_repetition-1))
                ##################### silence repetition handling #####################
                samples = topk_sampling(
                        logits_adjust, top_k=top_k, top_p=top_p, temperature=temperature
                    ) # [K, 1]
                assert samples.shape == torch.Size((self.args.n_codebooks, 1)), f"samples.shape: {samples.shape}"
                if cur_num_gen < self.args.n_codebooks-1:
                    for jj in range(1, self.args.n_codebooks - cur_num_gen):
                        samples[-jj, 0] = self.args.empty_token

                if (
                    samples[0,0] == eog_inference or torch.argmax(logits[0], dim=-1) == eog_inference or y_input.shape[1] > x_lens[0] * (self.args.encodec_sr//5)
                ): # last one means y is already too long, shouldn't happen, but put it here
                    samples[0,0] = eog_inference
                    codebook_eog[0] = True
                ##################### silence repetition handling #####################
                if samples[0,0] in silence_tokens and samples[0,0] == prev_token:
                    consec_silence_count += 1
                else:
                    consec_silence_count = 0
                prev_token = samples[0,0]
                ##################### silence repetition handling #####################
                return samples, codebook_eog, prev_token, consec_silence_count
            else:
                assert sum(codebook_eog[i] for i in range(n_eog)) == n_eog, f"codebook_eog: {codebook_eog}, but n_eog: {n_eog}"
                logits_adjust = logits
                for jj in range(n_eog+1,self.args.n_codebooks):
                    logits_adjust[jj][eog_inference] = -10000
                    logits_adjust[jj][self.args.empty_token] = -10000
                samples = topk_sampling(
                        logits_adjust, top_k=top_k, top_p=top_p, temperature=temperature
                    ) # [K, 1]
                for jj in range(n_eog):
                    samples[jj, 0] = self.args.empty_token
                samples[n_eog, 0] = eog_inference
                codebook_eog[n_eog] = True
                return samples, codebook_eog, prev_token, consec_silence_count
        while True:
            y_out, present = self.dec_forward(
                            x_input, 
                            x_lens,
                            x_attention_mask,
                            x_padding_mask,
                            y_input,
                            new_y_lens,
                            y_attention_mask,
                            y_padding_mask,
                            past=past
                        )
            if past != None:
                past = torch.cat([past, present.to(past.dtype)], dim=-2) if past.ndim > 3 else present.to(past.dtype)


            y_out = y_out[:, -1:] # only take the last token
            logits = torch.stack([self.predict_layer[i](y_out) for i in range(self.args.n_codebooks)], dim=1) # [B K S card], B==S==1, so [1 K 1 card]
            logits = logits.squeeze(0).squeeze(1) # [K card]
            assert logits.shape == torch.Size((self.args.n_codebooks, self.n_audio_tokens[0])), f"{logits.shape}"

            n_eog = sum(codebook_eog)
            assert n_eog < self.args.n_codebooks
            if self.args.eos > 0: # if we are using end-of-sentence token (which is used by default), eog shouldn't be used here, as there is no masked spans
                for jj in range(self.args.n_codebooks):
                    logits[jj][self.args.eog] = -10000.
            
            samples, codebook_eog, prev_token, consec_silence_count = sample_helper(n_eog, logits, codebook_eog, top_k, top_p, temperature, prev_token, consec_silence_count, stop_repetition, silence_tokens, cur_num_gen)
            
            cur_num_gen += 1
            cur_generated.append(samples.squeeze(-1)) # [K,1] -> [K]

            # samples.shape is [K,1]
            # ge samples_emb
            samples_emb = torch.stack([self.audio_embedding[k](samples[k]) for k in range(self.args.n_codebooks)], dim=0) # [K,1,D]
            samples_emb = samples_emb.sum(dim=0,keepdim=True) # [1,1,D]

            if sum(codebook_eog) == self.args.n_codebooks: # generation for the current span is done
                codebook_eog = [False] * self.args.n_codebooks
                num_gen.append(cur_num_gen)
                cur_num_gen = 0
                generated.append(cur_generated)
                cur_generated = []
                break
            else:
                assert samples_emb.shape == torch.Size((1,1,self.args.d_model)), f"samples_emb.shape: {samples_emb.shape}"
            
            embedded_y = torch.cat([embedded_y, samples_emb], dim=1)
            y_input = self.audio_positional_embedding(embedded_y) # [B T D]
            # make attention mask and padding mask
            y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y.device)
            new_y_lens = torch.LongTensor([y_input.shape[1]]).to(y.device)
            y_padding_mask = torch.full((1,new_y_lens[0]), False).to(y.device)
        
        assert len(generated) == 1, f"len(generated): {len(generated)}"

        # revert the pattern
        flatten_gen = []
        for l, orig_span in enumerate(generated):
            span = torch.stack(orig_span, dim=0) # [T, K]
            span = span.transpose(1,0) # [K, T]
            assert span.shape[0] == self.args.n_codebooks, span.shape
            unshifted_span = []
            for j, s in enumerate(span):
                start_from = j
                end_at = - (self.args.n_codebooks - start_from)
                unshifted_span.append(s[start_from:end_at])
            unshifted_span = torch.stack(unshifted_span, dim=0)

            assert unshifted_span.shape[1] == num_gen[l] - self.args.n_codebooks, f"len(unshifted_spans[0]): {len(unshifted_span[0])}, num_gen[l]: {num_gen[l]}"

            flatten_gen.append(unshifted_span)
        assert len(flatten_gen) == 1, len(flatten_gen)
        
        # combine 
        res = [y[0], flatten_gen[0]]
        res = torch.cat(res, dim=1).unsqueeze(0) # [K, new_t] -> [1, K, new_T]

        expected_y_len = y_len + sum([item - self.args.n_codebooks for item in num_gen])
        assert res.shape == torch.Size((1, self.args.n_codebooks, expected_y_len)), f"res.shape: {res.shape}, expected_y_len: {expected_y_len}. y_len + sum([item - self.args.n_codebooks for item in num_gen]): {y_len} + {sum([item - self.args.n_codebooks for item in num_gen])}"

        if self.args.special_first:
            res = res - int(self.args.n_special)
            flatten_gen = flatten_gen - int(self.args.n_special)

        return res, flatten_gen[0].unsqueeze(0)


    def inference_tts_batch(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        top_k: int=-100,
        top_p: float=1.0,
        temperature: float=1.0,
        stop_repetition: int=3,
        kvcache: int=1,
        batch_size: int=5,
        silence_tokens: list[int]=[1388,1898,131],
        *kargs
    ) -> torch.Tensor:
        """
        have a batch size when forward passing, but they are equivalant to same example but different random seed, therefore as long as one example generated eog, we can drop all other samlpes
        different from inference_tts, this implementation uses kvcache, which should have significant speed up
        Args:
          x:
            A 2-D tensor of shape (1, L).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, K).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          top_p: (`optional`) float
            For Neucleus sampling
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        """
        eog_inference = self.args.eos if self.args.eos>0 else self.args.eog
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        if self.args.special_first:
            y = y + int(self.args.n_special)
        y = y.transpose(2,1) # [1,T,K] -> [1,K,T]
        assert y.shape[0] == 1 and y.shape[1] == self.args.n_codebooks, y.shape # there is no padding

        # make x attention mask and x_input
        x_attention_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x.device)
        # x_attention_mask = torch.zeros(x.shape[1], x.shape[1]).bool().to(x.device)
        x_input = self.text_embedding(x)
        x_input = self.text_positional_embedding(x_input)

        y_len = y.shape[2]
        y_lens = torch.LongTensor([y_len]).to(y.device)

        # rearrange y, we don't add eog to the end, this doesn't actually do anything in the tts scenario
        rearranged_y = [[y[0]]]
        assert rearranged_y[0][0].shape[0] == self.args.n_codebooks, rearranged_y[0][0].shape

        # shift y to create the delayed pattern
        shifted_y, patterns = self.shift(rearranged_y) # each element [K S], patterns is not used, as we directly use the original input y
        assert shifted_y[0][0].shape[0] == self.args.n_codebooks, shifted_y[0][0].shape
        assert len(shifted_y[0]) == 1, len(shifted_y[0])

        # below is different from forward or inference
        # where we cut this shifted part 
        shifted_y[0][0] = shifted_y[0][0][:, :-(self.args.n_codebooks-1)]
        assert not (shifted_y[0][0][self.args.n_codebooks:] == self.args.empty_token).any() and not (shifted_y[0][0][self.args.n_codebooks:] == self.args.eog).any(), shifted_y[0][0]

        # next section in inference is insert mask at the intersection of each tensor in a sample, but we don't need to do that 
        # next section is concate tensors of each sample to one tensor, which we also don't need
        cated_y = shifted_y[0][0].unsqueeze(-1) #[K,S]->[K,S,B]
        new_y_lens = torch.LongTensor([cated_y.shape[1]]).to(cated_y.device)
        assert cated_y.shape == torch.Size((self.args.n_codebooks, cated_y.shape[1], 1))
        assert not (cated_y == self.args.audio_pad_token).any(), cated_y

        # replace tokens in y with the embeddings, add sum codebooks up
        embedded_y = torch.stack([self.audio_embedding[k](cated_y[k]) for k in range(self.args.n_codebooks)], dim=0) # [K, S, B, D]
        assert embedded_y.shape[0] == self.args.n_codebooks, embedded_y.shape
        assert embedded_y.shape[-1] == self.args.d_model, embedded_y.shape
        embedded_y = embedded_y.sum(dim=0) # [K,S,B,D]->[S,B,D]
        embedded_y = embedded_y.transpose(1,0) # [S,B,D]->[B,S,D]
        
        # positional embedding
        y_input = self.audio_positional_embedding(embedded_y)

        # make attention mask and padding mask
        y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y.device)

        x_padding_mask = torch.full((1,x_lens[0]), False).to(x.device)
        y_padding_mask = torch.full((1,new_y_lens[0]), False).to(y.device)

        # entering the generation stage
        # starting from line 708
        codebook_eog = [False] * self.args.n_codebooks
        generated = [] # doesn't contain any empty token, contain eog
        cur_generated = [[] for _ in range(batch_size)]
        # say 0 is empty, 4 is eog
        # tensor([[ 1,  2,  3,  4,  0,  0],
        #         [ 0,  1,  2,  3,  4,  0],
        #         [ 0,  0,  1,  2,  3,  4]])
        num_gen = []
        cur_num_gen = 0
        ##################### silence repetition handling #####################
        ##################### silence repetition handling #####################
        logging.info(f"silence tokens: {silence_tokens}, note that if you are not using the pretrained encodec 6f79c6a8, make sure you specified it yourself, rather than using the default")
        consec_silence_counts = [0 for _ in range(batch_size)]
        prev_tokens = [None for _ in range(batch_size)]
        ##################### silence repetition handling #####################
        ##################### silence repetition handling #####################

        # prepare the cache placeholder
        # n_layers, 2, bsz, num_heads, src_len, head_dim
        past = torch.ones([self.args.num_decoder_layers, 2, x.shape[0]], device=x.device, dtype=torch.float32) if kvcache else None
        # logging.info(f"number of decoder layers: {self.args.num_decoder_layers}")
        # logging.info(f"number of decoder layers: {self.args.num_decoder_layers}")
        # logging.info(f"number of decoder layers: {self.args.num_decoder_layers}")
        keep = None # NOTE: this very important, tells which sample to keep
        def sample_helper(n_eog, logits, codebook_eog, top_k, top_p, temperature, prev_tokens, consec_silence_counts, stop_repetition, silence_tokens, cur_num_gen, keep):
            if n_eog == 0:
                logits_adjust = logits
                for jj in range(1,self.args.n_codebooks):
                    logits_adjust[:,jj,eog_inference] = -10000
                    logits_adjust[:,jj,self.args.empty_token] = -10000
                if cur_num_gen <= self.args.encodec_sr // 5: # this shouldn't happen, but just in case the model stopped too early
                    logits_adjust[:,:,eog_inference] = -10000
                ##################### silence repetition handling #####################
                for b in range(batch_size):
                    prev_token = prev_tokens[b]
                    consec_silence_count = consec_silence_counts[b]
                    if stop_repetition > 0 and prev_token in silence_tokens and consec_silence_count > stop_repetition:
                        if logits_adjust[b, 0, prev_token] < 0:
                            logits_adjust[b, 0, prev_token] = logits_adjust[b, 0, prev_token] * (consec_silence_count - (stop_repetition-1))
                        else:
                            logits_adjust[b, 0, prev_token] = logits_adjust[b, 0, prev_token] / (consec_silence_count - (stop_repetition-1))
                ##################### silence repetition handling #####################
                samples = topk_sampling(
                        logits_adjust.reshape(batch_size * self.args.n_codebooks, logits_adjust.shape[-1]), top_k=top_k, top_p=top_p, temperature=temperature
                    ) # [B*K, 1]
                samples = samples.reshape(batch_size, self.args.n_codebooks, 1)
                assert samples.shape == torch.Size((batch_size, self.args.n_codebooks, 1)), f"samples.shape: {samples.shape}"
                for b in range(batch_size):
                    if cur_num_gen < self.args.n_codebooks-1:
                        for jj in range(1, self.args.n_codebooks - cur_num_gen):
                            samples[b, -jj, 0] = self.args.empty_token

                    if (
                        samples[b,0,0] == eog_inference or torch.argmax(logits[b,0], dim=-1) == eog_inference or y_input.shape[1] > x_lens[b] * (self.args.encodec_sr//5)
                    ): # last one means y is already too long, shouldn't happen, but put it here
                        samples[b,0,0] = eog_inference
                        codebook_eog[0] = True
                        keep = b # NOTE keep is a very important variable, we only return this one, note that if eog shows up in two samples, keep will be overwritten by the later one (or the last one)
                    ##################### silence repetition handling #####################
                    if samples[b,0,0] in silence_tokens and samples[b,0,0] == prev_tokens[b]:
                        consec_silence_counts[b] += 1
                    else:
                        consec_silence_counts[b] = 0
                    prev_tokens[b] = samples[b,0,0]
                ##################### silence repetition handling #####################
                return samples, codebook_eog, prev_tokens, consec_silence_counts, keep
            else:
                assert sum(codebook_eog[i] for i in range(n_eog)) == n_eog, f"codebook_eog: {codebook_eog}, but n_eog: {n_eog}"
                logits_adjust = logits
                for jj in range(n_eog+1,self.args.n_codebooks):
                    logits_adjust[:,jj,eog_inference] = -10000
                    logits_adjust[:,jj,self.args.empty_token] = -10000
                samples = topk_sampling(
                        logits_adjust.reshape(batch_size * self.args.n_codebooks, logits_adjust.shape[-1]), top_k=top_k, top_p=top_p, temperature=temperature
                    ) # [B, K, 1]
                samples = samples.reshape(batch_size, self.args.n_codebooks, 1)
                for jj in range(n_eog):
                    samples[keep, jj, 0] = self.args.empty_token
                samples[keep, n_eog, 0] = eog_inference
                codebook_eog[n_eog] = True
                return samples, codebook_eog, prev_tokens, consec_silence_counts, keep
        while True:
            # if cur_num_gen > 0, should have everything in kvcache, so only pass in the last token
            # in the first generation step, we repeat each tensor to make their first dimension of length the batch size 
            if cur_num_gen == 0:
                assert x_input.ndim == 3 and x_input.shape[0] == 1, x_input.shape
                assert x_padding_mask.ndim == 2 and x_padding_mask.shape[0] == 1, x_padding_mask.shape
                assert y_input.ndim == 3 and y_input.shape[0] == 1 and y_input.shape[1] == new_y_lens[0], y_input.shape
                assert embedded_y.ndim == 3 and embedded_y.shape[0] == 1 and embedded_y.shape[1] == new_y_lens[0], embedded_y.shape
                x_input = x_input.repeat(batch_size, 1, 1)
                x_lens = x_lens.repeat(batch_size)
                # x_attention_mask = x_attention_mask.repeat(batch_size, 1, 1) # no need to work with attention mask, it doesn't contain batch dimension
                x_padding_mask = x_padding_mask.repeat(batch_size, 1)
                y_input = y_input.repeat(batch_size, 1, 1)
                new_y_lens = new_y_lens.repeat(batch_size)
                # y_attention_mask = y_attention_mask.repeat(batch_size, 1, 1) # no need to work with attention mask, it doesn't contain batch dimension
                y_padding_mask = y_padding_mask.repeat(batch_size, 1)
                embedded_y = embedded_y.repeat(batch_size, 1, 1) # will be used to concat with newly generated token embedding
                past = past.repeat(1, 1, batch_size) if past != None else None
            else:
                assert x_input.shape[0] == batch_size and x_padding_mask.shape[0] == batch_size and y_input.shape[0] == batch_size and new_y_lens.shape[0] == batch_size, f"x_input.shape: {x_input.shape}, x_padding_mask.shape: {x_padding_mask.shape}, y_input.shape: {y_input.shape}, new_y_lens.shape: {new_y_lens.shape}"
            y_out, present = self.dec_forward(
                            x_input, 
                            x_lens,
                            x_attention_mask,
                            x_padding_mask,
                            y_input,
                            new_y_lens,
                            y_attention_mask,
                            y_padding_mask,
                            past=past
                        )
            if past != None:
                past = torch.cat([past, present.to(past.dtype)], dim=-2) if past.ndim > 3 else present.to(past.dtype)

            # if no eog emerges, y_out should have batch size of batch_size
            if sum(codebook_eog) == 0:
                assert y_out.shape[0] == batch_size and y_out.ndim == 3, y_out.shape
            y_out = y_out[:, -1:] # only take the last token
            logits = torch.stack([self.predict_layer[i](y_out) for i in range(self.args.n_codebooks)], dim=1) # [B K S card], S==1, so [B K 1 card]
            logits = logits.squeeze(2) # [B K card]
            assert logits.shape == torch.Size((batch_size, self.args.n_codebooks, self.n_audio_tokens[0])), f"{logits.shape}"

            n_eog = sum(codebook_eog)
            if self.args.eos > 0:
                for jj in range(self.args.n_codebooks):
                    logits[:,jj,self.args.eog] = -10000.
            samples, codebook_eog, prev_tokens, consec_silence_counts, keep = sample_helper(n_eog, logits, codebook_eog, top_k, top_p, temperature, prev_tokens, consec_silence_counts, stop_repetition, silence_tokens, cur_num_gen, keep)
            
            cur_num_gen += 1
            if sum(codebook_eog) == 0: # no eog yet, keep batch_size of samples
                assert keep == None
                for b in range(batch_size):
                    cur_generated[b].append(samples[b].squeeze(-1))
            elif sum(codebook_eog) == 1: # the first eog just showed up in this step
                assert keep != None
                cur_generated = cur_generated[keep]
                cur_generated.append(samples[keep].squeeze(-1))
            else: # we are generating the rest eogs for the 'keep' sample 
                cur_generated.append(samples[keep].squeeze(-1))

            # samples.shape is [K,1]
            # ge samples_emb
            samples_emb = torch.stack([self.audio_embedding[k](samples[:, k]) for k in range(self.args.n_codebooks)], dim=1) # [B, K,1,D]
            assert samples_emb.shape == torch.Size([batch_size, self.args.n_codebooks, 1, self.args.d_model])
            samples_emb = samples_emb.sum(dim=1,keepdim=False) # [B,1,D]
            if sum(codebook_eog) == self.args.n_codebooks: # generation for the current span is done
                codebook_eog = [False] * self.args.n_codebooks
                num_gen.append(cur_num_gen)
                cur_num_gen = 0
                generated.append(cur_generated)
                cur_generated = [[] for _ in range(batch_size)]
                break
            else:
                assert samples_emb.shape == torch.Size((batch_size,1,self.args.d_model)), f"samples_emb.shape: {samples_emb.shape}"
            
            embedded_y = torch.cat([embedded_y, samples_emb], dim=1)
            y_input = self.audio_positional_embedding(embedded_y) # [B T D]
            # make attention mask and padding mask
            y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y.device)
            new_y_lens = torch.LongTensor([y_input.shape[1]]).to(y.device).repeat(batch_size)
            y_padding_mask = torch.full((batch_size,new_y_lens[0]), False).to(y.device)
        
        assert len(generated) == 1, f"len(generated): {len(generated)}"

        # revert the pattern
        flatten_gen = []
        for l, orig_span in enumerate(generated):
            span = torch.stack(orig_span, dim=0) # [T, K]
            span = span.transpose(1,0) # [K, T]
            assert span.shape[0] == self.args.n_codebooks, span.shape
            unshifted_span = []
            for j, s in enumerate(span):
                start_from = j
                end_at = - (self.args.n_codebooks - start_from)
                unshifted_span.append(s[start_from:end_at])
            unshifted_span = torch.stack(unshifted_span, dim=0)

            assert unshifted_span.shape[1] == num_gen[l] - self.args.n_codebooks, f"len(unshifted_spans[0]): {len(unshifted_span[0])}, num_gen[l]: {num_gen[l]}"

            flatten_gen.append(unshifted_span)
        assert len(flatten_gen) == 1, len(flatten_gen)
        
        # combine 
        res = [y[0], flatten_gen[0]]
        res = torch.cat(res, dim=1).unsqueeze(0) # [K, new_t] -> [1, K, new_T]

        expected_y_len = y_len + sum([item - self.args.n_codebooks for item in num_gen])
        assert res.shape == torch.Size((1, self.args.n_codebooks, expected_y_len)), f"res.shape: {res.shape}, expected_y_len: {expected_y_len}. y_len + sum([item - self.args.n_codebooks for item in num_gen]): {y_len} + {sum([item - self.args.n_codebooks for item in num_gen])}"

        if self.args.special_first:
            res = res - int(self.args.n_special)
            flatten_gen = flatten_gen - int(self.args.n_special)

        return res, flatten_gen[0].unsqueeze(0)
