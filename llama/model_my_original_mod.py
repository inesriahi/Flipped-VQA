# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List, Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Embedding, Linear
import torch

from llama.tokenizer import Tokenizer

@dataclass
class ModelArgs: # check how they are set again in llama_vqa.py
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    adapter_len: int=10
    adapter_layer: int=30

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.max_feats = args.max_feats

        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.gate1 = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))
        self.gate2 = torch.nn.Parameter(torch.ones(1, self.n_local_heads, 1, 1) * -args.bias)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, adapter_len).to(mask)
            mask = torch.cat([extra_mask, mask], dim=-1)
        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        if adapter is not None:            
            adapter_scores = F.softmax(scores[..., :adapter_len].float(), dim=-1).type_as(xq) * self.gate1.tanh().half()
            if video_start is not None:
                vt_scores = scores[..., adapter_len:].clone()
                vt_scores[:, :, video_start + self.max_feats:, video_start:video_start + self.max_feats] = \
                    vt_scores[:, :, video_start + self.max_feats:, video_start:video_start + self.max_feats] + self.gate2.half()
                vt_scores = F.softmax(vt_scores.float(), dim=-1).type_as(xq)
            else:
                vt_scores = F.softmax(scores[..., adapter_len:], dim=-1)
            scores = torch.cat([adapter_scores, vt_scores], dim=-1)
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CrossAttentionModule(nn.Module):
  def __init__(self, feature_dim):
    super().__init__()
    self.query = nn.Linear(feature_dim, feature_dim)
    self.key = nn.Linear(feature_dim, feature_dim)
    self.value = nn.Linear(feature_dim, feature_dim)
    self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))

  def forward(self, video_features, audio_features):
    # Convert scale to the same device and dtype as video_features
    scale = self.scale.to(video_features.device, dtype=video_features.dtype)
    audio_features = audio_features.float()

    # Perform operations in half precision
    Q = self.query(video_features)
    K = self.key(audio_features)
    V = self.value(audio_features)

    QK_T = torch.matmul(Q, K.transpose(-2, -1))  # (bs, seq_len = 10, 1)
    QK_T = QK_T / scale

    attention_scores = F.softmax(QK_T, -1)
    
    res = torch.matmul(attention_scores, V)  # (bs, seq_len = 10, feature_dim = 768)
    return res


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, video_start)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, args):
        super().__init__()
        params.max_feats = args.max_feats
        params.bias = args.bias
        self.args = args
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_feats = args.max_feats

        self.tokenizer = Tokenizer(model_path=f'{args.llama_model_path}./tokenizer.model', args=args)
        self.eos_id = self.tokenizer.eos_id
        self.answer_token_id = self.tokenizer.a_token_id

        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.adapter_query = Embedding(params.adapter_len * params.adapter_layer, params.dim)
        if args.audio and args.audio_only:
            self.audio_proj = Linear(1024, params.dim, bias=False).half()
        
        elif args.audio and args.audio_merge == 'concat': # audio and video and the method is to concat
            self.visual_proj = Linear(768 + 1024, params.dim, bias=False).half() # since audio features are concatenated with video features
        
        elif args.audio and args.audio_merge in 'sum':
            self.audio_proj = Linear(1024, params.dim, bias=False).half()
            self.visual_proj = Linear(768, params.dim, bias=False).half()
        
        elif args.audio and args.audio_merge == 'attention': # from dataloader, the shape of audio here is (1, 1024) assuming that the audio feature model is imagebind
            self.audio_proj = Linear(1024, 768, bias=False).half()
            self.visual_proj = Linear(768, params.dim, bias=False).half()
        else: # video only
            self.visual_proj = Linear(768, params.dim, bias=False).half()
            self.visual_proj = self.visual_proj.half()
        
        if args.audio and args.audio_merge == 'attention':
            self.video_audio_cross_attn = CrossAttentionModule(768).float()

        self.temporal_emb = Embedding(self.max_feats, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.vqa_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.vaq_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.qav_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.inference_criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

        self.video_label = torch.arange(1, self.max_feats)
        self.tau = args.tau

    def forward(self, data, inference=False):
        if not self.args.audio: # video only
            video = data['video'].cuda()
        elif not self.args.audio_only: # video and audio
            video = data['video'].cuda()
            audio = data['audio'].cuda().half()
        else: # audio only
            audio = data['audio'].cuda().half()
        # shuffled_video = data['shuffled_video_frames'].cuda()
        # shuffeld_frame_indecies = data['shuffeld_frame_indecies'].cuda()
        vqa_id, vaq_id, qav_id = data['text_id']['vqa'].cuda(), data['text_id']['vaq'].cuda(), data['text_id']['qav'].cuda()
        vqa_label, vaq_label, qav_label = data['label']['vqa'].cuda(), data['label']['vaq'].cuda(), data['label']['qav'].cuda()
        #vqa_label_mask, vaq_label_mask, qav_label_mask = data['label_mask']['vqa'].cuda(), data['label_mask']['vaq'].cuda(), data['label_mask']['qav'].cuda()
        vqa_video_start, vaq_video_start, qav_video_index = data['video_start']['vqa'][0], data['video_start']['vaq'][0], data['video_index']['qav'].cuda()
        
        # if self.args.is_generation_task and inference:
        #     original_vqa_id = vqa_id.clone()
        #     original_vqa_id = vqa_id.clone()
        #     original_qav_id = qav_id.clone()
        #     vqa_id = vqa_id[:,0:1,:]
        #     vaq_id = vaq_id[:,0:1,:]
        #     qav_id = qav_id[:,0:1,:]

        bsz, n_options, seqlen = vqa_id.shape # in case of training, n_options is 1, in case of val or test, n_options is len(answer_mapping) with same text repeated except for the encoded answer
        if self.args.debug and not inference:
            for i in range(n_options):
                print("Decoded VQA label in model:", self.tokenizer.decode(vqa_label[:, i, :].tolist()))
                print("Decoded VQA id in model::", self.tokenizer.decode(vqa_id[:, i, :].tolist()))
                print("Decoded VAQ label in model:", self.tokenizer.decode(vaq_label[:, i, :].tolist()))
                print("Decoded VAQ id in model::", self.tokenizer.decode(vaq_id[:, i, :].tolist()))
        if self.args.debug:
            print("vqa_id.shape:", vqa_id.shape)
        vqa_id, vaq_id = vqa_id.reshape(-1, seqlen), vaq_id.reshape(-1, seqlen) #(bsz * n_options, seqlen)
        vqa_label, vaq_label = vqa_label.reshape(-1, seqlen), vaq_label.reshape(-1, seqlen)
        vqa_label, vaq_label = vqa_label[:, 1:].flatten(), vaq_label[:, 1:].flatten()
        
        qav_id = qav_id.reshape(-1, seqlen)
        qav_label = qav_label.reshape(-1, seqlen)
        qav_video_mask = qav_label.ge(0)
        qav_label = qav_label[:, 1:].flatten()
        
        
        with torch.no_grad():
            vqa_h = self.tok_embeddings(vqa_id) #(bsz * n_options, seqlen, dim)
            # print("Before fill vqa_h.shape:", vqa_h.shape)
            # vqa_h = vqa_h.masked_fill(vqa_label_mask.unsqueeze(-1).bool(), -1)
            # print("After fill vqa_h.shape:", vqa_h.shape)
            
            if self.args.vaq and not inference:
                vaq_h = self.tok_embeddings(vaq_id)
                # vaq_h = vaq_h.masked_fill(vaq_label_mask.unsqueeze(-1).bool(), -1)
            
            if self.args.qav and not inference:
                qav_h = self.tok_embeddings(qav_id)
            
        freqs_cis = self.freqs_cis.to(vqa_h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=vqa_h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(vqa_h)
        start_pos = 0
        vaq_loss, qav_loss = torch.tensor([0]).cuda(), torch.tensor([0]).cuda()
        
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)

        if self.args.audio and self.args.audio_only:
            _video_feature = self.audio_proj(audio) # (10, params.dim)
        
        elif self.args.audio and self.args.audio_merge == 'concat': # audio and video and the method is to concat
            concatted = torch.cat([video, audio], dim = -1)  # (10, 768 + 1024) # since audio features are concatenated with video features
            _video_feature = self.visual_proj(concatted)
        
        elif self.args.audio and self.args.audio_merge in 'sum':
            _video_feature = self.audio_proj(audio).half() + self.visual_proj(video).half() # (10, params.dim)
        
        elif self.args.audio and self.args.audio_merge == 'attention': # from dataloader, the shape of audio here is (1, 1024) assuming that the audio feature model is imagebind
            audio_features = self.audio_proj(audio) # (1,768)
            _video_feature = self.video_audio_cross_attn(video, audio_features) # (10, 768)

            _video_feature = self.visual_proj(_video_feature).half() # (10, params.dim)
        else: # video only
            _video_feature = self.visual_proj(video)       
            
        if inference:
            _video_feature = _video_feature.unsqueeze(1).repeat(1, n_options, 1, 1).view(-1, _video_feature.shape[-2], _video_feature.shape[-1])
        video_feature = (_video_feature + self.temporal_emb.weight[None, :, :]).half()
        
        vqa_h = vqa_h.clone()
        vqa_h[:, vqa_video_start:vqa_video_start+self.max_feats] = video_feature

        
        if self.args.vaq and not inference:
            vaq_h = vaq_h.clone()
            vaq_h[:, vaq_video_start:vaq_video_start+self.max_feats] = video_feature
            
        if self.args.qav and not inference:
            qav_h = qav_h * ~qav_video_mask[..., None]
            qav_h.scatter_add_(1, qav_video_index[..., None].repeat(1, 1, self.params.dim), video_feature)
        
        for i, layer in enumerate(self.layers[-1 * self.adapter_layer:]):
            vqa_h = layer(vqa_h, start_pos, freqs_cis, mask, adapter[i].half(), vqa_video_start)
            
            if self.args.vaq and not inference:
                vaq_h = layer(vaq_h, start_pos, freqs_cis, mask, adapter[i].half(), vaq_video_start)
            
            if self.args.qav and not inference:
                qav_h = layer(qav_h, start_pos, freqs_cis, mask, adapter[i].half(), None)
    
        vqa_h = self.norm(vqa_h)
        vqa_output = self.output(vqa_h)
        vqa_output = vqa_output[:, :-1, :].reshape(-1, self.vocab_size) # (bsz x num_options x (seqlen - 1), vocab_size)
        vqa_loss = self.vqa_criterion(vqa_output, vqa_label)
        
        if self.args.vaq and not inference:
            vaq_h = self.norm(vaq_h)
            vaq_output = self.output(vaq_h)
            vaq_output = vaq_output[:, :-1, :].reshape(-1, self.vocab_size)
            vaq_loss = self.vaq_criterion(vaq_output, vaq_label)
            
        if self.args.qav and not inference:
            qav_h = self.norm(qav_h)
            qav_output = torch.bmm(qav_h[:, :-1].float(), _video_feature.transpose(1, 2).float()).reshape(-1, self.max_feats)
            qav_loss = self.qav_criterion(qav_output / self.tau, qav_label)
            if self.args.debug:
                print(f"QAV Label inside model, first sequence: {qav_label[:128]}")
        
        if inference:
            individual_losses = self.inference_criterion(vqa_output, vqa_label)
            individual_losses = individual_losses.reshape(bsz, n_options, -1)

            if self.args.debug and not self.args.is_generation_task:
                                # Extracting the most likely token sequence from the output
                vqa_output_reshaped_argmax = torch.argmax(vqa_output, dim=-1).reshape(bsz, -1, (seqlen - 1))
                vqa_output_tokens = vqa_output_reshaped_argmax[:, 0, :]  # Shape: (batch_size, (seqlen - 1))
                
                # More debugging output
                if self.args.debug:
                    print("vqa_output_tokens.shape in val classification", vqa_output_tokens.shape)
                    # Printing the first few contents of the output tokens for inspection
                    for i in range(3):
                        print(f"vqa_output_tokens content {i+1}:", vqa_output_reshaped_argmax[:, i, :])
                    for batch_item in vqa_output_reshaped_argmax:
                        for i in range(n_options):
                            print("decoded output in classification:", self.tokenizer.decode(batch_item[i].tolist()))

            if self.args.is_generation_task:
                if self.args.debug:
                    print("vqa_output.shape", vqa_output.shape)  # (bsz x num_options x (seqlen - 1), vocab_size)
                
                # Extracting the most likely token sequence from the output
                vqa_output_reshaped_argmax = torch.argmax(vqa_output, dim=-1).reshape(bsz, -1, (seqlen - 1))
                vqa_output_tokens = vqa_output_reshaped_argmax[:, 0, :]  # Shape: (batch_size, (seqlen - 1))
                
                # More debugging output
                if self.args.debug:
                    print("vqa_output_tokens.shape", vqa_output_tokens.shape)
                    # Printing the first few contents of the output tokens for inspection
                    for i in range(3):
                        print(f"vqa_output_reshaped_argmax content {i+1}:", vqa_output_reshaped_argmax[:, i, :])
                    for batch_item in vqa_output_reshaped_argmax:
                        for i in range(n_options):
                            print("decoded output:", self.tokenizer.decode(batch_item[i].tolist()))

                # Creating a mask to identify the answer part in the sequence
                vqa_placeholder_mask = vqa_label.reshape(bsz, -1, seqlen-1)[:, 0, :] != 0  # Non-zero values represent the answer part ([bsz, seqlen-1])
                vqa_id_reshaped = vqa_id.reshape(bsz, n_options, seqlen)
                
                extracted_answers_per_batch = []
                for batch_item in vqa_id_reshaped:
                    start_index = batch_item[0].tolist().index(self.answer_token_id) + 5 # "Answer" token id and skip 5 tokens for the ": The answer is "
                    extracted_answers = []
                    if start_index is not None:
                        for choice in batch_item:
                            eos_index = choice[start_index:].tolist().index(self.eos_id) + start_index if self.eos_id in choice[start_index:].tolist() else len(choice)
                            answer = self.tokenizer.decode(choice[start_index:eos_index].tolist())
                            extracted_answers.append(answer)
                            if self.args.debug:
                                print("Extracted answer:", answer)
                    else:
                        extracted_answers = [''] * n_options # Default to empty string if no answer part is found
                    extracted_answers_per_batch.append(extracted_answers)
                choice_embeddings_agg = []
                for extracted_answers in extracted_answers_per_batch:
                    # Encode each answer and convert to tensor
                    encoded_answers = [torch.tensor(self.tokenizer.encode(answer, bos=False, eos=False), dtype=torch.long) for answer in extracted_answers]
                    # Pad the sequences so they all have the same length
                    padded_encoded_answers = torch.nn.utils.rnn.pad_sequence(encoded_answers, batch_first=True, padding_value=0)
                    # Move to the appropriate device
                    padded_encoded_answers = padded_encoded_answers.to(self.tok_embeddings.weight.device)
                    # Get the embeddings
                    answer_embeddings = self.tok_embeddings(padded_encoded_answers)
                    # Aggregate along sequence length
                    answer_embeddings_agg = torch.mean(answer_embeddings, dim=1)
                    choice_embeddings_agg.append(answer_embeddings_agg)
                
                choice_embeddings_agg = torch.stack(choice_embeddings_agg)  # Shape: (bsz, n_options, embed_size)
                        
                if self.args.debug:
                    print("vqa_placeholder_mask.shape", vqa_placeholder_mask.shape)
                    vqa_id_reshaped = vqa_id.reshape(bsz, n_options, seqlen)
                    decoded_choices_output = [self.tokenizer.decode(choice.tolist()) for choice in vqa_id_reshaped.view(-1, seqlen)]
                    print("Decoded choices:", decoded_choices_output)

                    # New debug prints for the extracted answers
                    for batch_index, extracted_answers in enumerate(extracted_answers_per_batch):
                        print(f"Batch {batch_index + 1} extracted answers:", extracted_answers)

                    # Debug prints for the embeddings of the extracted answers
                    for batch_index, embeddings in enumerate(choice_embeddings_agg):
                        print(f"Batch {batch_index + 1} choice embeddings shape:", embeddings.shape)

                    # Additional debug prints for reshaped labels and IDs
                    for i in range(n_options):
                        vqa_label_reshaped = vqa_label.reshape(bsz, -1, seqlen-1)[:, i, :]
                        vqa_id_reshaped_option = vqa_id_reshaped[:, i, :]
                        print(f"vqa_id_reshaped_option {i+1}.shape", vqa_id_reshaped_option.shape)

                # Filtering the output tokens based on the placeholder mask
                filtered_vqa_output_tokens = [tokens[mask] for tokens, mask in zip(vqa_output_tokens, vqa_placeholder_mask)]
                
                # Initializing a list to store output embeddings
                vqa_output_embed = []
                if self.args.debug:
                    decoded_sequences_output = []

                # Process each set of output tokens
                for output_tokens in filtered_vqa_output_tokens:
                    if self.args.debug:
                        # Decoding the output tokens for debugging
                        decoded_sequences_output.append(self.tokenizer.decode(output_tokens.tolist()))
                    
                    # Identifying the end of the answer using 'eos_id' and extracting relevant part
                    eos_index = (output_tokens == self.eos_id).nonzero(as_tuple=True)[0]
                    if eos_index.numel() > 0:
                        output_tokens = output_tokens[:eos_index[0]]
                    
                    # Embedding the tokens and adding to the list
                    vqa_output_embed.append(self.tok_embeddings(output_tokens))
                
                
                if self.args.debug:
                    print("decoded sequences output:", decoded_sequences_output)


                # Aggregating the embeddings along the sequence length
                vqa_output_embed_agg = torch.stack([torch.mean(embed, dim=0) if embed.numel() > 0 else torch.zeros(embed.size(1)) for embed in vqa_output_embed])
                
                
                if self.args.debug:
                    print("vqa_output_embed_agg.shape", vqa_output_embed_agg.shape)
                    print("choice_embeddings_agg.shape", choice_embeddings_agg.shape)
                    
                # Calculate similarity for each instance in the batch considering options
                most_similar_indices, similarities = self.find_most_similar(vqa_output_embed_agg, choice_embeddings_agg)

                return individual_losses, most_similar_indices
            else: # if not generation task
                return individual_losses
        else:
            return vqa_loss, vaq_loss, qav_loss
        
    def find_most_similar(self, output_embeddings: torch.Tensor, choice_embeddings: torch.Tensor):
        """
        Find the most similar text for each instance in the batch considering multiple options.
        output_embeddings: Tensor of shape (batch_size, embed_size)
        choice_embeddings: Tensor of shape (batch_size, n_options, embed_size)
        
        Returns:
            max_indices: Tensor of shape (batch_size, ) with each element representing the most similar option to that batch instance
            similarities: Tensor of shape (batch_size, n_options) with each row representing the similarity scores of that batch instance to the other options given for that instance
        """
        batch_size = output_embeddings.size(0)

        # Normalize the embeddings to unit vectors
        output_embeddings_norm = torch.nn.functional.normalize(output_embeddings, p=2, dim=1)
        choice_embeddings_norm = torch.nn.functional.normalize(choice_embeddings, p=2, dim=2)

        # Compute the cosine similarities
        # (batch_size, n_options), each row represents all similarities between that instance in the batch and the other options that correspond to that instance
        similarities = torch.bmm(choice_embeddings_norm, output_embeddings_norm.unsqueeze(-1)).squeeze(-1)

        max_indices = torch.argmax(similarities, dim=1)

        if self.args.debug:
            print("Output embeddings in Cosine Similarity:", output_embeddings)
            print("Output embeddings shape in Cosine Similarity:", output_embeddings.shape)
            print("Choice embeddings in Cosine Similarity:", choice_embeddings)
            print("Choice embeddings shape in Cosine Similarity:", choice_embeddings.shape)

        return max_indices, similarities  # (batch_size, ), (batch_size, n_options)