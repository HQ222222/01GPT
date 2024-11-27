import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig
from attention_v1 import MultiHeadAttention
from transformers.modeling_outputs import CausalLMOutputWithPast

MODEL_CONFIG = {
    "vocab_size": 32000, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean)/(torch.sqrt(var + self.eps))
        return self.scale * x_norm + self.shift


class FeedForward(nn.Module):
    def __init__(self, emb_dim:int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )
    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.atten = MultiHeadAttention(
            dim_in=kwargs['emb_dim'], 
            dim_out=kwargs['emb_dim'], 
            context_length=kwargs['context_length'], 
            num_heads=kwargs['n_heads'], 
            dropout_rate=kwargs['drop_rate'],
            qkv_bias=kwargs['qkv_bias'])
        self.ffn = FeedForward(kwargs['emb_dim'])
        self.drop = nn.Dropout(kwargs['drop_rate'])
        self.layernorm1 = LayerNorm(kwargs['emb_dim'])
        self.layernorm2 = LayerNorm(kwargs['emb_dim'])

    def forward(self, x, pos_cis, attention_mask=None, use_kv_cache=False, past_kv=None):
        shortcut = x
        x = self.layernorm1(x)
        x, past_kv = self.atten(x, pos_cis, attention_mask, use_kv_cache, past_kv)
        x = self.drop(x)
        x = x + shortcut

        shortcut = x
        x = self.layernorm2(x)
        x = self.ffn(x)
        x = self.drop(x)
        x = x + shortcut

        return x, past_kv

def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis

def attention_mask_to_4d(attention_mask, num_heads):
    batch_size, seq_len = attention_mask.size()
    # expand dimensions to (batch, 1, 1, seq_len)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    # repleat to match (batch, 1, seq_len, seq_len)
    attention_mask = attention_mask.repeat(1, num_heads, seq_len, 1)
    # invert the attention mask，where the position of value 1 will be masked with -inf
    return (1 - attention_mask)

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_length = cfg['context_length']
        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        
        # self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.decode_layers = nn.Sequential(*[
            TransformerBlock(cfg) for _ in range(cfg['n_layers'])
        ])
        pos_cis = precompute_pos_cis(cfg['emb_dim'] // cfg['n_heads'], cfg['context_length'])
        self.register_buffer("pos_cis", pos_cis, persistent=False)
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'])

    def forward(self, inputs):
        b, seq_len = inputs.shape
        pos_cis = self.pos_cis[:seq_len]
        x = self.token_emb(inputs)
        # pos_embeddings = self.pos_emb(torch.arange(seq_len, device=inputs.device))
        # x  = token_embeddings + pos_embeddings
        x = self.drop_emb(x)
        for i, block in enumerate(self.decode_layers):
            x = block(x, pos_cis)
        # x = self.decode_layers(x, pos_cis)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
 
    def generate(self, token_ids, max_new_tokens=256, stop_token_id=-1):
        assert isinstance(max_new_tokens, int) and max_new_tokens > 0
        eos_reached = torch.zeros(len(token_ids), dtype=torch.bool, device=token_ids.device)
        for _ in range(max_new_tokens):
            # 如果生成序列过程中超出上下文长度，则由后往前截取context_length个token。
            context_ids = token_ids[:, -self.context_length:]  
            with torch.no_grad():
                output = self(context_ids)  # shape: batch, n_tokens, vocab_size

            # 只取每个序列最后一个token的输出向量作为logits, shape变为: batch, vocab_size
            logits = output[:, -1, :]        
            # 使用softmax函数将logits转换为下一个token的概率分布，shape仍是: batch, vocab_size
            probs = torch.softmax(logits, dim=-1)   
            # 取概率最大的作为next_token_ids，形状变为：batch, 1
            next_token_ids = torch.argmax(probs, dim=-1, keepdim=True)
            # 将next_token_id连接到下一个token的结尾， 形状变为：batch, n_tokens+1
            token_ids = torch.cat((token_ids, next_token_ids), dim=1)
            # 更新 eos_reached
            eos_reached |= (next_token_ids.squeeze(-1) == stop_token_id)
            if eos_reached.all():
                break

        return token_ids

class MiniGPTConfig(PretrainedConfig):
    # 每个模型都必须有一个独特的model_type，否则会报"Should have a `model_type` key in its config.json"
    model_type = "minigpt"

    def __init__(self, 
        context_length: int = 1024,
        vocab_size: int = 32000,
        emb_dim: int = 768,
        drop_rate: float = 0.1,
        n_layers: int = 12,
        n_heads: int = 12,
        qkv_bias: bool = False,
        **kwargs,
    ):
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.drop_rate = drop_rate
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        super().__init__(**kwargs)

class MiniGPT(PreTrainedModel):
    config_class = MiniGPTConfig

    @classmethod  
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):  
        # 自定义加载逻辑，通常包括加载权重和配置  
        model = super(MiniGPT, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)  
        return model  

    def __init__(self, config: MiniGPTConfig):
        super().__init__(config)
        self.context_length = config.context_length
        self.num_heads = config.n_heads
        self.n_layers = config.n_layers
        self.token_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.decode_layers = nn.Sequential(*[
            TransformerBlock(**(config.to_dict())) for _ in range(config.n_layers)
        ])

        pos_cis = precompute_pos_cis(config.emb_dim // config.n_heads, config.context_length)
        self.register_buffer("pos_cis", pos_cis, persistent=False)
        self.final_norm = LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size)
        self.out = CausalLMOutputWithPast()

    def forward(self, 
                inputs:Optional[torch.Tensor]=None, 
                attention_mask:Optional[torch.Tensor]=None, 
                use_kv_cache=False,
                past_kvs=None,
                return_dict=False,
                **kwargs):
        if not past_kvs:
            past_kvs = [None for _ in range(self.n_layers)]
        if 'input_ids' in kwargs:
            inputs = kwargs['input_ids']
        assert isinstance(inputs, torch.Tensor), f"expect torch.Tensor, but got{type(inputs)}"
        b, seq_len = inputs.shape
        pos_cis = self.pos_cis[:seq_len]
        x = self.token_emb(inputs)
        x = self.drop_emb(x)
        
        # 支持注意力掩码计算
        if attention_mask != None:
            assert isinstance(attention_mask, torch.Tensor), f"expect torch.Tensor, but got{type(attention_mask)}"
            assert attention_mask.size() == inputs.size(), f"size of inputs {inputs.size()} and attention_mask {attention_mask.size()} must be the same."
            attention_mask = attention_mask_to_4d(attention_mask, self.num_heads)

        for i, block in enumerate(self.decode_layers):
            x, past_kvs[i] = block(x, pos_cis, attention_mask, use_kv_cache, past_kvs[i])

        x = self.final_norm(x)
        logits = self.out_head(x)
        if not return_dict:
            return logits
        
        self.out.__setitem__('logits', logits)
        self.out.__setitem__('past_kvs', past_kvs)
        return self.out
 
    @torch.inference_mode()
    def generate(self, input_ids, attention_mask=None, max_length=512, pad_token_id=-1, **kwargs):
        assert isinstance(max_length, int) and max_length > 0
        eos_reached = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
        use_kv_cache = True if 'use_kv_cache' not in kwargs else kwargs['use_kv_cache']
        past_kvs = None
        return_dict=True
        del kwargs['use_kv_cache']
        print("use_kv_cache: ", use_kv_cache)
         
        for _ in range(max_length):
            # 如果生成序列过程中超出上下文长度，则由后往前截取context_length个token。
            context_ids = input_ids[:, -self.context_length:]  
            with torch.no_grad():
                output = self(context_ids, attention_mask, use_kv_cache, past_kvs, return_dict, **kwargs)  # shape: batch, n_tokens, vocab_size
            past_kvs = output["past_kvs"] if use_kv_cache else None
            # 只取每个序列最后一个token的输出向量作为logits, shape变为: batch, vocab_size
            logits = output["logits"][:, -1, :]        
            # 使用softmax函数将logits转换为下一个token的概率分布，shape仍是: batch, vocab_size
            probs = torch.softmax(logits, dim=-1)   
            # 取概率最大的作为next_input_ids，形状变为：batch, 1
            next_token_ids = torch.argmax(probs, dim=-1, keepdim=True)
            # 将next_token_id连接到下一个token的结尾， 形状变为：batch, n_tokens+1
            input_ids = torch.cat((input_ids, next_token_ids), dim=1)
            # 更新 eos_reached
            eos_reached |= (next_token_ids.squeeze(-1) == pad_token_id)
            if eos_reached.all():
                break

        return input_ids
    
def generate_sequence(model, token_ids, max_new_tokens, context_length):
    # TODO 所有序列都结束时，提前退出循环
    for _ in range(max_new_tokens):
        # 如果生成序列过程中超出上下文长度，则由后往前截取context_length个token。
        context_ids = token_ids[:, -context_length:]  

        with torch.no_grad():
            output = model(context_ids)  # shape: batch, n_tokens, vocab_size

        # 只取每个序列最后一个token的输出向量作为logits, shape变为: batch, vocab_size
        logits = output[:, -1, :]        
        # 使用softmax函数将logits转换为下一个token的概率分布，shape仍是: batch, vocab_size
        probs = torch.softmax(logits, dim=-1)   
        # 取概率最大的作为next_token_ids，形状变为：batch, 1
        next_token_ids = torch.argmax(probs, dim=-1, keepdim=True)
        # 将next_token_id连接到下一个token的结尾， 形状变为：batch, n_tokens+1
        token_ids = torch.cat((token_ids, next_token_ids), dim=1)

    return token_ids