from typing import Union
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as torch_f
import ipdb
from .news_encoder import NewsEncoder
from torch.autograd import Variable
# from src.utils import pairwise_cosine_similarity


class MPNR(nn.Module):
    r"""
    Implementation of Candidate-aware causal approach for news recommendation.
    """
    def __init__(self, news_encoder: NewsEncoder, use_category_bias: bool, num_context_codes: int,
                 context_code_dim: int, score_type: str, dropout: float, num_category: Union[int, None] = None,
                 category_embed_dim: Union[int, None] = None, category_pad_token_id: Union[int, None] = None,
                 category_embed: Union[Tensor, None] = None, pop_size: Union[Tensor, None] = -1, use_candidate_aware = False):
        r"""
        Initialization

        Args:
            news_encoder: NewsEncoder object.
            use_category_bias: whether to use Category-aware attention weighting.
            num_context_codes: the number of attention vectors ``K``.
            context_code_dim: the number of features in a context code.
            score_type: the ways to aggregate the ``K`` matching scores as a final user click score ('max', 'mean' or
                'weighted').
            dropout: dropout value.
            num_category: the size of the dictionary of categories.
            category_embed_dim: the size of each category embedding vector.
            category_pad_token_id: ID of the padding token type in the category vocabulary.
            category_embed: pre-trained category embedding.
        """
        super().__init__()
        self.news_encoder = news_encoder
        self.news_embed_dim = self.news_encoder.embed_dim
        self.use_category_bias = use_category_bias
        self.output_dim = self.news_embed_dim + 64
        self.pop_size = pop_size
        self.pop_encoder = nn.Embedding(num_embeddings=pop_size, embedding_dim=self.output_dim)

        self.category_encoder = CategoryEncoder(category_embed=category_embed, category_pad_token_id=category_pad_token_id, \
                                            dropout=dropout, output_dim=self.output_dim-self.news_embed_dim)
        self.category_embed_dim = self.output_dim-self.news_embed_dim
        self.score_type = score_type
        if use_candidate_aware == True:
            self.news_attn = TargetAwareAttention(self.news_embed_dim)
            self.cate_attn = TargetAwareAttention(self.category_embed_dim)
        else:
            self.news_attn = UserAttention(self.news_embed_dim)
            self.cate_attn = UserAttention(self.category_embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, title: Tensor, title_mask: Tensor, his_title: Tensor, his_title_mask: Tensor,
                his_mask: Tensor, sapo: Union[Tensor, None] = None, sapo_mask: Union[Tensor, None] = None,
                his_sapo: Union[Tensor, None] = None, his_sapo_mask: Union[Tensor, None] = None,
                category: Union[Tensor, None] = None, his_category: Union[Tensor, None] = None, 
                popularity: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            title: tensor of shape ``(batch_size, num_candidates, title_length)``.
            title_mask: tensor of shape ``(batch_size, num_candidates, title_length)``.
            his_title: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_title_mask: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_mask: tensor of shape ``(batch_size, num_clicked_news)``.
            sapo: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            sapo_mask: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            his_sapo: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            his_sapo_mask: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            category: tensor of shape ``(batch_size, num_candidates)``.
            his_category: tensor of shape ``(batch_size, num_clicked_news)``.

        Returns:
            tuple
                - multi_user_interest: tensor of shape ``(batch_size, num_context_codes, embed_dim)``
                - matching_scores: tensor of shape ``(batch_size, num_candidates)``
        """
        # ipdb.set_trace()
        candidate_repr = self.news_repr(title, title_mask, sapo, sapo_mask)

        history_repr = self.news_repr(his_title, his_title_mask, his_sapo, his_sapo_mask)

        # Click predictor
        matching_scores, dc_loss = self.compute_score(his_category, category, history_repr, candidate_repr, his_mask, popularity, 'train')

        return matching_scores, dc_loss

    def news_repr(self, title, title_mask, sapo, sapo_mask):
        # ipdb.set_trace()
        batch_size = title.shape[0]
        num_candidates = title.shape[1]
        # Representation of candidate news
        title = title.view(batch_size * num_candidates, -1)
        title_mask = title_mask.view(batch_size * num_candidates, -1)
        # sapo = sapo.view(batch_size * num_candidates, -1)
        # sapo_mask = sapo_mask.view(batch_size * num_candidates, -1)

        candidate_repr = self.news_encoder(title_encoding=title, title_attn_mask=title_mask, sapo_encoding=sapo,sapo_attn_mask=sapo_mask)
        candidate_repr = candidate_repr.view(batch_size, num_candidates, -1)
        return candidate_repr
    
    def compute_score(self, his_category, category, history_repr, candi_repr, his_mask, popularity, mode='test'):
        his_category_embed = self.category_encoder(his_category)
        candi_category_embed = self.category_encoder(category)

        user_fea = self.news_attn(candi_repr, history_repr, history_repr, his_mask)
        cate_fea = self.cate_attn(candi_category_embed, his_category_embed, his_category_embed, his_mask)
        
        news_repr = self.dropout(torch.cat((candi_repr, candi_category_embed), -1))
        user_repr = self.dropout(torch.cat((user_fea, cate_fea), -1))
        
        pop_emb = self.pop_encoder(popularity)
        news_repr_c = pop_emb * news_repr
        matching_scores = torch.mul(news_repr_c, user_repr).sum(-1)

        if mode == 'train':
            
            decorrelation_loss = self.decorrelation(news_repr, pop_emb)
            return matching_scores, decorrelation_loss
        else:
            return matching_scores

    def compute_score_with_no_popularity(self, his_category, category, history_repr, candi_repr, his_mask, popularity, mode='test'):
        his_category_embed = self.category_encoder(his_category)
        candi_category_embed = self.category_encoder(category)

        user_fea = self.news_attn(candi_repr, history_repr, history_repr, his_mask)
        cate_fea = self.cate_attn(candi_category_embed, his_category_embed, his_category_embed, his_mask)
        
        news_repr = self.dropout(torch.cat((candi_repr, candi_category_embed), -1))
        user_repr = self.dropout(torch.cat((user_fea, cate_fea), -1))
        
        news_repr_c =  news_repr
        matching_scores = torch.mul(news_repr_c, user_repr).sum(-1)
        return matching_scores

    def decorrelation(self, candidate_repr, pop_emb):
        candidate_repr_d = candidate_repr.detach()
        loss = Variable(torch.FloatTensor([0]).cuda())
        feature = pop_emb * candidate_repr_d
        # cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum=sum).cuda()
        # ipdb.set_trace()
        for i in range(feature.size()[-1]):
            cfeature = feature[:,:,i]
            cov1 = cov(cfeature)
            cov_matrix = cov1 * cov1
            loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
        # ipdb.set_trace()
        return loss


class CategoryEncoder(nn.Module):
    def __init__(self, category_embed, category_pad_token_id, dropout, output_dim):
        super().__init__()
        category_embedding = torch.from_numpy(np.load(category_embed)).float()
        self.category_encoder = nn.Embedding.from_pretrained(category_embedding, freeze=False,
                                                                       padding_idx=category_pad_token_id)
        self.category_embed_dim = category_embedding.shape[1]
        self.reduce_dim = nn.Linear(in_features=self.category_embed_dim, out_features=output_dim)
        self.cat_embed_dropout = nn.Dropout(dropout)

    def forward(self, categories):
        category_emb = self.category_encoder(categories)
        category_repr = self.reduce_dim(category_emb)
        category_repr = self.cat_embed_dropout(category_repr)
        return category_repr

class UserAttention(nn.Module):
    """Implementation of target-aware attention network"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.att_matrix = nn.Parameter(nn.init.xavier_uniform_(torch.empty(embed_dim, 1),
                                                                  gain=nn.init.calculate_gain('tanh')))
        # self.key_linear = nn.Linear(embed_dim, embed_dim)
    def forward(self, query: Tensor, key: Tensor, value: Tensor,  attn_mask: Tensor):
        r"""
        Forward propagation

        Args:
            query: tensor of shape ``(batch_size, num_candidates, embed_dim)``
            key: tensor of shape ``(batch_size, num_click, embed_dim)``
            value: tensor of shape ``(batch_size, num_click, num_context_codes)``

        Returns:
            tensor of shape ``(batch_size, num_candidates)``
        """
        # proj = torch_f.gelu(self.que_linear(query))
        # key = torch_f.gelu(self.key_linear(key))
        proj = self.att_matrix.unsqueeze(0)
        key = key
        # ipdb.set_trace()
        weights = torch.matmul(key, proj).permute(0,2,1)
        weights = weights.masked_fill_(~attn_mask.unsqueeze(dim=1), 1e-30)
        weights = torch_f.softmax(weights, dim=2)
        outputs = torch.matmul(weights, value)
        
        return outputs

class TargetAwareAttention(nn.Module):
    """Implementation of target-aware attention network"""
    def __init__(self, embed_dim: int):
        r"""
        Initialization

        Args:
            embed_dim: The number of features in query and key vectors
        """
        super().__init__()
        # self.key_linear = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        # self.que_linear = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,  attn_mask: Tensor):
        r"""
        Forward propagation

        Args:
            query: tensor of shape ``(batch_size, num_candidates, embed_dim)``
            key: tensor of shape ``(batch_size, num_click, embed_dim)``
            value: tensor of shape ``(batch_size, num_click, num_context_codes)``

        Returns:
            tensor of shape ``(batch_size, num_candidates)``
        """
        # proj = torch_f.gelu(self.que_linear(query))
        # key = torch_f.gelu(self.key_linear(key))
        proj = query
        key = key
        weights = torch.matmul(proj, key.permute(0, 2, 1))
        weights = weights.masked_fill_(~attn_mask.unsqueeze(dim=1), 1e-30)
        weights = torch_f.softmax(weights, dim=2)
        outputs = torch.matmul(weights, value)
        
        return outputs

def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
        # import ipdb
        # ipdb.set_trace()

    return res

