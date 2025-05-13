# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


class DefaultConfig:

        # = Common = args
    norm = ''
    loss = 'nce'
    gamma = 0.01
    pop_size = 10
    model = 'MPNR'
    tokenizer = 'roberta'
    pretrained_tokenizer = "../../pre-train-model/nb-bert-base"
    max_title_length = 32
    max_sapo_length = 64
    his_length = 50
    seed = 42
    use_gpu = True
    gpu_id = 1
    multi_gpu = False
    gpu_ids = []
    dt = 'small'
    user_dim = 300
    attn_dim = 512
    word_dim = 300
    word_att = True
    news_att = True
    disagreement = False
    enc_method = 'transformer'
    enc_user = 'self-attention'

    # = Data = args

    print_step = 5000

    # = Model = args
    pretrained_embedding = "../../pre-train-model/nb-bert-base"
    apply_reduce_dim = True
    use_sapo = False
    decorrelation = False
    word_embed_dim = 200
    category_embed_dim = 100
    
    combine_type = 'linear'
    num_context_codes = 1
    context_code_dim = 200
    score_type = 'weighted'
    dropout = 0.1
    metrics = ['auc', 'mean_mrr', 'ndcg@5;10']
    # = Train = args
    npratio = 4
    train_batch_size = 16
    batch_size = 64
    eval_batch_size = 1
    dataloader_drop_last = True
    dataloader_num_workers = 4
    dataloader_pin_memory = True
    gradient_accumulation_steps = 16
    epochs = 5
    learning_rate = 1e-5
    warmup_ratio = 0.05
    max_grad_norm = 1.0
    weight_decay = 0.01
    logging_steps = 100
    evaluation_info = 'metrics'
    eval_steps = 1600
    freeze_transformer = False
    lstm_num_layers = 0
    lstm_dropout = 0.2
    use_category_bias = True

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)
        self.w2v_path = f"./data/{self.dt}/train/embedding_{self.dt}.npy"
        self.e2v_path = './pro_data/raw/entity_embedding.npy'
        self.news_title_index = f"./data/{self.dt}/train/news_title_index.npy"
        self.news_abs_index = f"./data/{self.dt}/train/news_abs_index.npy"
        self.category_embed = f'./data/{self.dt}/train/embedding_cat.npy'
        if self.dt == 'large':
            self.eval_steps = self.eval_steps * 2
        if self.use_sapo:
            self.train_batch_size = int(self.train_batch_size/4)

opt = DefaultConfig()
