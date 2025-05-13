# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


class DefaultConfig:
    title_size = 32
    abs_size = 64
    his_size = 50
    ratio_K = 4

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

opt = DefaultConfig()
