from models.single_rate_pred import UnionModel, RNN, ProfileEmbedding
from models.lib import Pipeline, CONFIG
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--profile', default=False, action='store_true')
    parser.add_argument('--rnn', default=False, action='store_true')
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--name', type=str, default=None, help='saved model name')
    args = parser.parse_args()
    models = [ProfileEmbedding, RNN]
    model_masks = (args.profile, args.hidden)
    configs = [None, {'max_length': 30, 'num_hidden': args.hidden, 'hidden_layer_num': args.layer}]
    pretrain_masks = (False, False)
    pretrain_dirs = ('profile', 'rnn')
    name = args.name
    lr = args.lr
    if name is None:
        name = '_'.join([x for m, x in zip(model_masks, pretrain_dirs) if m])
        name += '_lr%f' % lr
        if model_masks[1]:
            name += '_%d_%d' % (args.layer, args.hidden)
    pipeline = Pipeline(UnionModel, name, task='single')
    pipeline.run(models=models, model_masks=model_masks, pretrain_masks=pretrain_masks, pretrain_dirs=pretrain_dirs,
                 configs=configs, lr=lr)
    print('done')
    with open(os.path.join(CONFIG.models_folder, 'logs/%s.log' % name), 'a', encoding='utf8') as f:
        print(configs, file=f)
