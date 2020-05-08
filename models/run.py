from .baseCF import UserCF, ItemCF, LFM
import os
# from nade import NADE_CF
from .lib import Pipeline,CONFIG
from .sentiment_rate import UnionModel, TagRepresentation, SentimentRating, NeuralCF, TripleRepresentationV2
import argparse


def test():
    pipeline = Pipeline(UserCF, remake_dataset=False)
    pipeline.run(use_iif_similarity=False)
    pipeline = Pipeline(UserCF, model_type='UserCF_IIF')
    pipeline.run(use_iif_similarity=True)

    pipeline = Pipeline(ItemCF, remake_dataset=False)
    pipeline.run(use_iuf_similarity=False)
    pipeline = Pipeline(ItemCF, model_type='ItemCF_IIF')
    pipeline.run(use_iuf_similarity=True)

    pipeline = Pipeline(LFM)
    pipeline.run()


if __name__ == '__main__':
    # test()
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--ncf', default=False, action='store_true')
    parser.add_argument('--ncf_pre', default=False, action='store_true')
    parser.add_argument('--triple', default=False, action='store_true')
    parser.add_argument('--triple_pre', default=False, action='store_true')
    parser.add_argument('--tag', default=False, action='store_true')
    parser.add_argument('--tag_pre', default=False, action='store_true')
    parser.add_argument('--triplev2', default=False, action='store_true')
    parser.add_argument('--triplev2_pre', default=False, action='store_true')
    parser.add_argument('--name', type=str, default=None, help='saved model name')
    args = parser.parse_args()
    models = [NeuralCF, SentimentRating, TagRepresentation, TripleRepresentationV2]
    configs = [None, None, None, None]
    model_masks = (args.ncf, args.triple, args.tag, args.triplev2)
    pretrain_masks = (args.ncf_pre, args.triple_pre, args.tag_pre, args.triplev2_pre)
    pretrain_dirs = ('ncf', 'triple', 'tag', 'triplev2_ens')
    if args.name is None:
        args.name = '_'.join(
            [x + '_pre' if pretrain_masks[i] else x for i, x in enumerate(pretrain_dirs) if model_masks[i]])
    pipeline = Pipeline(UnionModel, args.name)
    pipeline.run(models=models, model_masks=model_masks, pretrain_masks=pretrain_masks, pretrain_dirs=pretrain_dirs,
                 configs=configs, lr=args.lr)
    print('done')
    with open(os.path.join(CONFIG.models_folder, 'logs/%s.log' % args.name), 'a', encoding='utf8') as f:
        print(args, file=f)
