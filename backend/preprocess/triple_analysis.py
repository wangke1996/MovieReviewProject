from backend.sentiment import *
from backend.sentiment.sentiment_analysis import SentimentAnalysis
from backend.functionLib.function_lib import cut_sentences, load_json_file, save_json_file, split_sentences, concat_list
import argparse
import os

analyzer = SentimentAnalysis()


def analysis_single_cut(cut_sentence):
    result = []
    try:
        result = analyzer.analyze(cut_sentence)
    except:
        pass
    return list(map(lambda item: [item[0], item[1], item[4]], result))


def analysis_review(review):
    cut_sentence_list = cut_sentences(split_sentences(review))
    items = concat_list(map(analysis_single_cut, cut_sentence_list))
    return items


def analysis(in_file, out_file=None, cut_out_file=None, start_index=None, end_index=None, map_fun=lambda x: x[0]):
    data = load_json_file(in_file)
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)
    if out_file is None:
        out_file = '%s_%d_%d' % (in_file, start_index, end_index)
    if cut_out_file is None:
        cut_out_file = '%s_cut' % out_file
    data = map(map_fun, data[start_index:end_index])
    if os.path.exists(cut_out_file):
        cut = load_json_file(cut_out_file)
    else:
        cut = list(map(lambda x: cut_sentences(split_sentences(x)), data))
        save_json_file(cut_out_file, cut)
    if os.path.exists(out_file):
        out = load_json_file(out_file)
    else:
        profile_fun = lambda c: concat_list(map(analysis_single_cut, c))
        out = list(map(profile_fun, cut))
        save_json_file(out_file, out)
    # res = list(map(analysis_review, data))
    # save_json_file(out_file, res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--cut_out_file', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    analysis(args.in_file, args.out_file, args.cut_out_file, args.start, args.end)


if __name__ == '__main__':
    main()
