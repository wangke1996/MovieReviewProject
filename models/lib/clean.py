import tensorflow as tf
from models.lib.data_helper import read_lines
from models.lib.scorer import calculate_score
import numpy as np


def rename_var_in_ckpt(checkpoint_dir, should_replace=lambda x: True, rename=lambda x: x, dry_run=False):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            if should_replace(var_name):
                new_name = rename(var_name)
                print('%s would be renamed to %s.' % (var_name, new_name))

            if not dry_run:
                # print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)


def test():
    import pickle as pkl
    models = ["ItemCF", "ItemCF_IIF", "UserCF", "UserCF_IIF", "ncf", "triplev2_ens"]
    print('model\tacc\tf1\tmse\tacc_std\tf1_std\tmse_std')
    for model in models:
        f1s = []
        accs = []
        mses = []
        for i in range(5):
            prediction_file = "/data/wangke/MovieProject/MovieData/save/model/%s-%d/predictions.pkl" % (model, i)
            label_file = "/data/wangke/MovieProject/MovieData/data/train/rate_record_%d.txt" % i
            with open(prediction_file, 'rb') as f:
                preds = pkl.load(f)
            labels = read_lines(label_file, lambda x: int(x.strip().split()[2]))
            f1s.append(calculate_score(preds, labels, 'macro_F1'))
            accs.append(calculate_score(preds, labels, 'ACC'))
            mses.append(calculate_score(preds, labels, 'MSE'))
        print('%s\t%f\t%f\t%f\t%f\t%f\t%f' % (
            model, np.mean(accs), np.mean(f1s), np.mean(mses), np.max(accs) - np.mean(accs), np.max(f1s) - np.mean(f1s),
            np.max(mses) - np.mean(mses)))


if __name__ == '__main__':
    test()
    # make_rate_prediction_dataset()
