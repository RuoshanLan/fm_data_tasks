import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(usage="For qualitative error analysis. Takes in dataset name, trial number and k (default to 0), example size- default is 200, boolean validation \
    Example: python3 read_feather.py Buy -k 0 | less")
    parser.add_argument('filename')     # Name of dataset e.g buy
    parser.add_argument('-k', type=int, default=10, help='[k] in k-shot')
    parser.add_argument('--trial_no', type=int, default=0, help='[trial_number]')
    parser.add_argument(
        "--valid",
        action="store_true",
        help="Use validation instead of test",
    )
    parser.add_argument(
        "-n",
        type=int,
        help="Number of examples",
        default=200,
    )
    parser.add_argument(
        "--ncb",
        action="store_true",
        help="If not class_balanced. Default is class balanced for k = 10, not class balanced otherwise",
    )
    args = vars(parser.parse_args())

    if args['ncb'] or args['k'] != 10:
        val = 0
    else:
        val = 1

    print(val)

    path = "../{}/{}/default/{}k_{}inst_{}cb_{}_{}run_0dry/trial_{}.feather".format(
        args['filename'], ("validation" if args['valid'] else "test"), args['k'],(1 if not args['k'] else 0) , val,
        ("manual" if args['k'] == 1 else "random"), args['n'], args['trial_no']
    )
    data = pd.read_feather(path)
    print(data.head(20))
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    data_part = data[data["label_str"].str.strip().str.lower() != data["preds"].str.strip().str.lower()][["text", "label_str", "preds"]]
    if "Buy" in path:
        data_part["text"] = data_part["text"].str.split(".").str[0]
        data_part["text"] = data_part["text"].where(data_part["text"].str.len() < 100, data_part["text"].str.slice(0, 100))
    print(list(data_part.columns))
    print(data_part)

if __name__ == "__main__":
    main()
