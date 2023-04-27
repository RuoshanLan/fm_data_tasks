import pandas as pd
import os

def main():
    print(os.path.abspath("../Fodors-Zagats/test/default/10k_0inst_1cb_random_200run_0dry/trial_0.feather"))
    data = pd.read_feather("trial_2.feather")#("../Fodors-Zagats/test/default/0k_1inst_0cb_random_200run_0dry/trial_0.feather")
    print(data.head(20))
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    print(data[data["label_str"].str.strip("\n") != data["preds"]][["text", "label_str", "preds"]])

if __name__ == "__main__":
    main()
