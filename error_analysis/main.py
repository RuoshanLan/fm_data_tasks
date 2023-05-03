import pandas as pd


# path = "datasets/Error_detection/Hospital/ChatGPT/k=0.feather"
path = "datasets/Error_detection/Hospital/ChatGPT/k=10.feather"
# path = "datasets/Error_detection/Hospital/GPT-3/k=0.feather"
# path = "datasets/Entity_match/Beer/GPT-3/k=0.feather"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    df = pd.read_feather(path)
    # print(df[['queries']])
    for i in range(len(df)):
        label = (df.loc[i, "label_str"]).strip().lower()
        pred = df.loc[i, "preds"].strip().lower()
        # print(f"label={label}, preds={pred}")
        # print(f"pred.startswith(label) is {pred.startswith(label)}")
        if(pred.startswith(label)):
            pass
        else:
            query = df.loc[i, "queries"].split("\n\n")[-1]
            print(f"query: {query}, label={label}, preds={pred}")





