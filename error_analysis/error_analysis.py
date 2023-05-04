import pandas as pd


# path = "datasets/Error_detection/Hospital/ChatGPT/k=0.feather"
# path = "datasets/Error_detection/Hospital/ChatGPT/k=0/trial_0.feather"
# path = "datasets/Error_detection/Hospital/GPT-3/k=10/trial_0.feather"
# path = "datasets/Entity_match/Beer/GPT-3/k=0.feather"
path = "datasets/Data_imputation/Restaurant/chatgpt/k=0/trial_0.feather"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    df = pd.read_feather(path)
    print(len(df[['queries']]))
    for i in range(len(df)):
        label = (df.loc[i, "label_str"]).strip().lower()
        pred = df.loc[i, "preds"].strip().lower()
        # print(f"preds are {pred}")
        # print(f"label={label}, preds={pred}")
        # print(f"pred.startswith(label) is {pred.startswith(label)}")
        query = df.loc[i, "queries"].split("\n\n")[-1]


        if " is " in pred:
             startst = pred.split("is")[0]
             endst = pred.split("is")[-1]
             #logger.info(f"start is {startst}, end is {endst}, label is {label}")
             if label in endst or label in startst:
                 pass
             else:
                print(f"query: {query},  label={label} is not in preds={pred}")
        else:
            if pred not in label and label not in pred:
                print(f"query: {query}, label={label} and preds={pred} are not same")

        
        
        # if(pred.startswith(label)):
        #     pass
        # else:
        #     query = df.loc[i, "queries"].split("\n\n")[-1]
        #     print(f"query: {query}, label={label}, preds={pred}")





