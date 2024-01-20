import json
import pandas as pd

root_path = "./datasets/"

with open(f'{root_path}gsm8k_test.json') as json_file:
    data = json.load(json_file)
    print("number of data: ", len(data))
    df = pd.DataFrame(data)
    df.to_excel(f'{root_path}gsm8k_test.xlsx')
    # for item in data:
    #     print(item)