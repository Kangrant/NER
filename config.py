from pathlib import Path

#data_dir = Path("./dataset/cluener")
data_dir = Path("./dataset/colab") #实验室数据

train_path = data_dir / 'train.json'
dev_path =data_dir / 'dev.json'
test_path = data_dir / 'test.json'
output_dir = Path("./outputs")

# label2id = {
#     "O": 0,
#     "B-address":1,
#     "B-book":2,
#     "B-company":3,
#     'B-game':4,
#     'B-government':5,
#     'B-movie':6,
#     'B-name':7,
#     'B-organization':8,
#     'B-position':9,
#     'B-scene':10,
#     "I-address":11,
#     "I-book":12,
#     "I-company":13,
#     'I-game':14,
#     'I-government':15,
#     'I-movie':16,
#     'I-name':17,
#     'I-organization':18,
#     'I-position':19,
#     'I-scene':20,
#     "S-address":21,
#     "S-book":22,
#     "S-company":23,
#     'S-game':24,
#     'S-government':25,
#     'S-movie':26,
#     'S-name':27,
#     'S-organization':28,
#     'S-position':29,
#     'S-scene':30,
#     "<START>": 31,
#     "<STOP>": 32
# }
label2id = {
    "O":0,
    'B-a':1,
    'I-a':2,
    "E-a":3,
    "B-e":4,
    "I-e":5,
    "E-e":6,
    "B-exp-fac":7,
    "I-exp-fac":8,
    "E-exp-fac":9,
    "B-exp-con":10,
    "I-exp-con":11,
    "E-exp-con":12,
    "B-exp-rea":13,
    "I-exp-rea":14,
    "E-exp-rea":15,
    "B-exp-sug":16,
    "I-exp-sug":17,
    "E-exp-sug":18,
    'S-a': 19,
    'S-e':20,
    'S-exp-fac':21,
    "<START>":22,
    "<STOP>":23
}
