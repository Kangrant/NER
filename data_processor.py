import json
from vocabulary import Vocabulary
from pathlib import Path
import pickle
#clunner数据集
# class CluenerProcessor:
#     """Processor for the chinese ner data set."""
#     def __init__(self,data_dir):
#         self.vocab = Vocabulary()
#         self.data_dir = data_dir
#
#     def get_vocab(self):
#         """
#         VOCAB如果存在，直接读取。如果不存在，开始建立并保存
#         """
#         vocab_path = self.data_dir / 'vocab.pkl'
#         if vocab_path.exists():
#             self.vocab.load_from_file(str(vocab_path))
#         else:
#             files = ["train.json", "dev.json", "test.json"]
#             for file in files:
#                 with open(str(self.data_dir / file), 'r',encoding='utf-8') as fr:
#                     for line in fr:
#                         line = json.loads(line.strip())
#                         text = line['text']
#                         self.vocab.update(list(text))
#             self.vocab.build_vocab()
#             self.vocab.save(vocab_path)
#
#     def get_train_examples(self):
#         """See base class."""
#         return self._create_examples(str(self.data_dir / "train.json"), "train")
#
#     def get_dev_examples(self):
#         """See base class."""
#         return self._create_examples(str(self.data_dir / "dev.json"), "dev")
#
#     def get_test_examples(self):
#         """See base class."""
#         return self._create_examples(str(self.data_dir / "test.json"), "test")
#
#     def _create_examples(self,input_path,mode):
#         """
#         Returns:List[Dict]
#         [{'id':train_1 ,'context':['中','国','人'] ,'tag':[B-name,I-name,I-name] ,'raw_context':[中国人] ,},{},{}...]
#
#         """
#
#         examples = []
#         with open(input_path, 'r',encoding='utf-8') as f:
#             idx = 0
#             for line in f:
#                 json_d = {}
#                 line = json.loads(line.strip())
#                 text = line['text']
#                 label_entities = line.get('label', None)
#                 words = list(text) #['中','国','人']
#                 labels = ['O'] * len(words)
#                 if label_entities is not None:
#                     for key, value in label_entities.items():
#                         for sub_name, sub_index in value.items():
#                             for start_index, end_index in sub_index:
#                                 assert ''.join(words[start_index:end_index + 1]) == sub_name
#                                 #转换标签
#                                 #首末索引号相等说明是单个词
#                                 if start_index == end_index:
#                                     labels[start_index] = 'S-' + key
#                                 else:
#                                     #第一个词是B，之后是I
#                                     labels[start_index] = 'B-' + key
#                                     labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
#
#                 json_d['id'] = f"{mode}_{idx}"
#                 json_d['context'] = " ".join(words)
#                 json_d['tag'] = " ".join(labels)
#                 json_d['raw_context'] = "".join(words)
#                 idx += 1
#                 examples.append(json_d)
#         return examples




#cloab数据集

class CluenerProcessor:
    """Processor for the chinese ner data set."""
    def __init__(self,data_dir):
        self.vocab = Vocabulary()
        self.data_dir = data_dir

    def get_vocab(self):
        """
        VOCAB如果存在，直接读取。如果不存在，开始建立并保存
        """
        vocab_path = self.data_dir / 'vocab.pkl'
        if vocab_path.exists():
            self.vocab.load_from_file(str(vocab_path))
        else:
            #只需要创建train的词表就可以。不使用预训练模型时，非train的字也训练不到
            files = ["train.txt", "dev.txt", "test.txt"]
            for file in files:
                with open(str(self.data_dir / file), 'r',encoding='utf-8') as fr:
                    for line in fr:
                        #line = json.loads(line.strip())
                        line = line.strip().split(' ')
                        text = line[0]
                        self.vocab.update(list(text))
            self.vocab.build_vocab()
            self.vocab.save(vocab_path)

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "train.txt"), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "dev.txt"), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(str(self.data_dir / "test.txt"), "test")

    def _create_examples(self,input_path,mode):
        """
        Returns:List[Dict]

        [{'context':['中','国','人'] ,'tag':[B-name,I-name,I-name]} ,{},{}...]
        """

        examples = []
        with open(input_path, 'r',encoding='utf-8') as f:

            words,labels = [],[]
            flag = False
            for line in f:
                json_d = {}
                content = line.strip()
                tokens = content.split(' ') #[word,label]

                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                    if tokens[-1] != 'O':
                        flag = True

                else:
                    if len(content) == 0 and len(words)>0:
                        if flag:
                            json_d['context'] = " ".join(words)
                            json_d['tag'] = " ".join(labels)
                            words, labels = [], []
                            examples.append(json_d)
                            flag = False
                        else:
                            words = []
                            labels = []

        return examples

if __name__ == '__main__':
    #pass
    processor = CluenerProcessor(data_dir=Path("./dataset/colab"))

    #查看词表
    # processor.get_vocab()
    # with open('./dataset/colab/vocab.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     print(data)

    examples = processor.get_test_examples()
    for x in examples:
        print(x)
