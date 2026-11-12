#simple tokenizer serving unkonw words
import re
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) #rozdzielenie na tekst
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed] #poprawka na uknown
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #r'\1' to r ze znakiem
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) #wykurwia spacje przed znakami interpunkcyjnymi
        return text