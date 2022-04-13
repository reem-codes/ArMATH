import re
import json
from torch.utils.data import Dataset, DataLoader

import pyarabic.araby as araby
import torch

PAD_token = 0


class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """

    def save(self, filepath):
        info = {
            "word2index": self.word2index,
            "word2count": self.word2count,
            "index2word": self.index2word,
            "n_words": self.n_words,
            "num_start": self.num_start
        }
        with open(filepath, 'w') as fp:
            json.dump(info, fp, ensure_ascii=False)

    def load(self, filepath):
        with open(filepath) as file:
            info = json.load(file)
            self.word2index = info['word2index']
            self.word2count = info['word2count']
            self.index2word = info['index2word']
            self.n_words = info['n_words'] 
            self.num_start = info['num_start']

    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0

    def add_sen_to_vocab(self, sentence):
        """ add words of sentence to vocab"""
        for word in sentence:
            if re.search("N\d+|NUM|\d+", word):
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):
        """ trim words below a certain count threshold"""
        keep_words = [k for k, v in self.word2count.items()  if v >= min_count]
        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["PAD", "NUM", "UNK"] + self.index2word
        else:
            self.index2word = ["PAD", "NUM"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang(self, generate_num, copy_nums):
        # build the output lang vocab and dict
        self.index2word = ["PAD", "EOS"] + self.index2word + generate_num + \
                          ["N" + str(i) for i in range(copy_nums)] + \
                          ["SOS", "UNK"]
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):
        # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + \
                          ["N" + str(i) for i in range(copy_nums)] + ["UNK"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i


def load_raw_data(filename):  
    # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data


def clean_str(text):
    """
    Clean/Normalize Arabic Text
    * remove some special characters
    * normalize أ إ ئ .. etc
    * remove tashkeel
    * turn math operations into words (because most pre-trained embeddings don't have them)

    and then return the tokens
    """
    op = "+-*/^"
    op2 = "زائد ناقص ضرب قسمة أس".split()
    for i in range(0, len(op)):
        text = text.replace(op[i], op2[i])
    text = text.replace('؛', '،')
    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    text = araby.strip_tashkeel(text)
    # trim
    text = text.strip()

    return araby.tokenize(text)


def tokenize_aravec(text, model):
    """ Given a test and the aravec model, return the cleaned input tokens and their ids """
    input_seq = clean_str(text)
#     op = "+-*/^"
#     op2 = "زائد ناقص ضرب قسمة أس"
#     for i in range(0, len(op)):
#         text = text.replace(op[i], op2[i])
#     input_seq = text.split()
    input_ids = [model.wv.key_to_index.get(word, 0) for word in input_seq]
    return input_seq, input_ids


class MathDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.input_len_max = max(i[1] for i in pairs)
        self.output_len_max = max(i[3] for i in pairs)
        self.num_stack_len_max = max(len(i[6]) for i in pairs)
        self.num_pos_len_max = max(len(i[5]) for i in pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        pair = self.pairs[index]
        i, li, j, lj, num, num_pos, num_stack = pair
        input_length = li
        output_length = lj
        input_batch = pad_seq(i, li, self.input_len_max)
        output_batch = pad_seq(j, lj, self.output_len_max)
        num_batch = len(num)
        num_stack_batch = '\n'.join(['\t'.join([str(i) for i in subarray]) for subarray in num_stack])
        num_pos_batch = '\t'.join(map(str, num_pos))
        num_size_batch = len(num_pos)
        # sequence mask for attention
        seq_mask = torch.ByteTensor([0] * input_length + [1] * (self.input_len_max - input_length))
        return input_batch, seq_mask, input_length, output_batch, output_length, num_batch, num_stack_batch, num_pos_batch, num_size_batch 


def create_data_loader(pairs_to_batch, batch_size, num_workers=4):
    ds = MathDataset(
        pairs=pairs_to_batch
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )


def transfer_num(data, arabic=True):  # transfer num into "NUM"
    """ given the data and the bert tokenizer, replace \d with  NUM
        returns:
        * input and output pairs
        * numbers not in the question (like PI)
        * max variable count
    """
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []

        if arabic:
            seg = d["segmented"].strip().split()
        else:
            seg = d["segmented_text"].strip().split()

        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start(): pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "NUM":
                num_pos.append(i)
        assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos))

    if arabic:
        return pairs, generate_nums_dict, copy_nums

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return pairs, temp_g, copy_nums


def transfer_num_bert(data, tokenizer):
    """ given the data and the bert tokenizer, replace \d with  NUM
        returns:
        * input and output pairs (with input ids)
        * numbers not in the question (like PI)
        * max variable count
    """
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = re.findall(pattern, d["segmented"])
        text = re.sub(pattern, " N ", d["segmented"])
        input_seq = []
        input_ids = tokenizer.encode_plus(text.replace("/", "\\"))['input_ids']
        input_seq = tokenizer.convert_ids_to_tokens(input_ids)
        equations = d["equation"][2:]
        if copy_nums < len(nums):
            copy_nums = len(nums)
        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "N":
                num_pos.append(i)
        if len(nums) != len(num_pos):
            print(nums, num_pos)
            print(input_seq, out_seq, nums)

        assert len(nums) == len(num_pos)
        pairs.append((input_seq, input_ids, out_seq, nums, num_pos))

    return pairs, generate_nums_dict, copy_nums



def transfer_num_aravec(data, model):
    """ given the data and the aravec tokenizer, replace \d with  NUM
        returns:
        * input and output pairs (with input ids)
        * numbers not in the question (like PI)
        * max variable count
    """

    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = re.findall(pattern, d["segmented"])
        text = re.sub(pattern, " مجهول ", d["segmented"])
        input_seq = []
        input_seq, input_ids = tokenize_aravec(text, model)
        equations = d["equation"][2:]
        if copy_nums < len(nums):
            copy_nums = len(nums)
        nums_fraction = []
        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1:
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        for s in out_seq:  # tag the num which is generated
            if s[0].isdigit() and s not in generate_nums and s not in nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        for i, j in enumerate(input_seq):
            if j == "مجهول":
                num_pos.append(i)
        if len(nums) != len(num_pos):
            print(nums, num_pos)
            print(input_seq, out_seq, nums)

        assert len(nums) == len(num_pos)
        pairs.append((input_seq, input_ids, out_seq, nums, num_pos))

    return pairs, generate_nums_dict, copy_nums


def indexes_from_sentence(lang, sentence, tree=False):
    """ Return a list of indexes, one for each word in the sentence, plus EOS """
    res = [lang.word2index.get(word, lang.word2index.get("UNK")) for word in sentence if len(word) != 0]
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res


def sentence_from_indexes(lang, indices, ar=False):
    out = ' '.join([lang.index2word[i] for i in indices])
    out = out.replace("PAD", "")
    out = out.replace("EOS", "")
    out = out.replace("SOS", "")
    if ar:
        out = out.replace("UNK", "؟")
        out = out.replace("NUM", "مجهول")
    return out


def sentence_from_indexes_aravec(lang, indices, ar=False):
    out = ' '.join([lang.wv.index_to_key[i] for i in indices])
    out = out.replace("NUM", "مجهول")
    return out


def prepare_data(pairs_trained, pairs_tested, trim_min_count, generate_nums, copy_nums, tree=False):
    """ prepare data for training and make language
        given the training and testing pairs, trimming, the numbers to genrate and the max number of variables, 
        return the language and pairs ready for training. Each pair is:
            * input ids + length
            * output ids + lengths
            * NUM -> num
            * the position of NUM in input
            * num stack
    """
    input_lang = Lang()
    output_lang = Lang()
    train_pairs = []
    test_pairs = []

    print("Indexing words...")
    for pair in pairs_trained:
        if not tree:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
        elif pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
            output_lang.add_sen_to_vocab(pair[1])
    input_lang.build_input_lang(trim_min_count)
    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    train_pairs = get_model_pairs(pairs_trained, output_lang, input_lang, tree)
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    test_pairs = get_model_pairs(pairs_tested, output_lang, input_lang, tree)
    print('Number of testing data %d' % (len(test_pairs)))

    return input_lang, output_lang, train_pairs, test_pairs


def get_model_pairs(pairs, output_lang, input_lang, tree=False):
    """ return the language and pairs ready for training. Each pair is:
        * input ids + length
        * output ids + lengths
        * NUM -> num
        * the position of NUM in input
        * num stack
    """
    pairs2 = []
    for pair in pairs:
        num_stack = []
        for word in pair[1]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[2]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[2]))])

        num_stack.reverse()
        input_cell = indexes_from_sentence(input_lang, pair[0])
        output_cell = indexes_from_sentence(output_lang, pair[1], tree)

        pairs2.append((input_cell, len(input_cell), output_cell,
                       len(output_cell), pair[2], pair[3], num_stack))
    return pairs2


def prepare_data_embedding(pairs_trained, pairs_tested, generate_nums, copy_nums, tree=False): 
    """ prepare data for training, assumes input used a pretrained embedding model"""
    output_lang = Lang()
    train_pairs = []
    test_pairs = []
    input_word_count = set()

    for pair in pairs_trained:
        if pair[-1]:
            output_lang.add_sen_to_vocab(pair[2])

    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)

    train_pairs = get_model_pairs_embedding(pairs_trained, output_lang, tree)
    print('Indexed %d words in input language, %d words in output' % 
          (len(input_word_count), output_lang.n_words))
    print('Number of training data %d' % (len(train_pairs)))
    test_pairs = get_model_pairs_embedding(pairs_tested, output_lang, tree)
    print('Number of testing data %d' % (len(test_pairs)))

    return output_lang, train_pairs, test_pairs


def get_model_pairs_embedding(pairs, output_lang, tree=False):
    """ return the language and pairs ready for training. Each pair is:
        * input ids + length
        * output ids + lengths
        * NUM -> num
        * the position of NUM in input
        * num stack
    """
    pairs2 = []
    for pair in pairs:
        num_stack = []
        for word in pair[2]:
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                flag_not = False
                for i, j in enumerate(pair[3]):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair[3]))])

        num_stack.reverse()
        input_cell = pair[1]
        output_cell = indexes_from_sentence(output_lang, pair[2], tree)

        pairs2.append((input_cell, len(input_cell), output_cell,
                       len(output_cell), pair[3], pair[4], num_stack))
    return pairs2


def pad_seq(seq, seq_len, max_length):
    """ Pad a with the PAD symbol """
    extra_padding = [PAD_token] * (max_length - seq_len)
    return seq + extra_padding


def make_regression_data(pairs, k=5):
    """ given training pairs
        make 5 + 1 lists:
        * 5 containing the top 5 most repititive classes (eg all samples with template = / N0 N1)
        * a list containing others 
        use it to train N + M, S times
    """
    pairs_dic = {}
    for sample in pairs:
        try:
            pairs_dic[tuple(sample[2])].append(sample)
        except:
            pairs_dic[tuple(sample[2])] = [sample]

    top_k_keys = sorted(pairs_dic, key=lambda key: len(pairs_dic[key]), reverse=True)[:k]
    top_k_list = [pairs_dic[key] for key in top_k_keys]

    other_list = []
    for key in pairs_dic.keys():
        if key not in top_k_keys:
            other_list.extend(pairs_dic[key])
    return pairs_dic, top_k_list, other_list