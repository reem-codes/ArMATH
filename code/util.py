import json
import torch
from pre_data import *
from model import *
from expressions_transfer import * 


def save_models(config, encoder, predict, generate, merge):
    # save models
    if config["embedding_type"] in ["aravec", "fasttext"]:
        # remove fixed embedding first to save space
        e = encoder.state_dict().copy()
        del e['embedding.weight']
        torch.save(e, f'{config["output_dir"]}encoder_{config["fold"]}.pt')
    else:
        torch.save(encoder.state_dict(), f'{config["output_dir"]}encoder_{config["fold"]}.pt')

    torch.save(predict.state_dict(), f'{config["output_dir"]}predict_{config["fold"]}.pt')
    torch.save(generate.state_dict(), f'{config["output_dir"]}generate_{config["fold"]}.pt')
    torch.save(merge.state_dict(), f'{config["output_dir"]}merge_{config["fold"]}.pt')


def make_models(config, device, generate_nums, copy_nums, output_lang, input_lang, embedding_model):
    if config["embedding_type"] in ["aravec", "fasttext"]:
        encoder = EncoderSeqEmbedding(embedding_model=embedding_model, n_layers=config["n_layers"],
                                      embedding_size=config["embedding_size"], hidden_size=config["hidden_size"])
    else:
        encoder = EncoderSeq(input_size=input_lang.n_words, n_layers=config["n_layers"],
                                      embedding_size=config["embedding_size"], hidden_size=config["hidden_size"])

    predict = Prediction(hidden_size=config["hidden_size"], op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=config["hidden_size"], op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=config["embedding_size"])
    merge = Merge(hidden_size=config["hidden_size"], embedding_size=config["embedding_size"])
    encoder.to(device)
    predict.to(device)
    generate.to(device)
    merge.to(device)

    return encoder, predict, generate, merge


def load_models(config, device, generate_nums, copy_nums, output_lang, input_lang):
    print("Loading models...")
    embedding_model = None
    if config["embedding_type"] in ['aravec', 'fasttext']:
        import gensim
        if config["embedding_type"] == 'fasttext':
            embedding_model = gensim.models.fasttext.load_facebook_model(config["embedding_model_name"])
        else:
            embedding_model = gensim.models.Word2Vec.load(config["embedding_model_name"])

    encoder, predict, generate, merge = make_models(config, device, generate_nums, copy_nums, output_lang, input_lang, embedding_model)

    if config["embedding_type"] == "one-hot":
        encoder.load_state_dict(torch.load(f'{config["output_dir"]}encoder_{config["fold"]}.pt'))
    else:
        pretrained_dict = torch.load(f'{config["output_dir"]}encoder_{config["fold"]}.pt')
        model_dict = encoder.state_dict()
        pretrained_dict["embedding.weight"] = model_dict["embedding.weight"]
        encoder.load_state_dict(pretrained_dict)

    predict.load_state_dict(torch.load(f'{config["output_dir"]}predict_{config["fold"]}.pt'))
    generate.load_state_dict(torch.load(f'{config["output_dir"]}generate_{config["fold"]}.pt'))
    merge.load_state_dict(torch.load(f'{config["output_dir"]}merge_{config["fold"]}.pt'))

    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    return encoder, predict, generate, merge


def load_splitted_pairs(config):
    with open(f'{config["output_dir"]}generate_nums.json') as fp:
        generate_nums = json.load(fp)
    with open(f'{config["output_dir"]}copy_nums.json') as fp:
        copy_nums = json.load(fp)
    print("Loading pairs...")
    # load pairs
    with open(f'{config["output_dir"]}train_{config["fold"]}.json') as file:
        train_pairs = json.load(file)
    with open(f'{config["output_dir"]}test_{config["fold"]}.json') as file:
        test_pairs = json.load(file)
    # load lang
    print("Loading lang...")
    output_lang = Lang()
    input_lang = Lang()
    output_lang.load(f'{config["output_dir"]}output_lang_{config["fold"]}.json')

    if config["embedding_type"] in ['aravec', 'fasttext']:
        import gensim
        if config["embedding_type"] == 'fasttext':
            embedding_model = gensim.models.fasttext.load_facebook_model(config["embedding_model_name"])
        else:
            embedding_model = gensim.models.Word2Vec.load(config["embedding_model_name"])
        input_lang = embedding_model

    else:
        input_lang.load(f'{config["output_dir"]}input_lang_{config["fold"]}.json')
    generate_num_ids = [output_lang.word2index[num] for num in generate_nums]
    return train_pairs, test_pairs, generate_nums, generate_num_ids, copy_nums, output_lang, input_lang


def load_from_filepath(filepath, device):
    # load config
    with open(filepath) as fp:
        print("Loading config...")
        config = json.load(fp)

    # load pairs
    train_pairs, test_pairs, generate_nums, generate_num_ids, copy_nums, output_lang, input_lang = load_splitted_pairs(config)

    encoder, predict, generate, merge = load_models(config, device, generate_nums, copy_nums, output_lang, input_lang)
    return config, train_pairs, test_pairs, generate_nums, generate_num_ids, copy_nums, output_lang, input_lang, encoder, predict, generate, merge
