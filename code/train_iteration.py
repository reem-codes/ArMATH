import json
import time
import random
from tqdm import tqdm

from train_and_evaluate import *
from util import *
import numpy as np


def train_model(config):
    print("TRAINING MODEL WITH THESE CONFIGURATIONS: ")

    # prepare config
    config["loss"] = []
    config["answer_acc"] = []
    config["equation_acc"] = []
    config["best"] = []
    print(config)

    # set the seed
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    input_lang = None
    embedding_model = None

    # load pretrained model for transfer learning
    if config["transfer_learning"]:
        config_c, _, _, _, _, _, _, _, encoder_c, predict_c, generate_c, merge_c = load_from_filepath(config["transfer_learning_model"], device)

        def copy_param(model1, model2):
            params1 = model1.named_parameters()
            params2 = model2.named_parameters()

            dict_params2 = dict(params2)
            for name1, param1 in params1:
                if name1 in dict_params2:
                    try:
                        dict_params2[name1].data.copy_(param1.data)
                    except:
                        print("failed", name1)
                        pass
            return dict_params2

    # load data
    if not config["arabic"]:
        data = load_raw_data(config["data_path"])
        random.shuffle(data)
        pairs, generate_nums, copy_nums = transfer_num(data, arabic=config["arabic"])
        pairs = [(p[0], from_infix_to_prefix(p[1]), p[2], p[3]) for p in pairs]
        fold_size = int(len(pairs) * 0.2)
        fold_pairs = []
        for split_fold in range(5):
            fold_start = fold_size * split_fold
            fold_end = fold_size * (split_fold + 1)
            fold_pairs.append(pairs[fold_start:fold_end])
        fold_pairs.append(pairs[(fold_size * (5)):])
    else:
        fold_pairs = []
        generate_nums_dict = {}
        copy_nums = 0
        if config["embedding_type"] in ['aravec', 'fasttext']:
            import gensim
            if config["embedding_type"] == 'fasttext':
                embedding_model = gensim.models.fasttext.load_facebook_model(config["embedding_model_name"])
            else:
                embedding_model = gensim.models.Word2Vec.load(config["embedding_model_name"])

        for fold in range(5):
            with open(f'{config["data_path"]}/fold_{fold}.json') as file:
                print("Loading file...")
                data = json.load(file)
                random.shuffle(data)
            if config["embedding_type"] in ['aravec', 'fasttext']:
                pairs, generate_nums_fold, copy_nums_fold = transfer_num_aravec(data, embedding_model)
                pairs = [(p[0], p[1], from_infix_to_prefix(p[2]), p[3], p[4]) for p in pairs]
            else:
                pairs, generate_nums_fold, copy_nums_fold = transfer_num(data, arabic=config["arabic"])
                pairs = [(p[0], from_infix_to_prefix(p[1]), p[2], p[3]) for p in pairs]
            for gen, count in generate_nums_fold.items():
                if generate_nums_dict.get(gen):
                    generate_nums_dict[gen] += count
                else:
                    generate_nums_dict[gen] = count
            copy_nums = max(copy_nums, copy_nums_fold)
            fold_pairs.append(pairs)
        generate_nums = [g for g, v in generate_nums_dict.items() if v >= 5]
        with open(f'{config["output_dir"]}generate_nums.json', 'w') as fp:
            json.dump(generate_nums, fp)
        with open(f'{config["output_dir"]}copy_nums.json', 'w') as fp:
            json.dump(copy_nums, fp)

    # cross validation across 5 folds
    best_acc_fold = []
    for fold in range(5):
        config["fold"] = str(fold)
        config["loss"].append([])
        config["answer_acc"].append([])
        config["equation_acc"].append([])
        config["best"].append({"equation": 0, "answer": 0, "epoch": 0})

        pairs_tested = []
        pairs_trained = []
        for fold_t in range(5):
            if config["train_one_fold_only"]:
                if fold_t == fold:
                    pairs_trained += fold_pairs[fold_t]

                else:
                    pairs_tested += fold_pairs[fold_t]

            else:
                if fold_t == fold:
                    pairs_tested += fold_pairs[fold_t]
                else:
                    pairs_trained += fold_pairs[fold_t]

        if config["embedding_type"] in ['aravec', 'fasttext']:
            output_lang, train_pairs, test_pairs = prepare_data_embedding(pairs_trained, pairs_tested, generate_nums,
                                                                          copy_nums, tree=True)
        else:
            input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested,
                                                                            config["trim_min_count"], generate_nums,
                                                                            copy_nums, tree=True)
            # save input lang
            input_lang.save(f'{config["output_dir"]}input_lang_{config["fold"]}.json')

        # save output lang
        output_lang.save(f'{config["output_dir"]}output_lang_{config["fold"]}.json')

        # save pairs
        with open(f'{config["output_dir"]}train_{config["fold"]}.json', 'w') as fp:
            json.dump(train_pairs, fp)
        with open(f'{config["output_dir"]}test_{config["fold"]}.json', 'w') as fp:
            json.dump(test_pairs, fp)

        generate_num_ids = [output_lang.word2index[num] for num in generate_nums]

        # make models
        encoder, predict, generate, merge = make_models(config, device, generate_nums, copy_nums, output_lang, input_lang, embedding_model)

        if config["transfer_learning"]:
            # copy weights if transfer learning
            if config["transfer_learning_transfer_encoder"]:
                encoder_param = copy_param(encoder_c, encoder)
                encoder.load_state_dict(encoder_param)
            if config["transfer_learning_transfer_decoder"]:
                predict_param = copy_param(predict_c, predict)
                predict.load_state_dict(predict_param)

                generate_param = copy_param(generate_c, generate)
                generate.load_state_dict(generate_param)

                merge_param = copy_param(merge_c, merge)
                merge.load_state_dict(merge_param)

        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        predict_optimizer = torch.optim.Adam(predict.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        generate_optimizer = torch.optim.Adam(generate.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        merge_optimizer = torch.optim.Adam(merge.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

        encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
        predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
        generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
        merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

        # save config
        with open(f'{config["output_dir"]}config_{config["fold"]}.json', 'w') as fp:
            json.dump(config, fp)

        # start training
        train_loader = create_data_loader(train_pairs, config["batch_size"], config["n_workers"])
        for epoch in range(config["n_epochs"]):
            config["epoch"] = str(epoch)
            loss_total = 0
            count_i = 1
            with tqdm(train_loader, unit="batch") as tepoch:
                for input_batches, seq_mask, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches in tepoch:
                    tepoch.set_description(f"Fold {fold} Epoch {epoch}")
                    num_stack_batches = [[[int(i) for i in subarray.split("\t") if len(subarray) != 0] for subarray in x.split("\n") if len(x) != 0] for x in num_stack_batches]
                    num_pos_batches = [[int(i) for i in x.split("\t") if len(x) != 0] for x in num_pos_batches]
                    input_batches = torch.stack(input_batches)
                    output_batches = torch.stack(output_batches)

                    loss = train_tree(
                        input_batches, seq_mask, input_lengths, output_batches, output_lengths,
                        num_stack_batches, num_size_batches, generate_num_ids, encoder, predict, generate, merge,
                        encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches)
                    loss_total += loss
                    tepoch.set_postfix({"Loss": loss, "Average Loss": loss_total / count_i})
                    count_i += 1
                config["loss"][fold].append(loss_total / count_i)

                encoder_scheduler.step()
                predict_scheduler.step()
                generate_scheduler.step()
                merge_scheduler.step()

            if epoch % 10 == 0 or epoch > config["n_epochs"] - 5:
                value_ac = 0
                equation_ac = 0
                eval_total = 0
                start = time.time()
                for test_batch in test_pairs:
                    test_res, _ = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                             merge, output_lang, test_batch[5], beam_size=config["beam_size"])
                    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
                    if val_ac:
                        value_ac += 1
                    if equ_ac:
                        equation_ac += 1
                    eval_total += 1
                print(f"Fold {fold} Epoch {epoch}")
                print(equation_ac, value_ac, eval_total)
                ans = float(value_ac) / eval_total
                eq = float(equation_ac) / eval_total
                print(f"test accuracy: equation: {eq*100:.3f}%\tanswer: {ans*100:.3f}%")
                config["answer_acc"][fold].append(ans)
                config["equation_acc"][fold].append(eq)

                print("testing time", time_since(time.time() - start))
                if config["best"][fold]["answer"] < ans or config["best"][fold]["equation"] < eq:
                    config["best"][fold]["answer"] = ans
                    config["best"][fold]["equation"] = eq
                    config["best"][fold]["epoch"] = epoch

                    print("Saving...")
                    save_models(config, encoder, predict, generate, merge)
                else:
                    print(f'Not saving, best @ {config["best"][fold]["epoch"]}')
                print("------------------------------------------------------")

                if epoch == config["n_epochs"] - 1:
                    best_acc_fold.append((equation_ac, value_ac, eval_total))

        with open(f'{config["output_dir"]}config_{config["fold"]}.json', 'w') as fp:
            json.dump(config, fp)
    a, b, c = 0, 0, 0
    for bl in range(len(best_acc_fold)):
        a += best_acc_fold[bl][0]
        b += best_acc_fold[bl][1]
        c += best_acc_fold[bl][2]
        print(best_acc_fold[bl])
    print(a / float(c), b / float(c))
    print("DONE.")


def evaluate_model(config):
    print("EVALUATING MODEL WITH CONFIG FILE AT", config["config_path"])
    config, train_pairs, test_pairs, generate_nums, generate_num_ids, copy_nums, output_lang, input_lang, encoder, predict, generate, merge = load_from_filepath(config["config_path"], device)
    print(config)

    templates = [' '.join(output_lang.index2word[x] for x in output[2]) for output in train_pairs]

    value_ac = 0
    equation_ac = 0
    eval_total = 0
    accuracies = {}
    samples_prediction = []
    with tqdm(test_pairs, unit="batch") as tepoch:
        for test_batch in tepoch:
            template = ' '.join(output_lang.index2word[x] for x in test_batch[2])
            if not accuracies.get(template):
                accuracies[template] = {}
                accuracies[template]["answer"] = 0
                accuracies[template]["equation"] = 0
                accuracies[template]["total"] = 0
                accuracies[template]["tran_total"] = 0
            test_res, _ = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                        merge, output_lang, test_batch[5], beam_size=config["beam_size"])
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4],
                                                              test_batch[6])
            samples_prediction.append(test_batch + [test_res, val_ac, equ_ac])
            if val_ac:
                value_ac += 1
                accuracies[template]["answer"] += 1
            if equ_ac:
                equation_ac += 1
                accuracies[template]["equation"] += 1
            eval_total += 1
            accuracies[template]["total"] += 1
            tepoch.set_postfix(
                {"Equation Accuracy": float(equation_ac) / eval_total, "Answer Accuracy": float(value_ac) / eval_total})
    bar_data = {k: [v['answer'] / v['total'], v['total'], k in templates, v['answer']] for k, v in accuracies.items()}
    acc_count = np.array(list(bar_data.values()))
    d = acc_count
    print(f"Accuracy over all templates: {sum(d[:, 3]) / sum(d[:, 1]):.2f}")
    d = acc_count[acc_count[:, 1] == 1]
    print(f"Accuracy over all templates appearing exactly once: {sum(d[:, 3]) / sum(d[:, 1]):.2f}")
    d = acc_count[acc_count[:, 1] > 1]
    print(f"Accuracy over all templates appearing more than once:  {sum(d[:, 3]) / sum(d[:, 1]):.2f}")
    print()
    d = acc_count[acc_count[:, 2] == 1]
    print(f"Accuracy over seen templates:  {sum(d[:, 3]) / sum(d[:, 1]):.2f}")
    d = acc_count[(acc_count[:, 2] == 1) & (acc_count[:, 1] == 1)]
    print(f"Accuracy over seen templates appearing exactly once: {sum(d[:, 3]) / sum(d[:, 1]):.2f}")
    d = acc_count[(acc_count[:, 2] == 1) & (acc_count[:, 1] > 1)]
    print(f"Accuracy over seen templates appearing more than once:  {sum(d[:, 3]) / sum(d[:, 1]):.2f}")
    print()
    d = acc_count[acc_count[:, 2] == 0]
    print(f"Accuracy over unseen templates:  {sum(d[:, 3]) / sum(d[:, 1]):.2f}")
    d = acc_count[(acc_count[:, 2] == 0) & (acc_count[:, 1] == 1)]
    print(f"Accuracy over unseen templates appearing exactly once: {sum(d[:, 3]) / sum(d[:, 1]):.2f}")
    d = acc_count[(acc_count[:, 2] == 0) & (acc_count[:, 1] > 1)]
    print(f"Accuracy over unseen templates appearing more than once:  {sum(d[:, 3]) / sum(d[:, 1]):.2f}")
    with open(f'{config["output_dir"]}evaluation_accuracies_{config["fold"]}.json', 'w') as fp:
        json.dump(accuracies, fp)
    with open(f'{config["output_dir"]}evaluation_predictions_{config["fold"]}.json', 'w') as fp:
        json.dump(samples_prediction, fp)
    print("DONE.")
