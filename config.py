import time


class SplitConfig:
    corpus = "MSVD"
    encoder_model = "InceptionV4"

    video_fpath = "data/{}/features/{}.hdf5".format(corpus, encoder_model)
    caption_fpath = "data/{}/metadata/MSR Video Description Corpus.csv".format(corpus)

    random_seed = 42
    n_train = 1200
    n_val = 100
    n_test = 670

    train_video_fpath = "data/{}/features/{}_train.hdf5".format(corpus, encoder_model)
    val_video_fpath = "data/{}/features/{}_val.hdf5".format(corpus, encoder_model)
    test_video_fpath = "data/{}/features/{}_test.hdf5".format(corpus, encoder_model)

    train_metadata_fpath = "data/{}/metadata/train.csv".format(corpus)
    val_metadata_fpath = "data/{}/metadata/val.csv".format(corpus)
    test_metadata_fpath = "data/{}/metadata/test.csv".format(corpus)


class TrainConfig:
    decoder_model = "GRU" # [ "LSTM", "GRU" ]
    frame_sampling_method = "uniform" # [ "uniform", "random", "uniform_jitter" ]
    decoder_teacher_forcing_ratio = 0.5


    model = "RecNet"
    corpus = "MSVD" # [ "MSVD" ]
    encoder_model = "InceptionV4" # [ "InceptionV4" ]
    # decoder_model = "LSTM" # [ "LSTM", "GRU" ]
    device = "cuda"

    """ Data Loader """
    build_train_data_loader = True
    build_val_data_loader = True
    build_test_data_loader = True
    total_video_fpath = "data/{}/features/{}.hdf5".format(corpus, encoder_model)
    total_caption_fpath = "data/{}/metadata/MSR Video Description Corpus.csv".format(corpus)
    train_video_fpath = "data/{}/features/{}_train.hdf5".format(corpus, encoder_model)
    train_caption_fpath = "data/{}/metadata/train.csv".format(corpus)
    val_video_fpath = "data/{}/features/{}_val.hdf5".format(corpus, encoder_model)
    val_caption_fpath = "data/{}/metadata/val.csv".format(corpus)
    test_video_fpath = "data/{}/features/{}_test.hdf5".format(corpus, encoder_model)
    test_caption_fpath = "data/{}/metadata/test.csv".format(corpus)
    min_count = 5 # N_vocabs = 1: 13501 | 2: 7424 | 3: 5692 | 4: 4191 | 5: 4188
    # frame_sampling_method = "uniform_jitter" # [ "uniform", "random", "uniform_jitter" ]
    caption_n_max_word = 30
    batch_size = 100
    shuffle = True
    num_workers = 4

    """ Train """
    train_n_iteration = 100000
    decoder_learning_rate = 1e-4
    reconstructor_learning_rate = 1e-6
    decoder_weight_decay = 1e-5
    reconstructor_weight_decay = 1e-5
    decoder_use_amsgrad = True
    reconstructor_use_amsgrad = False
    use_gradient_clip = True
    clip = 50.0 # Gradient clipping

    """ Word Embedding """
    init_word2idx = { '<PAD>': 0, '<SOS>': 1, '<EOS>': 2 }
    embedding_size = 468
    embedding_dropout = 0.5
    embedding_scale = 1

    """ Encoder """
    encoder_output_size = 1536
    encoder_output_len = 28

    """ Decoder """
    decoder_n_layers = 1
    decoder_hidden_size = 512
    decoder_attn_size = 128
    decoder_dropout = 0.5
    decoder_out_dropout = 0.5
    # decoder_teacher_forcing_ratio = 1.0

    """ Reconstructor """
    use_recon = True
    reconstructor_type = "global" # [ "global", "local" ]
    reconstructor_model = "LSTM"
    reconstructor_n_layers = 1
    reconstructor_hidden_size = 1536
    reconstructor_dropout = 0

    """ Log """
    log_every = 100
    validate_every = 1000
    test_every = 10000
    save_every = 100000
    n_val_logs = 10
    timestamp = time.strftime("%y%m%d-%H:%M:%S", time.gmtime())
    if corpus == "MSVD":
        n_val = 100
        n_test = 670

    """ ID """
    corpus_id = "{} tc-{} mc-{} sp-{}".format(corpus, caption_n_max_word, min_count, frame_sampling_method)
    encoder_id = "ENC {} sm-{}".format(encoder_model, encoder_output_len)
    decoder_id = "DEC {}-{} at-{} dr-{}-{} tf-{} lr-{}-wd-{} op-{}".format(
        decoder_model.lower(), decoder_n_layers, decoder_attn_size, decoder_dropout, decoder_out_dropout,
        decoder_teacher_forcing_ratio, decoder_learning_rate, decoder_weight_decay,
        ["adam", "amsgrad"][decoder_use_amsgrad])
    reconstructor_id = "REC lr-{}-wd-{} op-{}".format(
        reconstructor_learning_rate, reconstructor_weight_decay, ["adam", "amsgrad"][reconstructor_use_amsgrad])
    embedding_id = "EMB {} dr-{} sc-{}".format(embedding_size, embedding_dropout, embedding_scale)
    hyperparams_id = "bs-{}".format(batch_size)
    if use_gradient_clip:
        hyperparams_id = "{} | cp-{}".format(hyperparams_id, clip)

    if use_recon:
        id = " | ".join([ model, corpus_id, encoder_id, decoder_id, reconstructor_id, embedding_id,
                          hyperparams_id, timestamp ])
    else:
        id = " | ".join([ model, corpus_id, encoder_id, decoder_id, embedding_id, hyperparams_id,
                          timestamp ])
    train_log_dpath = "logs/{} | TRAIN".format(id)
    val_log_dpath = "logs/{} | VAL".format(id)
    Bleu1_log_dpath = "logs/{} | BLEU-1".format(id)
    Bleu2_log_dpath = "logs/{} | BLEU-2".format(id)
    Bleu3_log_dpath = "logs/{} | BLEU-3".format(id)
    Bleu4_log_dpath = "logs/{} | BLEU-4".format(id)
    CIDEr_log_dpath = "logs/{} | CIDEr".format(id)
    METEOR_log_dpath = "logs/{} | METEOR".format(id)
    ROUGE_L_log_dpath = "logs/{} | ROUGE_L".format(id)
    save_dpath = "checkpoints/{}".format(id)

    """ TensorboardX """
    tx_loss = "loss/total"
    tx_loss_decoder = "loss/decoder"
    tx_loss_reconstructor = "loss/reconstructor"
    tx_predicted_captions = "Ground Truths v.s. Predicted Captions"
    tx_lambda_decoder = "lambda/decoder_regularizer"
    tx_lambda_reconstructor = "lambda/reconstructor_regularizer"
    tx_lambda = "lambda/reconstructor"
    tx_score_Bleu1 = "score"
    tx_score_Bleu2 = "score"
    tx_score_Bleu3 = "score"
    tx_score_Bleu4 = "score"
    tx_score_CIDEr = "score"
    tx_score_METEOR = "score"
    tx_score_ROUGE_L = "score"


class EvalConfig:
    device = "cuda"

    """ Data Loader """
    test_video_fpath = lambda TC: "data/{}/features/{}_test.hdf5".format(TC.corpus, TC.encoder_model)
    test_caption_fpath = lambda TC: "data/{}/metadata/test.csv".format(TC.corpus)

    """ Model """
    model_dpath = "checkpoints"
    model_id = "RecNet | MSVD tc-30 mc-5 | ENC InceptionV4 sm-33 | DEC lstm-2 dr-0.5-0.5 tf-1.0 lr-0.0001-wd-1e-05 op-amsgrad | EMB 468 dr-0.5 sc-1 | bs-100 | cp-50.0 | 181105-11:49:01"
    model_iteration = 100000
    model_fpath = "{}/{}/{}_checkpoint.tar".format(model_dpath, model_id, model_iteration)

