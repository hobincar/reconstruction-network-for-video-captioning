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
    model = "RecNet"
    corpus = "MSVD" # [ "MSVD" ]
    encoder_model = "InceptionV4" # [ "InceptionV4" ]
    decoder_model = "LSTM" # [ "LSTM", "GRU" ]
    device = "cuda"

    """ Data Loader """
    total_video_fpath = "data/{}/features/{}.hdf5".format(corpus, encoder_model)
    total_caption_fpath = "data/{}/metadata/MSR Video Description Corpus.csv".format(corpus)
    train_video_fpath = "data/{}/features/{}_train.hdf5".format(corpus, encoder_model)
    train_caption_fpath = "data/{}/metadata/train.csv".format(corpus)
    val_video_fpath = "data/{}/features/{}_val.hdf5".format(corpus, encoder_model)
    val_caption_fpath = "data/{}/metadata/val.csv".format(corpus)
    min_count = 5 # N_vocabs = 1: 13501 | 2: 7424 | 3: 5692 | 4: 4191 | 5: 4188
    caption_n_max_word = 30
    batch_size = 100
    val_n_iteration = 1
    shuffle = True
    num_workers = 10

    """ Train """
    train_n_iteration = 100000
    decoder_learning_rate = 1e-5
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
    decoder_dropout = 0
    decoder_out_dropout = 0.5
    decoder_weight_decay = 1e-5
    decoder_teacher_forcing_ratio = 1.0

    """ Reconstructor """
    use_recon = False
    reconstructor_type = "global"
    reconstructor_n_layers = 1
    reconstructor_hidden_size = 1536
    reconstructor_dropout = 0

    """ Log """
    log_every = 100
    validate_every = 1000
    save_every = 100000
    n_val_logs = 3
    timestamp = time.strftime("%y%m%d-%H:%M:%S", time.gmtime())

    """ ID """
    encoder_id = "ENC {} sm-{}".format(encoder_model, encoder_output_len)
    decoder_id = "DEC {}-{} dr-{}-{} tf-{} lr-{}-wd-{} op-{}".format(
        decoder_model.lower(), decoder_n_layers, decoder_dropout, decoder_out_dropout,
        decoder_teacher_forcing_ratio, decoder_learning_rate, decoder_weight_decay,
        ["adam", "amsgrad"][decoder_use_amsgrad])
    reconstructor_id = "REC lr-{}-wd-{} op-{}".format(
        reconstructor_learning_rate, reconstructor_weight_decay, ["adam", "amsgrad"][reconstructor_use_amsgrad])
    embedding_id = "EMB {} dr-{} sc-{}".format(embedding_size, embedding_dropout, embedding_scale)
    hyperparams_id = "bs-{}".format(batch_size)
    if use_gradient_clip:
        hyperparams_id = "{} | cp-{}".format(hyperparams_id, clip)

    if use_recon:
        id = " | ".join([ model, corpus, encoder_id, decoder_id, reconstructor_id, embedding_id,
                          hyperparams_id, timestamp ])
        train_id = "{} | TRAIN".format(id)
        val_id = "{} | VAL".format(id)
    else:
        id = " | ".join([ model, corpus, encoder_id, decoder_id, embedding_id, hyperparams_id,
                          timestamp ])
        train_id = "{} | TRAIN".format(id)
        val_id = "{} | VAL".format(id)
    train_log_dpath = "logs/{}".format(train_id)
    val_log_dpath = "logs/{}".format(val_id)
    save_dpath = "checkpoints/{}".format(id)

    """ TensorboardX """
    tx_loss = "loss/total"
    tx_loss_decoder = "loss/decoder"
    tx_loss_reconstructor = "loss/reconstructor"
    tx_predicted_captions = "Ground Truths v.s. Predicted Captions"
    tx_lambda_decoder = "lambda/decoder_regularizer"
    tx_lambda_reconstructor = "lambda/reconstructor_regularizer"
    tx_lambda = "lambda/reconstructor"

