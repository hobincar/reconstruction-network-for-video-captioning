import time

class SplitConfig:
    corpus_name = "MSVD"
    encoder_model_name = "InceptionV4"

    video_fpath = "data/{}/features/{}.hdf5".format(corpus_name, encoder_model_name)
    caption_fpath = "data/{}/metadata/MSR Video Description Corpus.csv".format(corpus_name)

    random_seed = 42
    n_train = 1200
    n_val = 100
    n_test = 670

    train_video_fpath = "data/{}/features/{}_train.hdf5".format(corpus_name, encoder_model_name)
    val_video_fpath = "data/{}/features/{}_val.hdf5".format(corpus_name, encoder_model_name)
    test_video_fpath = "data/{}/features/{}_test.hdf5".format(corpus_name, encoder_model_name)

    train_metadata_fpath = "data/{}/metadata/train.csv".format(corpus_name)
    val_metadata_fpath = "data/{}/metadata/val.csv".format(corpus_name)
    test_metadata_fpath = "data/{}/metadata/test.csv".format(corpus_name)


class TrainConfig:
    model_name = "RecNet"
    corpus_name = "MSVD"
    encoder_model_name = "InceptionV4"

    """ Data Loader """
    total_video_fpath = "data/{}/features/{}.hdf5".format(corpus_name, encoder_model_name)
    total_caption_fpath = "data/{}/metadata/MSR Video Description Corpus.csv".format(corpus_name)
    train_video_fpath = "data/{}/features/{}_train.hdf5".format(corpus_name, encoder_model_name)
    train_caption_fpath = "data/{}/metadata/train.csv".format(corpus_name)
    val_video_fpath = "data/{}/features/{}_val.hdf5".format(corpus_name, encoder_model_name)
    val_caption_fpath = "data/{}/metadata/val.csv".format(corpus_name)
    min_count = 5 # N_vocabs = 1: 13501 | 2: 7424 | 3: 5692 | 4: 4191 | 5: 4188
    caption_n_max_word = 30
    batch_size = 100
    val_n_iteration = 1
    shuffle = True
    num_workers = 10

    """ Train """
    train_n_iteration = 100000
    learning_rate = 1e-6
    clip = 50.0 # Gradient clipping

    """ Word Embedding """
    init_word2idx = { '<PAD>': 0, '<SOS>': 1, '<EOS>': 2 }
    word_embedding_size = 468
    embedding_dropout = 0

    """ Encoder """
    encoder_output_size = 1536
    encoder_output_len = 28

    """ Decoder """
    decoder_hidden_size = 512
    decoder_n_layers = 1
    decoder_dropout = 0
    decoder_learning_ratio = 1
    decoder_teacher_forcing_ratio = 1.0

    """ Reconstructor """
    use_recon = True
    reconstructor_type = "global"
    reconstructor_n_layers = 1
    reconstructor_hidden_size = 1536
    reconstructor_dropout = 0
    reconstructor_learning_ratio = 0.1
    loss_lambda = 0.2

    """ Log """
    log_every = 100
    validate_every = 1000
    save_every = 10000
    n_val_logs = 3
    timestamp = time.strftime("%y%m%d-%H:%M:%S", time.gmtime())
    model_id = "{} | {} | ENC_{} | DEC_LSTM-{}_dr-{}_tf-{}_lr-{} | REC-{}_lr-{} | EMB_{}_dr-{} | bs-{} | ld-{} | {}".format(
        model_name,
        corpus_name,
        encoder_model_name,
        decoder_n_layers, decoder_dropout, decoder_teacher_forcing_ratio, learning_rate * decoder_learning_ratio,
        ['OFF', 'ON'][use_recon], learning_rate * reconstructor_learning_ratio,
        word_embedding_size, embedding_dropout,
        batch_size, loss_lambda, timestamp)
    log_dpath = "logs/{}".format(model_id)
    save_dpath = "checkpoints/{}".format(model_id)

    """ TensorboardX """
    tx_train_loss = "loss/train/total"
    tx_train_loss_decoder = "loss/train/decoder"
    tx_train_loss_reconstructor = "loss/train/reconstructor"
    tx_val_loss = "loss/val/total"
    tx_val_loss_decoder = "loss/val/decoder"
    tx_val_loss_reconstructor = "loss/val/reconstructor"
    tx_predicted_captions = "Ground Truths v.s. Predicted Captions"


