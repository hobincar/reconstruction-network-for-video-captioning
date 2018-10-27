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
    caption_n_max_word = 30
    batch_size = 40
    shuffle = True
    num_workers = 40

    """ Train """
    n_iteration = 1000000
    learning_rate = 1e-4
    clip = 50.0 # Gradient clipping

    """ Encoder """
    encoder_output_size = 1536
    encoder_sample_size = 28
    encoder_output_len = 28

    """ Word Embedding """
    init_word2idx = { '<PAD>': 0, '<SOS>': 1, '<EOS>': 2 }
    word_embedding_size = 468
    embedding_dropout = 0

    """ Attention """
    attn_model = 'general' # One of ['general', 'dot', 'concat']

    """ Decoder """
    decoder_hidden_size = 512
    decoder_n_layers = 1
    decoder_dropout = 0
    decoder_learning_ratio = 1
    decoder_teacher_forcing_ratio = 1.0

    """ Log """
    log_every = 100
    save_every = 100000
    timestamp = time.strftime("%y%m%d-%H:%M:%S", time.gmtime())
    model_id = "{} | enc_{} | dec_LSTM-{} | emb_{} | {} | {}".format(
        model_name, encoder_model_name, decoder_n_layers, word_embedding_size, corpus_name, timestamp)
    log_dpath = "logs/{}".format(model_id)
    save_dpath = "checkpoints/{}".format(model_id)

    """ Validation """
    validate_every = 10000
    n_logs = 3

    """ TensorboardX """
    tx_train_loss = "loss/train/total"
    tx_train_loss_decoder = "loss/train/decoder"
    tx_val_loss = "loss/val/total"
    tx_val_loss_decoder = "loss/val/decoder"
    tx_predicted_captions = "Ground Truths v.s. Predicted Captions"


