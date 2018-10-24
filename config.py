class TrainConfig:
    def __init__(self):
        self.max_length = 10
        self.save_dir = "data/save"
        self.corpus_name = "cornell movie-dialogs corpus"
        
        self.model_name = 'cb_model'
        self.attn_model = 'dot'
        self.hidden_size = 500
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        self.batch_size = 64

        self.loadFilename = None
        self.checkpoint_iter = 4000
        
        self.clip = 50.0
        self.teacher_forcing_ratio = 1.0
        self.learning_rate = 0.0001
        self.decoder_learning_ratio = 5.0
        self.n_iteration = 10
        self.print_every = 1
        self.save_every = 500

