from ltr.admin.environment import env_settings


class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()
        # Most common settings are assigned in the settings struct
        self.device = 'cuda'
        self.description = 'STFNet with default settings.'
        self.batch_size = 8
        self.num_workers = 4
        self.multi_gpu = False
        self.print_interval = 1
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

        # Transformer
        self.position_embedding = 'sine'
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 2048
        self.featurefusion_layers = 4

    def set_default(self):
        self.env = env_settings()
        self.use_gpu = True


