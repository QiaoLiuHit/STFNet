class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/qiao/code/TransT-Fusion'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/media/qiao/dataset/TrackingBenchmark/LaSOT'
        self.got10k_dir = '/media/qiao/dataset/TrackingBenchmark/GOT-10k'
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
