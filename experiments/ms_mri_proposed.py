from data.msmri_dataset import msmri_Dataloader
from utils.OTloss import OT_loss
import data.transformations as transformations
import torchvision.transforms as transforms
from models.MoSE import MoSE

experiment_name = 'MoSE_run_msmri'

# Setup directories
log_root = './log'
run_root = './runs'
log_dir_name = 'ms_mri'
data_folder = '../data/ms_mri/'

# Setup data
data_loader = msmri_Dataloader
n_classes = 2
input_channels = 4
image_size = (64, 64)
num_w = 4  # num workers
label_range = 'all'
eval_class_ids = [1]

transform = {
    'train':  transforms.Compose([
              transformations.RandomHorizontalFlip(),
              transformations.RandomVerticalFlip(),
              transformations.RandomRotation(angle=10),

              ]),
    'val':  None,
    'test': None
}

# loss function
loss_fn = OT_loss( cost_fn='iou', beta = 1, gamma0=1/2)

# Setup Model
net = MoSE(input_channels=input_channels,
                num_classes=n_classes,
                num_filters=[32, 64, 128, 192, 192, 192, 192],
                gating_input_layer = 4,
                latent_dim = 1,
                num_expert = 4,
                sample_per_mode=1,
                loss_fn = loss_fn,
                masked_pred = False,
                eval_class_ids = eval_class_ids,
                softmax = True,
                eval_sample_num = None,
                )
# Training settings
pretrained_model_full_path = ''

train_batch_size =16
val_batch_size =16
test_batch_size = 4
epochs = 200

lr = 1e-3
scheduler_options = {'min_lr': 1e-5,
                     'patience':10,
                     'factor': 0.5}

gamma_scheduler_options = {'updated_values': [1],
                            'patience': 10}

logging_frequency = 1
validation_frequency = 1
