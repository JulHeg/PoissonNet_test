import torch
import os
from src.models import FNO3d

def PoissonNet(sample_size = 25000, cuda = True):
    # dirname = os.path.dirname(__file__)
    # checkpoint_dir = os.path.join(dirname, 'weights')
    # filenames = os.listdir(checkpoint_dir)
    # filename = None
    # for f in filenames:
    #     if f.startswith('n_samples_' + str(sample_size)):
    #         filename = f
    #         break
    # if filename is None:
    #     raise ValueError('No checkpoint found for sample size: {}'.format(sample_size))
    # checkpoint_path = os.path.join(checkpoint_dir, filename)
    # state_dict = torch.load(checkpoint_path)
    # model = FNO3d.load_state_dict(state_dict)
    BASE_URL = 'https://juliushege.com/files/'
    filenames = {
        5000: 'n_samples_5000_sigma_2',
        10000: 'n_samples_10000_sigma_1.5',
        25000: 'n_samples_25000_sigma_1.0',
        50000: 'n_samples_50000_sigma_0.7',
        100000: 'n_samples_100000_sigma_0.5',
    }
    if sample_size not in filenames:
        raise ValueError('Sample size {} not supported. Supported sample sizes are: {}'.format(sample_size, filenames.keys()))
    URL = BASE_URL + filenames[sample_size]
    state_dict = torch.hub.load_state_dict_from_url(URL)
    width = 30
    modes = 10
    model = FNO3d(modes, modes, modes, width)
    if cuda:
        model = model.cuda()
    model.load_state_dict(state_dict)
    return model