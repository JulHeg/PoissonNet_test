import torch
import os
from src import models

def PoissonNet(**kwargs):
    poissonnetFNO = PoissonNetFNO(**kwargs)
    poissonnet = models.PoissonNet(poissonnetFNO)
    return poissonnet
    
def PoissonNetFNO(sample_size = 25000, cuda = True, width = 30, modes = 10):
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
    try:
        state_dict = torch.hub.load_state_dict_from_url(URL)
    except:
        raise ValueError("Could not load model weights, presumably because of the authors' failure to find durable hosting. Try downloading the weights manually from https://drive.google.com/drive/folders/1xQGvyRUg2davxsip1SyNnCkoe0kzeEJM.")
    model = models.FNO3d(modes, modes, modes, width)
    if cuda:
        model = model.cuda()
    model.load_state_dict(state_dict)
    return model