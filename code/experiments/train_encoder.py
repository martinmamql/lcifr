from datetime import datetime
from os import makedirs, path

import torch.nn as nn
import torch.utils.data
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from constraints import ConstraintBuilder
from dl2.training.supervised.oracles import DL2_Oracle
from experiments.args_factory import get_args
from metrics import equalized_odds, statistical_parity
from models import Autoencoder, LogisticRegression
from utils import Statistics

from datasets import ConditionalBatchSampler # Conditional contrastive learning sampler


args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

project_root = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
base_dir = path.join(
    f'{args.dataset}', args.constraint,
    '_'.join([str(l) for l in args.encoder_layers + args.decoder_layers[1:]]),
    f'dl2_weight_{args.dl2_weight}_learning_rate_{args.learning_rate}_'
    f'weight_decay_{args.weight_decay}_balanced_{args.balanced}_'
    f'patience_{args.patience}_quantiles_{args.quantiles}_'
    f'dec_weight_{args.dec_weight}'
)
models_dir = path.join(
    args.models_base if args.models_base else project_root, 'models', base_dir
)
makedirs(models_dir, exist_ok=True)
log_dir = path.join(
    project_root, 'logs', base_dir,
    datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
)

dataset = getattr(
    __import__('datasets'), args.dataset.capitalize() + 'Dataset'
)
train_dataset = dataset('train', args)
val_dataset = dataset('validation', args)

# A custom batch sampler for conditional contrastive learning
if args.conditional_training:
    conditional_sampler = ConditionalBatchSampler(
        train_dataset, args.batch_size, train_dataset.protected_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=conditional_sampler)
    print("Conditional contrastive learning enabled")
else:
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    print("Conditional contrastive learning disabled")

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False
)

autoencoder = Autoencoder(args.encoder_layers, args.decoder_layers)
cross_entropy = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(autoencoder.parameters()),
    lr=args.learning_rate, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=args.patience, factor=0.5
)


def run(autoencoder, optimizer, loader, split, epoch):
    contrastive_loss = Statistics.get_stats(1)

    progress_bar = tqdm(loader)

    for data_batch, targets_batch, protected_batch in progress_bar:
        batch_size = data_batch.shape[0]
        data_batch = data_batch.to(device)
        targets_batch = targets_batch.to(device)
        protected_batch = protected_batch.to(device)

        # A data batch with noise, to create positive and negative pairs
        data_batch_with_noise = data_batch + \
          torch.normal(mean=torch.zeros(data_batch.shape), std=torch.ones(data_batch.shape)*args.contrastive_noise).to(device)

        if split == 'train':
            autoencoder.train()

        latent_data = autoencoder.encode(data_batch) #bs, h_dim
        latent_data_with_noise = autoencoder.encode(data_batch_with_noise) #bs, h_dim
        
        # Positive pair: a sample and its variant with added noise
        # Negative pairs: a sample with all other noise-added variants except its own variant
        score = torch.matmul(latent_data, latent_data_with_noise.T)
        contrastive_loss = cross_entropy(score, torch.arange(batch_size).to(device)) 

        autoencoder.train()

        optimizer.zero_grad()
        contrastive_loss.backward()
        optimizer.step()

        progress_bar.set_description(
            f'[{split}] epoch={epoch:d}, contrastive_loss={contrastive_loss.mean():.4f}, '
        )

    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('Contrastive Loss/%s' % split, contrastive_loss.mean(), epoch)

    return contrastive_loss


print('saving model to', models_dir)
writer = SummaryWriter(log_dir)

for epoch in range(args.num_epochs):
    run(autoencoder, optimizer, train_loader, 'train', epoch)

    autoencoder.eval()

    torch.save(
        autoencoder.state_dict(),
        path.join(models_dir, f'autoencoder_{epoch}.pt')
    )
writer.close()
