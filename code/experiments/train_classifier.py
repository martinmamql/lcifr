from datetime import datetime
from os import makedirs, path

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from experiments.args_factory import get_args
from metrics import equalized_odds, statistical_parity
from models import Autoencoder, LogisticRegression
from utils import Statistics

args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert args.conditional_training==False # Should not enable conditional training here

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
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False
)

autoencoder = Autoencoder(args.encoder_layers, args.decoder_layers)
classifier = LogisticRegression(args.encoder_layers[-1])

for param in autoencoder.parameters():
    param.requires_grad_(False)

autoencoder.load_state_dict(
    torch.load(
        path.join(models_dir, f'autoencoder_{args.load_epoch}.pt'),
        map_location=lambda storage, loc: storage
    )
)

binary_cross_entropy = nn.BCEWithLogitsLoss(
    pos_weight=train_dataset.pos_weight('train') if args.balanced else None
)
optimizer = torch.optim.Adam(
    classifier.parameters(), lr=args.learning_rate,
    weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=args.patience, factor=0.5
)


def run(autoencoder, classifier, optimizer, loader, split):
    predictions, targets = list(), list()
    tot_ce_loss, tot_stat_par, tot_eq_odds = Statistics.get_stats(3)

    progress_bar = tqdm(loader)

    for data_batch, targets_batch, protected_batch in progress_bar:
        batch_size = data_batch.shape[0]
        data_batch = data_batch.to(device)
        targets_batch = targets_batch.to(device)
        protected_batch = protected_batch.to(device)

        if split == 'train':
            classifier.train()

        latent_data = autoencoder.encode(data_batch)

        logits = classifier(latent_data)
        ce_loss = binary_cross_entropy(logits, targets_batch)
        predictions_batch = classifier.predict(latent_data)

        stat_par = statistical_parity(predictions_batch, protected_batch)
        eq_odds = equalized_odds(
            targets_batch, predictions_batch, protected_batch
        )

        predictions.append(predictions_batch.detach().cpu())
        targets.append(targets_batch.detach().cpu())

        if split == 'train':
            optimizer.zero_grad()
            ce_loss.mean().backward()
            optimizer.step()

        tot_ce_loss.add(ce_loss.mean().item())
        tot_stat_par.add(stat_par.mean().item())
        tot_eq_odds.add(eq_odds.mean().item())

        progress_bar.set_description(
            f'[{split}] epoch={epoch:d}, ce_loss={tot_ce_loss.mean():.4f}'
        )

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    accuracy = accuracy_score(targets, predictions)
    balanced_accuracy = balanced_accuracy_score(targets, predictions)
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    f1 = f1_score(targets, predictions)

    writer.add_scalar('Accuracy/%s' % split, accuracy, epoch)
    writer.add_scalar('Balanced Accuracy/%s' % split, balanced_accuracy, epoch)
    writer.add_scalar('Cross Entropy/%s' % split, tot_ce_loss.mean(), epoch)
    writer.add_scalar('True Positives/%s' % split, tp, epoch)
    writer.add_scalar('False Negatives/%s' % split, fn, epoch)
    writer.add_scalar('True Negatives/%s' % split, tn, epoch)
    writer.add_scalar('False Positives/%s' % split, fp, epoch)
    writer.add_scalar('F1 Score/%s' % split, f1, epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('Stat. Parity/%s' % split, tot_stat_par.mean(), epoch)
    writer.add_scalar('Equalized Odds/%s' % split, tot_eq_odds.mean(), epoch)

    return tot_ce_loss


print('saving model to', models_dir)
writer = SummaryWriter(log_dir)

for epoch in range(args.num_epochs):
    run(autoencoder, classifier, optimizer, train_loader, 'train')

    autoencoder.eval()
    classifier.eval()

    valid_loss = run(autoencoder, classifier, optimizer, val_loader, 'valid')
    scheduler.step(valid_loss.mean())

    prefix = f'{args.label}_' if args.label else ''
    postfix = '_robust' if args.adversarial else ''

    torch.save(
        autoencoder.state_dict(), path.join(
            models_dir, prefix + f'autoencoder_{epoch}' + postfix + '.pt'
        )
    )
    torch.save(
        classifier.state_dict(), path.join(
            models_dir, prefix + f'classifier_{epoch}' + postfix + '.pt'
        )
    )

writer.close()
