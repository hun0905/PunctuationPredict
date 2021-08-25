from model import BRNN
import torch
import argparse
import time
import os
from dataset import build_data_loader
parser = argparse.ArgumentParser()

parser.add_argument('--train_data', type=str, required=True, help='Training text data path.')
parser.add_argument('--valid_data', type=str, required=True, help='Cross validation text data path.')
parser.add_argument('--vocab', type=str, required=True, help='Input vocab. (Don\'t include <UNK> and <END>)')
parser.add_argument('--punc_vocab', type=str, required=True, help='Output punctuations vocab. (Don\'t include " ")')
# Model hyper parameters
parser.add_argument('--num_embeddings', default=30000+2, type=int, help='Input vocab size. (Include <UNK> and <END>)')
parser.add_argument('--embedding_dim', default=256, type=int, help='Input embedding dim.')
parser.add_argument('--hidden_size', default=512, type=int, help='LSTM hidden size of each direction.')
parser.add_argument('--num_layers', default=2, type=int, help='Number of LSTM layers')
parser.add_argument('--bidirectional', default=1, type=int, help='Whether use bidirectional LSTM')
parser.add_argument('--num_class', default=5, type=int, help='Number of output classes. (Include blank space " ")')
# minibatch
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to generate minibatch')####有改

# optimizer
parser.add_argument('--lr', default=1e-3, type=float, help='Init learning rate')
parser.add_argument('--l2', default=0.0, type=float, help='weight decay (L2 penalty)')
# Training config
parser.add_argument('--use_cuda', default=1, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--half_lr', default=0, type=int, help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', default=0, type=int, help='Early stop training when halving lr but still get small improvement')
parser.add_argument('--max_norm', default=5, type=float, help='Gradient norm threshold to clip')
# save and load model
parser.add_argument('--save_folder', default='exp/temp', help='Dir to save models')
parser.add_argument('--checkpoint', default=0, type=int, help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar', help='model name')
# logging
parser.add_argument('--print_freq', default=10, type=int, help='Frequency of printing training infomation')

# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"

class Solver(object):
    
    def __init__(self, data, model, criterion, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.start_epoch = 1####
        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            model = self.model.module if self.use_cuda else self.model
            model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0
    def train(self):
        # Train model multi-epoches
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            

            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            self.tr_loss[epoch] = tr_avg_loss
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                model = self.model.module if self.use_cuda else self.model
                torch.save(model.serialize(model,
                                           self.optimizer, epoch + 1,
                                           tr_loss=self.tr_loss,
                                           cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)
            
            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)

            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Adjust learning rate (halving)
            if self.half_lr and val_loss >= self.prev_val_loss:
                if self.early_stop and self.halving:
                    print("Already start halving learing rate, it still gets "
                          "too small imporvement, stop training early.")
                    break
                self.halving = True
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
            self.prev_val_loss = val_loss

            # Save the best model
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                model = self.model.module if self.use_cuda else self.model
                torch.save(model.serialize(model,
                                           self.optimizer, epoch + 1,
                                           tr_loss=self.tr_loss,
                                           cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)
            
    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        # Main Loop
        for  i,(data) in enumerate(data_loader):
            padded_input, input_lengths, padded_target = data
            if self.use_cuda:
                padded_input = padded_input.cuda()
                input_lengths = input_lengths.cuda()
                padded_target = padded_target.cuda()
                padded_input = padded_input.to(self.device, dtype=torch.long)
                input_lengths = input_lengths.to(self.device,dtype = torch.long)
                padded_target = padded_target.to(self.device,dtype = torch.long)
            #print(padded_input,input_lengths,padded_target)
            pred = self.model(padded_input, input_lengths)
            pred = pred.view(-1, pred.size(-1))
            loss = self.criterion(pred, padded_target.view(-1))
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)
        return total_loss / (i + 1)
        
def num_param(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params




def main(args):
    # Build data loader
    tr_loader = build_data_loader(args.train_data, args.vocab, args.punc_vocab,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=args.num_workers)
    cv_loader = build_data_loader(args.valid_data, args.vocab, args.punc_vocab,
                                  batch_size=args.batch_size, drop_last=False)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    
    model = BRNN(args.num_embeddings, args.embedding_dim,
                # Build model
                args.hidden_size, args.num_layers, args.bidirectional,
                args.num_class)
    print(model)
    print("Number of parameters: %d" % num_param(model))
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # Build criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # Build optimizer
    optimizier = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.l2)
    # Build Solver
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    solver = Solver(data, model, criterion, optimizier, args)
    solver.train()
    #print(data['cv_loader'])
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)