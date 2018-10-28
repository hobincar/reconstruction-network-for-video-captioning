from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import cycle

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch import optim
import random
import os

from config import TrainConfig as C
from dataset.MSVD import MSVD as MSVD_dataset
from losses import maskNLLLoss
from models.decoder import Decoder


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def forward_decoder(encoder_outputs, target, target_mask, target_max_n_words, decoder, embedding, is_train):
    # Set device options
    encoder_outputs = encoder_outputs.to(device)
    target = target.to(device)
    target_mask = target_mask.to(device)

    # Initialize variables
    loss = 0
    loss_vals = []
    n_totals = 0
    decoder_output_indices_list = []

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[ C.init_word2idx['<SOS>'] for _ in range(C.batch_size) ]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = torch.zeros(C.decoder_n_layers, C.batch_size, C.decoder_hidden_size)
    decoder_hidden = decoder_hidden.to(device)

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = random.random() < C.decoder_teacher_forcing_ratio

    # Forward batch of sequences through decoder one time step at a time
    for t in range(target_max_n_words):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        if is_train and use_teacher_forcing:
            # Teacher forcing: next input is current target
            decoder_input = target[t].view(1, -1)
        else:
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_output_indices = [ topi[i][0] for i in range(C.batch_size) ]
            decoder_input = torch.LongTensor([ decoder_output_indices ])
            decoder_input = decoder_input.to(device)
            decoder_output_indices_list.append(decoder_output_indices)
        # Calculate and accumulate loss
        mask_loss, n_total = maskNLLLoss(decoder_output, target[t], target_mask[t], device)
        loss += mask_loss
        loss_vals.append(mask_loss.item() * n_total)
        n_totals += n_total
    loss_val = sum(loss_vals) / n_totals

    decoder_output_indices_list = torch.LongTensor(decoder_output_indices_list)
    return loss, loss_val, decoder_output_indices_list


def convert_preds_to_captions(preds):
    pred_captions = []
    pred_indices_list = np.asarray(preds).T
    for pred_indices in pred_indices_list:
        pred_captions.append([])
        for pred_idx in pred_indices:
            if pred_idx == 0: continue
            pred_captions[-1].append(vocab.idx2word[pred_idx])
        pred_captions[-1] = " ".join(pred_captions[-1])
    return pred_captions


def sample_n(lst, n):
    indices = list(range(len(lst)))
    sample_indices = np.random.choice(indices, n, replace=False)
    sample_lst = [ lst[i] for i in sample_indices ]
    return sample_lst


if __name__ == "__main__":
    writer = SummaryWriter(C.log_dpath)

    dataset = MSVD_dataset(C)
    vocab = dataset.vocab
    train_data_loader = cycle(iter(dataset.train_data_loader))
    val_data_loader = cycle(iter(dataset.val_data_loader))
    print('n_vocabs: {}, n_words: {}'.format(vocab.n_vocabs, vocab.n_words))

    embedding = nn.Embedding(vocab.n_vocabs, C.word_embedding_size)

    decoder = Decoder(
        attn_model=C.attn_model,
        encoder_output_size=C.encoder_output_size,
        embedding=embedding,
        embedding_dropout=C.embedding_dropout,
        input_size=C.word_embedding_size,
        hidden_size=C.decoder_hidden_size,
        output_size=vocab.n_vocabs,
        n_layers=C.decoder_n_layers,
        dropout=C.decoder_dropout)
    decoder = decoder.to(device)
    decoder.train()
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=C.learning_rate * C.decoder_learning_ratio)
    
    losses = []
    decoder_losses = []
    for iteration, batch in enumerate(train_data_loader, 1):

        decoder_optimizer.zero_grad()

        encoder_outputs, targets, target_masks, target_max_n_words = batch
        """
        print("encoder_outputs: ", encoder_outputs)
        print("targets: ", targets)
        print("target_masks: ", target_masks)
        print("target_max_n_words: ", target_max_n_words)
        assert 0
        """

        decoder_loss, decoder_loss_val, _ = forward_decoder(encoder_outputs, targets, target_masks,
                                                            target_max_n_words, decoder, embedding, is_train=True)

        # Perform backpropatation
        loss = decoder_loss
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), C.clip)
        decoder_optimizer.step()

        loss_val = decoder_loss_val
        losses.append(loss_val)
        decoder_losses.append(decoder_loss_val)

        # Print progress
        if iteration % C.log_every == 0:
            loss_avg = np.mean(losses)
            decoder_loss_avg = np.mean(decoder_losses)

            writer.add_scalar(C.tx_train_loss, loss_avg, iteration)
            writer.add_scalar(C.tx_train_loss_decoder, decoder_loss_avg, iteration)
            print("Iter {} / {} ({:.1f}%): loss {:.3f} | decoder_loss {:.3f}".format(
                iteration, C.train_n_iteration, iteration / C.train_n_iteration * 100, loss_avg, decoder_loss_avg))
            losses = []
            decoder_losses = []

        # Validate model
        if iteration % C.validate_every == 0:
            losses = []
            decoder_losses = []
            gt_captions = []
            pred_captions = []
            for i, batch in enumerate(val_data_loader):
                encoder_outputs, targets, target_masks, target_max_n_words = batch

                _, decoder_loss_val, preds = forward_decoder(encoder_outputs, targets, target_masks,
                                                             target_max_n_words, decoder, embedding,
                                                             is_train=False)

                loss_val = decoder_loss_val

                losses.append(loss_val)
                decoder_losses.append(decoder_loss_val)
                gt_captions += convert_preds_to_captions(targets.numpy())
                pred_captions += convert_preds_to_captions(preds.numpy())

                if i == C.val_n_iteration:
                    break
            loss_avg = np.mean(losses)
            decoder_loss_avg = np.mean(decoder_losses)

            writer.add_scalar(C.tx_val_loss, loss_avg, iteration)
            writer.add_scalar(C.tx_val_loss_decoder, decoder_loss_avg, iteration)
            print("[Validation] Iter {} / {} ({:.2f}%): loss {:.3f} | decoder_loss {:.3f}".format(
                iteration, C.train_n_iteration, iteration / C.train_n_iteration * 100, loss_avg, decoder_loss_avg))

            caption_pairs = [ (gt, pred) for gt, pred in zip(gt_captions, pred_captions) ]
            caption_pairs = sample_n(caption_pairs, C.n_val_logs)
            caption_log = "\n\n".join([ "[GT] {}  \n[PR] {}".format(gt, pred) for gt, pred in caption_pairs ])
            writer.add_text(C.tx_predicted_captions, caption_log, iteration)


        # Save checkpoint
        if iteration % C.save_every == 0:
            if not os.path.exists(C.save_dpath):
                os.makedirs(C.save_dpath)
            fpath = os.path.join(C.save_dpath, "{}_checkpoint.tar".format(iteration))

            torch.save({
                'iteration': iteration,
                'dec': decoder.state_dict(),
                'dec_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'embedding': embedding.state_dict()
            }, fpath)

        if iteration == C.train_n_iteration:
            break

