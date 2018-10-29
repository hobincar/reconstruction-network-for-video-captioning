from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from itertools import cycle
import os
import random

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch import optim

from config import TrainConfig as C
from dataset.MSVD import MSVD as MSVD_dataset
from models.attn_decoder import Decoder
from models.global_reconstructor import GlobalReconstructor


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

torch.multiprocessing.set_sharing_strategy('file_system')


def forward_decoder(encoder_outputs, targets, target_masks, target_max_n_words, decoder, loss_func, is_train):
    # Set device options
    encoder_outputs = encoder_outputs.to(device)
    targets = targets.to(device)
    target_masks = target_masks.to(device)

    # Initialize variables
    loss = 0
    n_totals = 0
    hiddens = []
    output_indices = []

    # Create initial decoder input (start with SOS tokens for each sentence)
    input = torch.LongTensor([[ C.init_word2idx['<SOS>'] for _ in range(C.batch_size) ]])
    input = input.to(device)

    hidden = torch.zeros(C.decoder_n_layers, C.batch_size, C.decoder_hidden_size)
    context = torch.zeros(C.decoder_n_layers, C.batch_size, C.decoder_hidden_size)
    hidden = hidden.to(device)
    context = context.to(device)

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = random.random() < C.decoder_teacher_forcing_ratio

    # Forward batch of sequences through decoder one time step at a time
    for t in range(target_max_n_words):
        output, (hidden, context), _ = decoder(
            input, (hidden, context), encoder_outputs
        )
        if is_train and use_teacher_forcing:
            # Teacher forcing: next input is current target
            input = targets[t].view(1, -1)
        else:
            # No teacher forcing: next input is decoder's own current output
            _, topi = output.topk(1)
            output_index = [ topi[i][0] for i in range(C.batch_size) ]
            input = torch.LongTensor([ output_index ])
            input = input.to(device)
            output_indices.append(output_index)

        # Calculate and accumulate loss
        masked_output = output[target_masks[t]]
        masked_target = targets[t][target_masks[t]]
        masked_loss = loss_func(masked_output, masked_target)
        n_total = target_masks[t].sum().item()

        loss += masked_loss
        n_totals += n_total
        hiddens.append(hidden)
    loss /= n_totals
    loss = loss.to(device)

    hiddens = torch.stack(hiddens)
    output_indices = torch.LongTensor(output_indices)
    return loss, hiddens, output_indices


def forward_global_reconstructor(decoder_hiddens, targets, reconstructor, loss_func):
    decoder_hiddens = decoder_hiddens.to(device)
    targets = targets.to(device)

    hidden = (
        torch.zeros(C.reconstructor_n_layers, C.batch_size, C.reconstructor_hidden_size).to(device),
        torch.zeros(C.reconstructor_n_layers, C.batch_size, C.reconstructor_hidden_size).to(device),
    )

    # decoder_hiddens: (MAX_LEN, C.decoder_n_layers, C.batch_size, C.decoder_hidden_size)
    outputs = []
    decoder_max_n_words = decoder_hiddens.size()[0]
    for t in range(decoder_max_n_words):
        output, hidden = reconstructor(decoder_hiddens[t], hidden, decoder_hiddens)
        outputs.append(output)
    outputs = torch.stack(outputs)
    outputs = outputs.mean(0)

    targets = targets.mean(1)
    loss = loss_func(outputs, targets)

    loss /= decoder_max_n_words

    return loss


def convert_idxs_to_sentences(idxs, idx2word):
    sentences = []
    idxs_list = np.asarray(idxs).T
    for idxs in idxs_list:
        sentences.append([])
        for idx in idxs:
            if idx == 0: break
            sentences[-1].append(idx2word[idx])
        sentences[-1] = " ".join(sentences[-1])
    return sentences


def sample_n(lst, n):
    indices = list(range(len(lst)))
    sample_indices = np.random.choice(indices, n, replace=False)
    sample_lst = [ lst[i] for i in sample_indices ]
    return sample_lst


def dec_rec_step(batch, decoder, decoder_loss_func, decoder_optimizer, reconstructor, reconstructor_loss_func,
                 reconstructor_optimizer, is_train):
    encoder_outputs, targets, target_masks, target_max_n_words = batch

    if is_train:
        decoder.train()
        decoder_optimizer.zero_grad()
    else:
        decoder.eval()
    decoder_loss, decoder_hiddens, decoder_output_indices = forward_decoder(
        encoder_outputs, targets, target_masks, target_max_n_words, decoder, decoder_loss_func, is_train=is_train)

    if is_train:
        reconstructor.train()
        reconstructor_optimizer.zero_grad()
    else:
        reconstructor.eval()
    if C.reconstructor_type == "global":
        recon_loss = forward_global_reconstructor(
            decoder_hiddens, encoder_outputs, reconstructor, reconstructor_loss_func)
    else:
        raise NotImplementedError("Unknown reconstructor type '{}'".format(C.reconstructor_type))

    loss = decoder_loss + C.loss_lambda * recon_loss

    # Perform backpropatation
    if is_train:
        loss.backward()
        # _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), C.clip)
        decoder_optimizer.step()
        reconstructor_optimizer.step()

    loss = loss.item()
    decoder_loss = decoder_loss.item()
    recon_loss = recon_loss.item()
    return loss, decoder_loss, decoder_output_indices, recon_loss


def dec_step(batch, decoder, decoder_loss_func, decoder_optimizer, is_train):
    encoder_outputs, targets, target_masks, target_max_n_words = batch

    if is_train:
        decoder.train()
        decoder_optimizer.zero_grad()
    else:
        decoder.eval()

    decoder_loss, decoder_hiddens, decoder_output_indices = forward_decoder(
        encoder_outputs, targets, target_masks, target_max_n_words, decoder, decoder_loss_func, is_train=is_train)

    # Perform backpropatation
    if is_train:
        decoder_loss.backward()
        # _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), C.clip)
        decoder_optimizer.step()

    decoder_loss = decoder_loss.item()
    return decoder_loss, decoder_output_indices


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--debug", "-D", action="store_true")
    args = a.parse_args()

    print("MODEL ID: {}".format(C.model_id))
    print("DEBUG MODE: {}".format(['OFF', 'ON'][args.debug]))

    if not args.debug:
        writer = SummaryWriter(C.log_dpath)

    dataset = MSVD_dataset(C)
    vocab = dataset.vocab
    train_data_loader = cycle(iter(dataset.train_data_loader))
    val_data_loader = cycle(iter(dataset.val_data_loader))
    print('n_vocabs: {} ({}), n_words: {} ({}). (): before trim (min_count: {})'.format(
        vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, C.min_count))

    decoder = Decoder(
        n_layers=C.decoder_n_layers,
        encoder_size=C.encoder_output_size,
        embedding_size=C.word_embedding_size,
        hidden_size=C.decoder_hidden_size,
        output_size=vocab.n_vocabs,
        embedding_dropout=C.embedding_dropout,
        dropout=C.decoder_dropout,
        max_length=C.encoder_output_len,
    )
    decoder = decoder.to(device)
    decoder_loss_func = nn.NLLLoss()
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=C.learning_rate * C.decoder_learning_ratio)


    if C.use_recon:
        if C.reconstructor_type == "global":
            reconstructor = GlobalReconstructor(
                n_layers=C.reconstructor_n_layers,
                decoder_hidden_size=C.decoder_hidden_size,
                hidden_size=C.reconstructor_hidden_size,
                dropout=C.reconstructor_dropout,
            )
        reconstructor = reconstructor.to(device)
        reconstructor_loss_func = nn.MSELoss()
        reconstructor_optimizer = optim.Adam(reconstructor.parameters(), lr=C.learning_rate * C.reconstructor_learning_ratio)
    
    train_loss = 0
    if C.use_recon:
        train_dec_loss = 0
        train_rec_loss = 0
    for iteration, batch in enumerate(train_data_loader, 1):
        # Train
        if C.use_recon:
            loss, decoder_loss, _, recon_loss = dec_rec_step(
                batch, decoder, decoder_loss_func, decoder_optimizer, reconstructor, reconstructor_loss_func,
                reconstructor_optimizer, is_train=True)
            train_dec_loss += decoder_loss
            train_rec_loss += recon_loss
        else:
            loss, _ = dec_step(batch, decoder, decoder_loss_func, decoder_optimizer, is_train=True)
        train_loss += loss


        # Print progress
        if args.debug or iteration % C.log_every == 0:
            train_loss /= C.log_every
            if C.use_recon:
                train_dec_loss /= C.log_every
                train_rec_loss /= C.log_every

            if not args.debug:
                writer.add_scalar(C.tx_train_loss, train_loss, iteration)
                if C.use_recon:
                    writer.add_scalar(C.tx_train_loss_decoder, train_dec_loss, iteration)
                    writer.add_scalar(C.tx_train_loss_reconstructor, train_rec_loss, iteration)

            if C.use_recon:
                print("Iter {} / {} ({:.1f}%): loss {:.5f} (dec {:.5f} + rec {:.5f})".format(
                    iteration, C.train_n_iteration, iteration / C.train_n_iteration * 100, train_loss,
                    train_dec_loss, train_rec_loss))
            else:
                print("Iter {} / {} ({:.1f}%): loss {:.5f}".format(
                    iteration, C.train_n_iteration, iteration / C.train_n_iteration * 100, train_loss))

            train_loss = 0
            if C.use_recon:
                train_dec_loss = 0
                train_rec_loss = 0


        # Validate model
        if args.debug or iteration % C.validate_every == 0:
            val_loss = 0
            val_dec_loss = 0
            val_rec_loss = 0
            gt_captions = []
            pd_captions = []
            for i, batch in enumerate(val_data_loader, 1):
                # Validate
                if C.use_recon:
                    loss, decoder_loss, decoder_output_indices, recon_loss = dec_rec_step(
                        batch, decoder, decoder_loss_func, decoder_optimizer, reconstructor,
                        reconstructor_loss_func, reconstructor_optimizer, is_train=False)
                    val_dec_loss += decoder_loss
                    val_rec_loss += recon_loss
                else:
                    loss, decoder_output_indices = dec_step(batch, decoder, decoder_loss_func, decoder_optimizer,
                        is_train=False)
                val_loss += loss

                _, targets, _, _ = batch
                gt_idxs = targets.numpy()
                pd_idxs = decoder_output_indices.numpy()
                gt_captions += convert_idxs_to_sentences(gt_idxs, vocab.idx2word)
                pd_captions += convert_idxs_to_sentences(pd_idxs, vocab.idx2word)

                if i == C.val_n_iteration:
                    break
            val_loss /= C.val_n_iteration
            val_dec_loss /= C.val_n_iteration
            val_rec_loss /= C.val_n_iteration

            if C.use_recon:
                print("[Validation] Iter {} / {} ({:.1f}%): loss {:.5f} (dec {:.5f} + rec {:5f})".format(
                    iteration, C.train_n_iteration, iteration / C.train_n_iteration * 100, val_loss,
                    val_dec_loss, val_rec_loss))
            else:
                print("[Validation] Iter {} / {} ({:.1f}%): loss {:.5f}".format(
                    iteration, C.train_n_iteration, iteration / C.train_n_iteration * 100, val_loss))

            caption_pairs = [ (gt, pred) for gt, pred in zip(gt_captions, pd_captions) ]
            caption_pairs = sample_n(caption_pairs, C.n_val_logs)
            caption_log = "\n\n".join([ "[GT] {}  \n[PD] {}".format(gt, pd) for gt, pd in caption_pairs ])

            if not args.debug:
                writer.add_scalar(C.tx_val_loss, val_loss, iteration)
                if C.use_recon:
                    writer.add_scalar(C.tx_val_loss_decoder, val_dec_loss, iteration)
                    writer.add_scalar(C.tx_val_loss_reconstructor, val_rec_loss, iteration)
                writer.add_text(C.tx_predicted_captions, caption_log, iteration)

        # Save checkpoint
        if iteration % C.save_every == 0:
            if not os.path.exists(C.save_dpath):
                os.makedirs(C.save_dpath)
            fpath = os.path.join(C.save_dpath, "{}_checkpoint.tar".format(iteration))

            if C.use_recon:
                torch.save({
                    'iteration': iteration,
                    'dec': decoder.state_dict(),
                    'rec': reconstructor.state_dict(),
                    'dec_opt': decoder_optimizer.state_dict(),
                    'rec_opt': reconstructor_optimizer.state_dict(),
                    'loss': loss,
                }, fpath)
            else:
                torch.save({
                    'iteration': iteration,
                    'dec': decoder.state_dict(),
                    'dec_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                }, fpath)

        if iteration == C.train_n_iteration:
            break


if __name__ == "__main__":
    main()
