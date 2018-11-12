from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from collections import defaultdict
import os
import random

from nlgeval import NLGEval
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch import optim

from config import TrainConfig as C
from dataset.MSVD import MSVD as _MSVD
# from models.decoder import Decoder
from models.attn_decoder import AttnDecoder as Decoder
from models.local_reconstructor import LocalReconstructor
from models.global_reconstructor import GlobalReconstructor
from utils import cycle, convert_idxs_to_sentences, sample_n


def forward_decoder(encoder_outputs, targets, target_masks, decoder, loss_func, is_train):
    # Initialize variables
    loss = 0
    n_totals = 0
    hiddens = []
    output_indices = []

    # Create initial decoder input (start with SOS tokens for each sentence)
    input = torch.LongTensor([ [C.init_word2idx['<SOS>'] for _ in range(C.batch_size)] ])
    input = input.to(C.device)

    if C.decoder_model == "LSTM":
        hidden = (
            torch.zeros(C.decoder_n_layers, C.batch_size, C.decoder_hidden_size).to(C.device),
            torch.zeros(C.decoder_n_layers, C.batch_size, C.decoder_hidden_size).to(C.device),
        )
    else:
        hidden = torch.zeros(C.decoder_n_layers, C.batch_size, C.decoder_hidden_size)
        hidden = hidden.to(C.device)

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = is_train and random.random() <= C.decoder_teacher_forcing_ratio

    # Forward batch of sequences through decoder one time step at a time
    for t in range(C.caption_n_max_word + 1):
        output, hidden = decoder(input, hidden, encoder_outputs)

        if use_teacher_forcing:
            input = targets[t].view(1, -1)
        else:
            _, topi = output.topk(1)
            output_index = [ topi[i][0] for i in range(C.batch_size) ]
            input = torch.LongTensor([ output_index ])
            input = input.to(C.device)
            output_indices.append(output_index)

        # Calculate and accumulate loss
        masked_output = output[target_masks[t]]
        masked_target = targets[t][target_masks[t]]
        masked_loss = loss_func(masked_output, masked_target)
        n_total = target_masks[t].sum()

        loss += masked_loss
        n_totals += n_total
        if C.decoder_model == "LSTM":
            hiddens.append(hidden[0])
        else:
            hiddens.append(hidden)

        if t == C.caption_n_max_word:
            break
    loss /= n_totals
    loss = loss.to(C.device)

    hiddens = torch.stack(hiddens)
    output_indices = torch.LongTensor(output_indices)
    return loss, hiddens, output_indices


def forward_global_reconstructor(decoder_hiddens, targets, reconstructor, loss_func):
    decoder_hiddens = decoder_hiddens.to(C.device)
    targets = targets.to(C.device)

    if C.reconstructor_model == "LSTM":
        hidden = (
            torch.zeros(C.reconstructor_n_layers, C.batch_size, C.reconstructor_hidden_size).to(C.device),
            torch.zeros(C.reconstructor_n_layers, C.batch_size, C.reconstructor_hidden_size).to(C.device),
        )
    else:
        hidden = torch.zeros(C.reconstructor_n_layers, C.batch_size, C.reconstructor_hidden_size)
        hidden = hidden.to(C.device)

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


def dec_rec_step(batch, decoder, decoder_loss_func, decoder_lambda, decoder_optimizer, reconstructor,
                 reconstructor_loss_func, reconstructor_lambda, reconstructor_optimizer, loss_lambda, is_train):
    _, encoder_outputs, targets = batch
    encoder_outputs = encoder_outputs.to(C.device)
    targets = targets.to(C.device)
    targets = targets.long()
    target_masks = targets > C.init_word2idx['<PAD>']

    """ Decoder """
    if is_train:
        decoder.train()
        decoder_optimizer.zero_grad()
    else:
        decoder.eval()
    decoder_loss, decoder_hiddens, decoder_output_indices = forward_decoder(
        encoder_outputs, targets, target_masks, decoder, decoder_loss_func, is_train=is_train)

    """ Reconstructor """
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

    """ Loss """
    decoder_reg_loss = sum([ torch.norm(param) for param in decoder.parameters() ])
    decoder_loss = decoder_loss + decoder_lambda * decoder_reg_loss
    recon_reg_loss = sum([ torch.norm(param) for param in reconstructor.parameters() ])
    recon_loss = recon_loss + reconstructor_lambda * recon_reg_loss
    loss = decoder_loss + loss_lambda * recon_loss

    # Perform backpropatation
    if is_train:
        loss.backward()
        if C.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), C.clip)
        decoder_optimizer.step()
        reconstructor_optimizer.step()

    loss = loss.item()
    decoder_loss = decoder_loss.item()
    recon_loss = recon_loss.item()
    return loss, decoder_loss, decoder_output_indices, recon_loss


def dec_step(batch, decoder, decoder_loss_func, decoder_lambda, decoder_optimizer, is_train):
    _, encoder_outputs, targets = batch
    encoder_outputs = encoder_outputs.to(C.device)
    targets = targets.to(C.device)
    targets = targets.long()
    target_masks = targets > C.init_word2idx['<PAD>']

    if is_train:
        decoder.train()
        decoder_optimizer.zero_grad()
    else:
        decoder.eval()

    decoder_loss, decoder_hiddens, decoder_output_indices = forward_decoder(
        encoder_outputs, targets, target_masks, decoder, decoder_loss_func, is_train=is_train)
    decoder_reg_loss = sum([ torch.norm(param) for param in decoder.parameters() ])
    decoder_loss = decoder_loss + decoder_lambda * decoder_reg_loss

    # Perform backpropatation
    if is_train:
        decoder_loss.backward()
        if C.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), C.clip)
        decoder_optimizer.step()

    decoder_loss = decoder_loss.item()
    return decoder_loss, decoder_output_indices


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--debug", "-D", action="store_true")
    args = a.parse_args()

    print("MODEL ID: {}".format(C.id))
    print("DEBUG MODE: {}".format(['OFF', 'ON'][args.debug]))

    if not args.debug:
        train_writer = SummaryWriter(C.train_log_dpath)
        val_writer = SummaryWriter(C.val_log_dpath)
        Bleu1_writer = SummaryWriter(C.Bleu1_log_dpath)
        Bleu2_writer = SummaryWriter(C.Bleu2_log_dpath)
        Bleu3_writer = SummaryWriter(C.Bleu3_log_dpath)
        Bleu4_writer = SummaryWriter(C.Bleu4_log_dpath)
        CIDEr_writer = SummaryWriter(C.CIDEr_log_dpath)
        METEOR_writer = SummaryWriter(C.METEOR_log_dpath)
        ROUGE_L_writer = SummaryWriter(C.ROUGE_L_log_dpath)


    """ Load DataLoader """
    MSVD = _MSVD(C)
    vocab = MSVD.vocab
    train_data_loader = iter(cycle(MSVD.train_data_loader))
    val_data_loader = iter(cycle(MSVD.val_data_loader))
    test_data_loader = iter(cycle(MSVD.test_data_loader))

    print('n_vocabs: {} ({}), n_words: {} ({}). MIN_COUNT: {}'.format(
        vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, C.min_count))


    """ Build Decoder """
    decoder = Decoder(
        model_name=C.decoder_model,
        n_layers=C.decoder_n_layers,
        encoder_size=C.encoder_output_size,
        embedding_size=C.embedding_size,
        embedding_scale=C.embedding_scale,
        hidden_size=C.decoder_hidden_size,
        attn_size=C.decoder_attn_size,
        output_size=vocab.n_vocabs,
        embedding_dropout=C.embedding_dropout,
        dropout=C.decoder_dropout,
        out_dropout=C.decoder_out_dropout,
    )
    decoder = decoder.to(C.device)
    decoder_loss_func = nn.CrossEntropyLoss()
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=C.decoder_learning_rate,
                                   weight_decay=C.decoder_weight_decay, amsgrad=C.decoder_use_amsgrad)
    decoder_lambda = torch.autograd.Variable(torch.tensor(0.001), requires_grad=True)
    decoder_lambda = decoder_lambda.to(C.device)


    """ Build Reconstructor """
    if C.use_recon:
        if C.reconstructor_type == "local":
            reconstructor = LocalReconstructor(
                n_layers=C.reconstructor_n_layers,
                hidden_size=C.reconstructor_hidden_size,
                dropout=C.reconstructor_dropout,
            )
        elif C.reconstructor_type == "global":
            reconstructor = GlobalReconstructor(
                model_name=C.reconstructor_model,
                n_layers=C.reconstructor_n_layers,
                decoder_hidden_size=C.decoder_hidden_size,
                hidden_size=C.reconstructor_hidden_size,
                dropout=C.reconstructor_dropout,
            )
        reconstructor = reconstructor.to(C.device)
        reconstructor_loss_func = nn.MSELoss()
        reconstructor_optimizer = optim.Adam(reconstructor.parameters(), lr=C.reconstructor_learning_rate,
                                             weight_decay=C.reconstructor_weight_decay,
                                             amsgrad=C.reconstructor_use_amsgrad)
        reconstructor_lambda = torch.autograd.Variable(torch.tensor(0.01), requires_grad=True)
        reconstructor_lambda = reconstructor_lambda.to(C.device)
        loss_lambda = torch.autograd.Variable(torch.tensor(1.), requires_grad=True)
        loss_lambda = loss_lambda.to(C.device)


    """ Build Qualitative Metrics """
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)
    reference_captions = defaultdict(lambda: [])
    for vid, _, caption in MSVD.test_dataset.video_caption_pairs:
        reference_captions[vid].append(caption)


    """ Train """
    train_loss = 0
    if C.use_recon:
        train_dec_loss = 0
        train_rec_loss = 0
    for iteration, batch in enumerate(train_data_loader, 1):
        if C.use_recon:
            loss, decoder_loss, _, recon_loss = dec_rec_step(
                batch, decoder, decoder_loss_func, decoder_lambda, decoder_optimizer, reconstructor,
                reconstructor_loss_func, reconstructor_lambda, reconstructor_optimizer, loss_lambda, is_train=True)
            train_dec_loss += decoder_loss
            train_rec_loss += recon_loss
        else:
            loss, _ = dec_step(batch, decoder, decoder_loss_func, decoder_lambda, decoder_optimizer, is_train=True)
        train_loss += loss


        """ Log Train Progress """
        if args.debug or iteration % C.log_every == 0:
            train_loss /= C.log_every
            if C.use_recon:
                train_dec_loss /= C.log_every
                train_rec_loss /= C.log_every

            if not args.debug:
                train_writer.add_scalar(C.tx_loss, train_loss, iteration)
                train_writer.add_scalar(C.tx_lambda_decoder, decoder_lambda.item(), iteration)
                if C.use_recon:
                    train_writer.add_scalar(C.tx_loss_decoder, train_dec_loss, iteration)
                    train_writer.add_scalar(C.tx_loss_reconstructor, train_rec_loss, iteration)
                    train_writer.add_scalar(C.tx_lambda_reconstructor, reconstructor_lambda.item(), iteration)
                    train_writer.add_scalar(C.tx_lambda, loss_lambda.item(), iteration)

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


        """ Log Validation Progress """
        if args.debug or iteration % C.validate_every == 0:
            val_loss = 0
            val_dec_loss = 0
            val_rec_loss = 0
            gt_captions = []
            pd_captions = []
            for batch in val_data_loader:
                if C.use_recon:
                    loss, decoder_loss, decoder_output_indices, recon_loss = dec_rec_step(
                        batch, decoder, decoder_loss_func, decoder_lambda, decoder_optimizer, reconstructor,
                        reconstructor_loss_func, reconstructor_lambda, reconstructor_optimizer, loss_lambda,
                        is_train=False)
                    val_dec_loss += decoder_loss * C.batch_size
                    val_rec_loss += recon_loss * C.batch_size
                else:
                    loss, decoder_output_indices = dec_step(batch, decoder, decoder_loss_func, decoder_lambda,
                                                            decoder_optimizer, is_train=False)
                val_loss += loss * C.batch_size

                _, _, targets = batch
                gt_idxs = targets.cpu().numpy()
                pd_idxs = decoder_output_indices.cpu().numpy()
                gt_captions += convert_idxs_to_sentences(gt_idxs, vocab.idx2word, vocab.word2idx['<EOS>'])
                pd_captions += convert_idxs_to_sentences(pd_idxs, vocab.idx2word, vocab.word2idx['<EOS>'])

                if len(pd_captions) >= C.n_val:
                    assert len(gt_captions) == len(pd_captions)
                    gt_captions = gt_captions[:C.n_val]
                    pd_captions = pd_captions[:C.n_val]
                    break
            val_loss /= C.n_val
            val_dec_loss /= C.n_val
            val_rec_loss /= C.n_val

            if C.use_recon:
                print("[Validation] Iter {} / {} ({:.1f}%): loss {:.5f} (dec {:.5f} + rec {:5f})".format(
                    iteration, C.train_n_iteration, iteration / C.train_n_iteration * 100, val_loss,
                    val_dec_loss, val_rec_loss))
            else:
                print("[Validation] Iter {} / {} ({:.1f}%): loss {:.5f}".format(
                    iteration, C.train_n_iteration, iteration / C.train_n_iteration * 100, val_loss))

            caption_pairs = [ (gt, pred) for gt, pred in zip(gt_captions, pd_captions) ]
            caption_pairs = sample_n(caption_pairs, min(C.n_val_logs, C.batch_size))
            caption_log = "\n\n".join([ "[GT] {}  \n[PD] {}".format(gt, pd) for gt, pd in caption_pairs ])

            if not args.debug:
                val_writer.add_scalar(C.tx_loss, val_loss, iteration)
                if C.use_recon:
                    val_writer.add_scalar(C.tx_loss_decoder, val_dec_loss, iteration)
                    val_writer.add_scalar(C.tx_loss_reconstructor, val_rec_loss, iteration)
                val_writer.add_text(C.tx_predicted_captions, caption_log, iteration)


        """ Log Test Progress """
        if args.debug or iteration % C.test_every == 0:
            pd_vid_caption_pairs = []
            for batch in test_data_loader:
                if C.use_recon:
                    _, _, decoder_output_indices, _ = dec_rec_step(
                        batch, decoder, decoder_loss_func, decoder_lambda, decoder_optimizer, reconstructor,
                        reconstructor_loss_func, reconstructor_lambda, reconstructor_optimizer, loss_lambda,
                        is_train=False)
                else:
                    _, decoder_output_indices = dec_step(batch, decoder, decoder_loss_func, decoder_lambda,
                                                            decoder_optimizer, is_train=False)

                vids, _, _ = batch
                pd_idxs = decoder_output_indices.cpu().numpy()
                pd_captions = convert_idxs_to_sentences(pd_idxs, vocab.idx2word, vocab.word2idx['<EOS>'])
                pd_vid_caption_pairs += [ ( vid, caption ) for vid, caption in zip(vids, pd_captions) ]

                if len(pd_vid_caption_pairs) >= C.n_test:
                    pd_vid_caption_pairs = pd_vid_caption_pairs[:C.n_test]
                    break
            Bleu1, Bleu2, Bleu3, Bleu4, CIDEr, METEOR, ROUGE_L = 0, 0, 0, 0, 0, 0, 0
            for vid, hypothesis_caption in pd_vid_caption_pairs:
                score = nlgeval.compute_individual_metrics(reference_captions[vid], hypothesis_caption)
                Bleu1 += score['Bleu_1']
                Bleu2 += score['Bleu_2']
                Bleu3 += score['Bleu_3']
                Bleu4 += score['Bleu_4']
                CIDEr += score['CIDEr']
                METEOR += score['METEOR']
                ROUGE_L += score['ROUGE_L']
            Bleu1 /= C.n_test
            Bleu2 /= C.n_test
            Bleu3 /= C.n_test
            Bleu4 /= C.n_test
            CIDEr /= C.n_test
            METEOR /= C.n_test
            ROUGE_L /= C.n_test
            print("[Test] Iter {} / {} ({:.1f}%): B1: {:.1f}, B2: {:.1f}, B3: {:.1f}, B4: {:.1f}, C: {:.1f}, M: {:.1f}, R: {:.1f}".format(
                iteration, C.train_n_iteration, iteration / C.train_n_iteration * 100, Bleu1, Bleu2, Bleu3, Bleu4,
                CIDEr, METEOR, ROUGE_L))
            if not args.debug:
                Bleu1_writer.add_scalar(C.tx_score_Bleu1, Bleu1, iteration)
                Bleu2_writer.add_scalar(C.tx_score_Bleu2, Bleu2, iteration)
                Bleu3_writer.add_scalar(C.tx_score_Bleu3, Bleu3, iteration)
                Bleu4_writer.add_scalar(C.tx_score_Bleu4, Bleu4, iteration)
                CIDEr_writer.add_scalar(C.tx_score_CIDEr, CIDEr, iteration)
                METEOR_writer.add_scalar(C.tx_score_METEOR, METEOR, iteration)
                ROUGE_L_writer.add_scalar(C.tx_score_ROUGE_L, ROUGE_L, iteration)


        """ Save checkpoint """
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
                    'config': C,
                }, fpath)
            else:
                torch.save({
                    'iteration': iteration,
                    'dec': decoder.state_dict(),
                    'dec_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'config': C,
                }, fpath)

        if iteration == C.train_n_iteration:
            break


if __name__ == "__main__":
    main()
