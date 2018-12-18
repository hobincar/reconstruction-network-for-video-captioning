import argparse
import os
import random

from tensorboardX import SummaryWriter
import torch

from config import TrainConfig as C
from dataset.MSVD import MSVD
from eval import evaluate
from models.decoder import Decoder
from models.local_reconstructor import LocalReconstructor
from models.global_reconstructor import GlobalReconstructor
from utils import cycle, convert_idxs_to_sentences


def forward_decoder(decoder, encoder_outputs, targets, target_masks, teacher_forcing_ratio=0.):
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
    use_teacher_forcing = random.random() <= teacher_forcing_ratio

    # Forward batch of sequences through decoder one time step at a time
    for t in range(C.caption_max_len + 1):
        output, hidden = decoder['model'](input, hidden, encoder_outputs)

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
        masked_loss = decoder['loss'](masked_output, masked_target)
        n_total = target_masks[t].sum()

        loss += masked_loss
        n_totals += n_total
        if C.decoder_model == "LSTM":
            hiddens.append(hidden[0])
        else:
            hiddens.append(hidden)

        if t == C.caption_max_len or torch.all(target_masks[t+1] == 0):
            break
    loss /= n_totals
    reg_loss = sum([ torch.norm(param) for param in decoder['model'].parameters() ])
    loss = loss + decoder['lambda_reg'] * reg_loss
    loss = loss.to(C.device)

    hiddens = torch.stack(hiddens)
    output_indices = torch.LongTensor(output_indices)
    return loss, hiddens, output_indices


def forward_global_reconstructor(decoder_hiddens, encoder_outputs, reconstructor):
    decoder_hiddens = decoder_hiddens.to(C.device)
    encoder_outputs = encoder_outputs.to(C.device)

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
        output, hidden = reconstructor['model'](decoder_hiddens[t], hidden, decoder_hiddens)
        outputs.append(output)
    outputs = torch.stack(outputs)

    outputs = outputs.mean(0)
    encoder_outputs = encoder_outputs.mean(1)

    loss = reconstructor['loss'](outputs, encoder_outputs)
    loss /= decoder_max_n_words
    reg_loss = sum([ torch.norm(param) for param in reconstructor['model'].parameters() ])
    loss = loss + reconstructor['lambda_reg'] * reg_loss
    return loss


def forward_local_reconstructor(decoder_hiddens, encoder_outputs, reconstructor):
    decoder_hiddens = decoder_hiddens.to(C.device)
    encoder_outputs = encoder_outputs.to(C.device)

    if C.reconstructor_model == "LSTM":
        hidden = (
            torch.zeros(C.reconstructor_n_layers, C.batch_size, C.reconstructor_hidden_size).to(C.device),
            torch.zeros(C.reconstructor_n_layers, C.batch_size, C.reconstructor_hidden_size).to(C.device),
        )
    else:
        hidden = torch.zeros(C.reconstructor_n_layers, C.batch_size, C.reconstructor_hidden_size)
        hidden = hidden.to(C.device)

    outputs = []
    for t in range(C.encoder_output_len):
        output, hidden = reconstructor['model'](hidden, decoder_hiddens)
        outputs.append(output)
    outputs = torch.stack(outputs)

    outputs = outputs.transpose(0, 1)
    loss = reconstructor['loss'](outputs, encoder_outputs)
    reg_loss = sum([ torch.norm(param) for param in reconstructor['model'].parameters() ])
    loss = loss + reconstructor['lambda_reg'] * reg_loss
    return loss


def build_decoder(n_vocabs):
    model = Decoder(
        model_name=C.decoder_model,
        n_layers=C.decoder_n_layers,
        encoder_size=C.encoder_output_size,
        embedding_size=C.embedding_size,
        embedding_scale=C.embedding_scale,
        hidden_size=C.decoder_hidden_size,
        attn_size=C.decoder_attn_size,
        output_size=n_vocabs,
        embedding_dropout=C.embedding_dropout,
        dropout=C.decoder_dropout,
        out_dropout=C.decoder_out_dropout)
    model = model.to(C.device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=C.decoder_learning_rate, weight_decay=C.decoder_weight_decay,
                           amsgrad=C.decoder_use_amsgrad)
    lambda_reg = torch.autograd.Variable(torch.tensor(0.001), requires_grad=True)
    lambda_reg = lambda_reg.to(C.device)

    decoder = {
        'model': model,
        'loss': loss,
        'optimizer': optimizer,
        'lambda_reg': lambda_reg,
    }
    return decoder


def build_reconstructor():
    if C.reconstructor_type == "local":
        model = LocalReconstructor(
            model_name=C.reconstructor_model,
            n_layers=C.reconstructor_n_layers,
            decoder_hidden_size=C.decoder_hidden_size,
            hidden_size=C.reconstructor_hidden_size,
            dropout=C.reconstructor_dropout,
            decoder_dropout=C.reconstructor_decoder_dropout,
            attn_size=C.reconstructor_attn_size)
    elif C.reconstructor_type == "global":
        model = GlobalReconstructor(
            model_name=C.reconstructor_model,
            n_layers=C.reconstructor_n_layers,
            decoder_hidden_size=C.decoder_hidden_size,
            hidden_size=C.reconstructor_hidden_size,
            dropout=C.reconstructor_dropout,
            decoder_dropout=C.reconstructor_decoder_dropout,
            caption_max_len=C.caption_max_len)
    else:
        raise NotImplementedError("Unknown reconstructor: {}".format(C.reconstructor_type))
    model = model.to(C.device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=C.reconstructor_learning_rate,
                           weight_decay=C.reconstructor_weight_decay, amsgrad=C.reconstructor_use_amsgrad)
    lambda_reg = torch.autograd.Variable(torch.tensor(0.01), requires_grad=True)
    lambda_reg = lambda_reg.to(C.device)

    reconstructor = {
        'model': model,
        'loss': loss,
        'optimizer': optimizer,
        'lambda_reg': lambda_reg,
    }
    return reconstructor


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--debug", "-D", action="store_true")
    a.add_argument("--loss_only", "-L", action="store_true")
    args = a.parse_args()

    print("MODEL ID: {}".format(C.id))
    print("DEBUG MODE: {}".format(['OFF', 'ON'][args.debug]))

    if not args.debug:
        summary_writer = SummaryWriter(C.log_dpath)


    """ Load DataLoader """
    dataset = MSVD(C)
    vocab = dataset.vocab
    train_data_loader = iter(cycle(dataset.train_data_loader))
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, C.min_count))


    """ Build Models """
    decoder = build_decoder(vocab.n_vocabs)
    if C.use_recon:
        reconstructor = build_reconstructor()
        lambda_recon = torch.autograd.Variable(torch.tensor(1.), requires_grad=True)
        lambda_recon = lambda_recon.to(C.device)


    """ Train """
    train_loss = 0
    if C.use_recon:
        train_dec_loss = 0
        train_rec_loss = 0

        if C.reconstructor_type == "global":
            forward_reconstructor = forward_global_reconstructor
        elif C.reconstructor_type == "local":
            forward_reconstructor = forward_local_reconstructor
        else:
            raise NotImplementedError("Unknown reconstructor type '{}'".format(C.reconstructor_type))
    for iteration, batch in enumerate(train_data_loader, 1):
        _, encoder_outputs, targets = batch
        encoder_outputs = encoder_outputs.to(C.device)
        targets = targets.to(C.device)
        targets = targets.long()
        target_masks = targets > C.init_word2idx['<PAD>']

        # Decoder
        decoder['model'].train()
        decoder_loss, decoder_hiddens, decoder_output_indices = forward_decoder(
            decoder, encoder_outputs, targets, target_masks, C.decoder_teacher_forcing_ratio)

        # Reconstructor
        if C.use_recon:
            reconstructor['model'].train()
            recon_loss = forward_reconstructor(decoder_hiddens, encoder_outputs, reconstructor)

        # Loss
        if C.use_recon:
            loss = decoder_loss + lambda_recon * recon_loss
        else:
            loss = decoder_loss

        # Backprop
        decoder['optimizer'].zero_grad()
        reconstructor['optimizer'].zero_grad()
        loss.backward()
        if C.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(decoder['model'].parameters(), C.gradient_clip)
        decoder['optimizer'].step()
        if C.use_recon:
            reconstructor['optimizer'].step()

            train_dec_loss += decoder_loss.item()
            train_rec_loss += recon_loss.item()
        train_loss += loss.item()


        """ Log Train Progress """
        if args.debug or iteration % C.log_every == 0:
            train_loss /= C.log_every
            if C.use_recon:
                train_dec_loss /= C.log_every
                train_rec_loss /= C.log_every

            if not args.debug:
                summary_writer.add_scalar(C.tx_train_loss, train_loss, iteration)
                summary_writer.add_scalar(C.tx_lambda_decoder, decoder['lambda_reg'].item(), iteration)
                if C.use_recon:
                    summary_writer.add_scalar(C.tx_train_loss_decoder, train_dec_loss, iteration)
                    summary_writer.add_scalar(C.tx_train_loss_reconstructor, train_rec_loss, iteration)
                    summary_writer.add_scalar(C.tx_lambda_reconstructor, reconstructor['lambda_reg'].item(), iteration)
                    summary_writer.add_scalar(C.tx_lambda, lambda_recon.item(), iteration)

            msg = "Iter {} / {} ({:.1f}%): loss {:.5f}".format(iteration, C.n_iterations,
                                                               iteration / C.n_iterations * 100, train_loss)
            if C.use_recon:
                msg += " (dec {:.5f} + rec {:.5f})".format(train_dec_loss, train_rec_loss)
            print(msg)

            train_loss = 0
            if C.use_recon:
                train_dec_loss = 0
                train_rec_loss = 0


        """ Log Validation Progress """
        if args.debug or iteration % C.validate_every == 0:
            val_loss = 0
            if C.use_recon:
                val_dec_loss = 0
                val_rec_loss = 0
            gt_captions = []
            pd_captions = []
            val_data_loader = iter(dataset.val_data_loader)
            for batch in val_data_loader:
                _, encoder_outputs, targets = batch
                encoder_outputs = encoder_outputs.to(C.device)
                targets = targets.to(C.device)
                targets = targets.long()
                target_masks = targets > C.init_word2idx['<PAD>']

                # Decoder
                decoder['model'].eval()
                decoder_loss, decoder_hiddens, decoder_output_indices = forward_decoder(
                    decoder, encoder_outputs, targets, target_masks)
   
                # Reconstructor
                if C.use_recon:
                    reconstructor['model'].eval()
                    recon_loss = forward_reconstructor(decoder_hiddens, encoder_outputs, reconstructor)

                # Loss
                if C.use_recon:
                    loss = decoder_loss + lambda_recon * recon_loss
                else:
                    loss = decoder_loss


                if C.use_recon:
                    val_dec_loss += decoder_loss.item() * C.batch_size
                    val_rec_loss += recon_loss.item() * C.batch_size
                val_loss += loss.item() * C.batch_size

                _, _, targets = batch
                gt_idxs = targets.cpu().numpy()
                pd_idxs = decoder_output_indices.cpu().numpy()
                gt_captions += convert_idxs_to_sentences(gt_idxs, vocab.idx2word, vocab.word2idx['<EOS>'])
                pd_captions += convert_idxs_to_sentences(pd_idxs, vocab.idx2word, vocab.word2idx['<EOS>'])

            n_vals = len(val_data_loader)
            val_loss /= n_vals
            if C.use_recon:
                val_dec_loss /= n_vals
                val_rec_loss /= n_vals

            msg = "[Validation] Iter {} / {} ({:.1f}%): loss {:.5f}".format(
                iteration, C.n_iterations, iteration / C.n_iterations * 100, val_loss)
            if C.use_recon:
                msg += " (dec {:.5f} + rec {:5f})".format(val_dec_loss, val_rec_loss)
            print(msg)

            if not args.debug:
                summary_writer.add_scalar(C.tx_val_loss, val_loss, iteration)
                if C.use_recon:
                    summary_writer.add_scalar(C.tx_val_loss_decoder, val_dec_loss, iteration)
                    summary_writer.add_scalar(C.tx_val_loss_reconstructor, val_rec_loss, iteration)
                caption_pairs = [ (gt, pred) for gt, pred in zip(gt_captions, pd_captions) ]
                caption_log = "\n\n".join([ "[GT] {}  \n[PD] {}".format(gt, pd) for gt, pd in caption_pairs ])
                summary_writer.add_text(C.tx_predicted_captions, caption_log, iteration)


        """ Log Test Progress """
        if not args.loss_only and (args.debug or iteration % C.test_every == 0):
            pd_vid_caption_pairs = []
            score_data_loader = dataset.score_data_loader
            print("[Test] Iter {} / {} ({:.1f}%)".format(
                iteration, C.n_iterations, iteration / C.n_iterations * 100))
            for search_method in C.search_methods:
                if isinstance(search_method, str):
                    method = search_method
                    search_method_id = search_method
                if isinstance(search_method, tuple):
                    method = search_method[0]
                    search_method_id = "-".join(( str(s) for s in search_method ))
                scores = evaluate(C, dataset, score_data_loader, decoder['model'], search_method)
                score_summary = " ".join([ "{}: {:.3f}".format(score, scores[score]) for score in C.scores ])
                print("\t{}: {}".format(search_method_id, score_summary))

                if not args.debug:
                    for score in C.scores:
                        summary_writer.add_scalar(C.tx_score[search_method_id][score], scores[score], iteration)


        """ Save checkpoint """
        if iteration % C.save_every == 0:
            if not os.path.exists(C.save_dpath):
                os.makedirs(C.save_dpath)
            ckpt_fpath = os.path.join(C.save_dpath, "{}_checkpoint.tar".format(iteration))

            if C.use_recon:
                torch.save({
                    'iteration': iteration,
                    'dec': decoder['model'].state_dict(),
                    'rec': reconstructor['model'].state_dict(),
                    'dec_opt': decoder['optimizer'].state_dict(),
                    'rec_opt': reconstructor['optimizer'].state_dict(),
                    'loss': loss,
                    'config': C,
                }, ckpt_fpath)
            else:
                torch.save({
                    'iteration': iteration,
                    'dec': decoder['model'].state_dict(),
                    'dec_opt': decoder['optimizer'].state_dict(),
                    'loss': loss,
                    'config': C,
                }, ckpt_fpath)

        if iteration == C.n_iterations:
            break


if __name__ == "__main__":
    main()

