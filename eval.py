from collections import defaultdict

import torch
import numpy as np

from coco_caption.pycocotools.msvd import MSVD as COCOMSVD
from coco_caption.pycocoevalcap.eval import COCOEvalCap
from coco_caption.pycocotools.utils import load_res
from config import EvalConfig as C
from dataset.MSVD import MSVD as _MSVD
from models.decoder import Decoder
from utils import convert_idxs_to_sentences


class MockConfig:
    pass


def greedy_search(config, decoder, input, hidden, encoder_outputs):
    output_indices = []
    for t in range(config.caption_max_len + 1):
        output, hidden = decoder(input, hidden, encoder_outputs)

        _, topi = output.topk(1)
        output_index = [ topi[i][0] for i in range(config.batch_size) ]
        input = torch.LongTensor([ output_index ])
        input = input.to(C.device)
        output_indices.append(output_index)

        if t == config.caption_max_len or torch.all(input == 0):
            break

    return output_indices


def beam_search(config, beam_width, n_vocabs, decoder, input, hidden, encoder_outputs):
    input_list = [ input ]
    hidden_list = [ hidden ]
    cum_prob_list = [ torch.cuda.FloatTensor([ 1. for _ in range(config.batch_size) ]) ]

    output_list = [ [[]] for _ in range(config.batch_size) ]
    for t in range(config.caption_max_len + 1):

        outputs = None
        tmp_next_hidden_list = []
        for i, (input, hidden, cum_prob) in enumerate(zip(input_list, hidden_list, cum_prob_list)):
            output, next_hidden = decoder(input, hidden, encoder_outputs)
            tmp_next_hidden_list.append(next_hidden)

            cum_prob = cum_prob.unsqueeze(1).expand_as(output)
            output *= cum_prob
            outputs = output if outputs is None else torch.cat(( outputs, output ), dim=1)
        topk_probs, topk_flat_idxs = outputs.topk(beam_width)

        topk_probs = topk_probs.transpose(0, 1) # (beam_width, batch_size)
        topk_flat_idxs = topk_flat_idxs.transpose(0, 1) # (beam_width, batch_size)
        topk_idxs = topk_flat_idxs % n_vocabs
        topk_is = topk_flat_idxs // n_vocabs

        next_input_list = topk_idxs.clone()
        next_cum_prob_list = topk_probs.clone()
        next_hidden_list = []
        next_output_list = [ [] for _ in range(config.batch_size) ]
        for topk_idx, topk_i in zip(topk_idxs, topk_is): # Iterate <beam_width> times
            next_hidden = []
            for b, (i, k) in enumerate(zip(topk_idx, topk_i)): # Iterate <batch_size> times
                next_hidden.append(tmp_next_hidden_list[k][:, b])
                next_output_list[b].append(output_list[b][k] + [ i.item() ])
            next_hidden = torch.cat(next_hidden)
            next_hidden.unsqueeze(0)
            next_hidden_list.append(next_hidden)

        input_list = [ input.unsqueeze(0) for input in next_input_list ]
        hidden_list = [ hidden.unsqueeze(0) for hidden in next_hidden_list ]
        cum_prob_list = next_cum_prob_list
        output_list = next_output_list

        if t == config.caption_max_len or torch.all(torch.cat(input_list) == 0):
            break

    top1_output_list = [ output[0] for output in output_list ]
    return top1_output_list


def evaluate(config, corpus, data_loader, decoder, search_method):
    total_vids = []
    total_pd_captions = []
    pd_vid_caption_dict = defaultdict(lambda: [])
    for batch in iter(data_loader):
        vids, encoder_outputs = batch
        encoder_outputs = encoder_outputs.to(C.device)

        input = torch.LongTensor([ [corpus.vocab.word2idx['<SOS>'] for _ in range(config.batch_size)] ])
        input = input.to(C.device)

        if config.decoder_model == "LSTM":
            hidden = (
                torch.zeros(config.decoder_n_layers, config.batch_size, config.decoder_hidden_size).to(C.device),
                torch.zeros(config.decoder_n_layers, config.batch_size, config.decoder_hidden_size).to(C.device),
            )
        else:
            hidden = torch.zeros(config.decoder_n_layers, config.batch_size, config.decoder_hidden_size)
            hidden = hidden.to(C.device)

        if isinstance(search_method, str) and search_method == "greedy":
            output_indices = greedy_search(config, decoder, input, hidden, encoder_outputs)
        elif isinstance(search_method, tuple) and search_method[0] == "beam":
            beam_width = search_method[1]
            output_indices = beam_search(config, beam_width, corpus.vocab.n_vocabs, decoder, input, hidden, encoder_outputs)
            output_indices = np.asarray(output_indices)
            output_indices = output_indices.T
        else:
            raise NotImplementedError("Unknown search method: {}".format(config.search_method))

        total_vids += vids
        total_pd_captions += convert_idxs_to_sentences(output_indices, corpus.vocab.idx2word, corpus.vocab.word2idx['<EOS>'])

    total_vids = total_vids[:config.n_test]
    total_pd_captions = total_pd_captions[:config.n_test]
    with open("predictions.txt", 'w') as fout:
        for vid, caption in zip(total_vids, total_pd_captions):
            fout.write("{}\t\t{}\n".format(vid, caption))

    for vid, caption in zip(total_vids, total_pd_captions):
        pd_vid_caption_dict[vid].append(caption)
    gts = COCOMSVD(corpus.test_dataset.video_caption_pairs)
    res = load_res(pd_vid_caption_dict)
    cocoEval = COCOEvalCap(gts, res)
    cocoEval.params['image_id'] = gts.getImgIds()
    cocoEval.evaluate()
    return cocoEval.eval


def main():
    checkpoint = torch.load(C.model_fpath)
    TC = MockConfig()
    TC_dict = dict(checkpoint['config'].__dict__)
    for key, val in TC_dict.items():
        setattr(TC, key, val)
    TC.build_train_data_loader = False
    TC.build_val_data_loader = False
    TC.build_test_data_loader = True
    TC.build_score_data_loader = True
    TC.test_video_fpath = C.test_video_fpath
    TC.test_caption_fpath = C.test_caption_fpath

    MSVD = _MSVD(TC)
    vocab = MSVD.vocab
    score_data_loader = MSVD.score_data_loader

    decoder = Decoder(
        model_name=TC.decoder_model,
        n_layers=TC.decoder_n_layers,
        encoder_size=TC.encoder_output_size,
        embedding_size=TC.embedding_size,
        embedding_scale=TC.embedding_scale,
        hidden_size=TC.decoder_hidden_size,
        attn_size=TC.decoder_attn_size,
        output_size=vocab.n_vocabs,
        embedding_dropout=TC.embedding_dropout,
        dropout=TC.decoder_dropout,
        out_dropout=TC.decoder_out_dropout,
    )
    decoder = decoder.to(C.device)

    decoder.load_state_dict(checkpoint['dec'])
    decoder.eval()

    scores = evaluate(TC, MSVD, score_data_loader, decoder, ("beam", 5))
    print(scores)


if __name__ == "__main__":
    main()

