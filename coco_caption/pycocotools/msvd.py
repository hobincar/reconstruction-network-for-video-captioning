from collections import defaultdict


class MSVD:
    def __init__(self, video_caption_pairs):
        self.video_caption_pairs = video_caption_pairs
        
        self.imgToAnns = defaultdict(lambda: [])
        for vid, _, caption in self.video_caption_pairs:
            self.imgToAnns[vid].append({ 'caption': caption })

    def getImgIds(self):
        return list(self.imgToAnns.keys())

