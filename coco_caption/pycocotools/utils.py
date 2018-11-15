class Mock:
    pass


def load_res(imgToAnns):
    mock = Mock()
    mock.imgToAnns = {}
    for vid, captions in imgToAnns.items():
        mock.imgToAnns[vid] = [ { 'caption': caption } for caption in captions ]
    return mock

