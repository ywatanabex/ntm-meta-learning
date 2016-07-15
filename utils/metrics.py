from collections import defaultdict


def _to_score(bools):
    a = 0
    score = []
    for k, b in enumerate(bools, 1):
        a += b.astype('float')
        score.append(a / k)
    return score
        

def few_shot_score(accs, labels):
    """
    labels : len_seq x batchsize
    accs: len_seq x batchsize
    """
    #label_set = set([l for lab in labels for l in lab])

    accs_ = zip(*accs)
    labels_ = zip(*labels)
    scores = []
    for acc, lab in zip(accs_, labels_):
        res = defaultdict(list)
        for a, l in zip(acc, lab):
            res[l].append(a)
        scores.append([ _to_score(res[l]) for l in res.keys()])      
    return scores

           

        

