import chainer.functions as F

def cosine_similarity(x, y, eps=1e-6):
    n1, n2, n3 = x.data.shape
    _, m2, _ = y.data.shape
    z = F.batch_matmul(x, y, transb=True)
    x2 = F.broadcast_to(F.reshape(F.sum(x * x, axis=2), (n1, n2 ,1)), (n1, n2, m2))
    y2 = F.broadcast_to(F.reshape(F.sum(y * y, axis=2), (n1, 1, m2)), (n1, n2, m2))
    z /= F.exp(F.log(x2 * y2 + eps) / 2)
    return z
