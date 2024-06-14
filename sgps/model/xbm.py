import torch

class XBM:
    def __init__(self, cfg):
        self.K = cfg.XBM.SIZE
        self.feats = torch.zeros(self.K, cfg.MODEL.HEAD.DIM).cuda()
        self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.indices = torch.zeros(self.K, dtype=torch.long).cuda()
        self.feats.requires_grad=False
        self.targets.requires_grad=False
        self.indices.requires_grad=False
        self.targets[:]=-1
        self.indices[:]=-1
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def load_from_file(self, cfg):
        feats = torch.load(cfg.XBM.FEATURE_FILE)
        targets = torch.load(cfg.XBM.TARGET_FILE)
        leng = feats.shape[0]
        if leng < self.K:
            self.feats[:leng] = feats
            self.targets[:leng] = targets
            self.ptr = leng
        else:
            self.feats[:] = feats[:self.K]
            self.targets[:] = targets[:self.K]
            self.ptr = self.K
        
    def get(self):
        if self.is_full:
            return self.feats, self.targets, self.indices
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr], self.indices[:self.ptr]

    def enqueue_dequeue(self, feats, targets, indices=None):
        
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.indices[-q_size:] = indices
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.indices[self.ptr: self.ptr + q_size] = indices
            self.ptr += q_size