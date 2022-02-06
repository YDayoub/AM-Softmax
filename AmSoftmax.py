import torch
import torch.nn as nn
import torch.nn.functional as F
class AdMSoftmaxLoss(nn.Module):
    
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''

        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for _, module in self.fc.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = F.normalize(module.weight, p=2, dim=1)
        
        batch_size = x.shape[0]
        num_classes = self.out_features
        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        classes = torch.arange(num_classes).long().to(x.device)
        labels = labels.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(classes.expand(batch_size, num_classes)).to(x.device)
        excl = torch.where(mask,torch.full_like(wf, -9e15),wf)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)