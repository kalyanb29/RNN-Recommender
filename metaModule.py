from net import *
class MetaLearner:
    def __init__(self, metalearner, input_size, output_size, hidden_size, nlayers, batch_size = 16, bidirectional = True):
        super(MetaLearner, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.bidirectional = 2 if bidirectional else 1
        self.metalearner = metalearner.cuda() if metalearner is not None else \
            LSTM(input_size, output_size, hidden_size, nlayers, bidirectional=bidirectional).cuda()
        self.actions = []
        self.batch_size = batch_size

    def takeAction(self, inputs, batch_size=1):
        if batch_size is None:
            hn = Variable(torch.zeros(self.nlayers * self.bidirectional, self.batch_size, self.hidden_size)).cuda()
            cn = Variable(torch.zeros(self.nlayers * self.bidirectional, self.batch_size, self.hidden_size)).cuda()
        else:
            hn = Variable(torch.zeros(self.nlayers * self.bidirectional, batch_size, self.hidden_size)).cuda()
            cn = Variable(torch.zeros(self.nlayers * self.bidirectional, batch_size, self.hidden_size)).cuda()
        output = self.metalearner(inputs, (hn, cn))
        return output
