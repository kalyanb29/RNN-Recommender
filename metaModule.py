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

    def takeAction(self, input):
        hn = Variable(torch.zeros(self.nlayers * self.bidirectional, self.batch_size, self.hidden_size)).cuda()
        cn = Variable(torch.zeros(self.nlayers * self.bidirectional, self.batch_size, self.hidden_size)).cuda()
        input = Variable(input).cuda()
        output = self.metalearner(input, (hn, cn))
        return output
