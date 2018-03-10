class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps


    def update(self, t):
        if(t > self.nsteps):
            self.epsilon = self.eps_end
        else:
            self.epsilon = self.eps_begin \
                + 1.0 * (self.eps_end - self.eps_begin) * t / self.nsteps