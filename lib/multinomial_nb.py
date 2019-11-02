import numpy as np

class MyMultinomialNB:

    def compute_prob(self, d1):

        p1 = d1["f_mle"] * self.f_mle
        p2 = d1["m_mle"] * self.m_mle

        N1 = p1
        N2 = p2
        D = p1 + p2

        d1["predict_prob"] = [ N2/D, N1/D ]
        if N1 > N2:
            d1["predict"] = 1
        else:
            d1["predict"] = 0

  
    def set_counts(self):
        self.m_count = self.f_count = 0
        gender, counts = np.unique( self.y, return_counts=True)
            
        if gender[0] == 0:
            self.m_count = counts[0]
            if len(counts)==1:
                self.f_count = 0
            else:
                self.f_count = counts[1]
        else:
            self.f_count = counts[0]
            if len(counts)==1:
                self.m_count = 0
            else:
                self.m_count = counts[1]
        self.total_count = self.m_count + self.f_count


    # this function computes basically theta_0 and theta_1
    # theta_1 = Number of female names / Total number of names
    # theta_0 = Number of male names / Total number of names
    #
    def compute_mle(self):
        self.m_mle = (self.m_count) / (self.total_count) 
        self.f_mle = (self.f_count) / (self.total_count)
      
        #with regularisation
        #self.m_mle = (self.m_count+1) / (self.total_count + len(self.uniq_letters))
        #self.f_mle = (self.f_count+1) / (self.total_count + len(self.uniq_letters))




    # this function compute theta with respect to letters ex: theta_a1 and theta_a0
    # theta_a1 = number of letter 'a' which are female / Number of female names
    # theta_a0 = number of letter 'a' which are male / Number of male names
    #    
    def compute_indvidual_mle(self):
        self.uniq_letters = np.unique(self.x)
        self.data = []
        for i, l in enumerate(self.uniq_letters):
            indices = np.where( self.x == l)[0]
            d1 = {"letter": l, "m_count": 0, "f_count": 0}

            for i in indices:
                if self.y[i] == 0:
                    d1["m_count"] += 1 
                else:
                    d1["f_count"] += 1 

            try:
                d1["m_mle"] = (d1["m_count"]) / (self.m_count )
                d1["m_rmle1"] = (d1["m_count"]+1) / (self.m_count + len(self.uniq_letters) )
            except:
                d1["m_mle"] = 0

            try:
                d1["f_mle"] = (d1["f_count"]) / (self.f_count )
                d1["f_rmle1"] = (d1["f_count"]+1) / (self.f_count + len(self.uniq_letters) )
            except:
                d1["f_mle"] = 0

            self.compute_prob(d1)
            self.data.append(d1)


    def fit(self, x, y):
        self.x = x
        self.y = y

        self.set_counts()
        self.compute_mle()
        self.compute_indvidual_mle()



    def predict(self, letter):
        l = [l_dict for l_dict in self.data if l_dict["letter"] == letter]
        if l:
            return l[0]["predict"]
        return [-1, -1]
    
    def predict_prob(self, letter):
        l = [l_dict for l_dict in self.data if l_dict["letter"] == letter]
        if l: 
            return [self.m_mle, self.f_mle]
        return [-1, -1]


