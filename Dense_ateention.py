import torch 

class DenseAttention:
    def __init__(self,Q,K,V) -> None:
        self.Q = Q if torch.is_tensor(Q) else torch.tensor(Q,dtype= torch.float32) 
        self.K = K if torch.is_tensor(K) else torch.tensor(K, dtype= torch.float32)
        self.V = V if torch.is_tensor(V) else torch.tensor(V, dtype= torch.float32)
        self.d = self.K.size()[-1]

    def attention(self,M):

        self.M = M if torch.is_tensor(M) else torch.tensor(M, dtype= torch.float32)

        scores = torch.matmul(self.Q,torch.transpose(self.K,-2,-1))/(self.d**0.5)
        Masked_score = scores + self.M
        softmax = torch.softmax(Masked_score,dim=-1)

        return torch.matmul(softmax,self.V)

if __name__ == "__main__":

    Q_test = [[1,0],[0,1]]
    K_test = [[1,0],[0,1]]
    V_test = [[10,20],[30,40]]

    Dense = DenseAttention(Q_test,K_test,V_test)
    M_test = [[0, float('-inf')], [0, 0]]

    print(Dense.attention(M_test))