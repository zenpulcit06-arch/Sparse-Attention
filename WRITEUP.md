# Write Up

# Resource whre I Learn

### Checkpoint 1
<details>
<Summary>3Blue1Brown </summary> 
https://youtu.be/eMlx5fFNoYc?si=tdTtkdiyvOygawi0
</details>

<details>
<summary> AI Used for Help </summary> 
Claude (Only Bug testing and resource gathering)
</details>

## Strategy Used

### Checkpoint 1

First intialise a DenseAttention class With **Querry (Q), Key (K) and value (V)** each sized **(batch,head,seq_len,head_dim)**
```Python
    class DenseAttention:
    def __init__(self,Q,K,V) -> None:
        self.Q = Q if torch.is_tensor(Q) else torch.tensor(Q,dtype= torch.float32) 
        self.K = K if torch.is_tensor(K) else torch.tensor(K, dtype= torch.float32)
        self.V = V if torch.is_tensor(V) else torch.tensor(V, dtype= torch.float32)
        self.d = self.K.size()[-1]
```


Then I use the **scaled dot product attention** 
    $$
        \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$

    ```python 
    def attention(self,M):

        self.M = M if torch.is_tensor(M) else torch.tensor(M, dtype= torch.float32)

        scores = torch.matmul(self.Q,torch.transpose(self.K,-2,-1))/(self.d**0.5)
        Masked_score = scores + self.M
        softmax = torch.softmax(Masked_score,dim=-1)

        return torch.matmul(softmax,self.V)
    
    ```

#### Test 
    Tested along hand work of 2x2 Matrix

    ```Python
        if __name__ == "__main__":

        Q_test = [[1,0],[0,1]]
        K_test = [[1,0],[0,1]]
        V_test = [[10,20],[30,40]]

        Dense = DenseAttention(Q_test,K_test,V_test)
        M_test = [[0, float('-inf')], [0, 0]]

        print(Dense.attention(M_test))
    ```

    ```Result 
        tensor([[10.0000, 20.0000],
        [23.3952, 33.3952]])
    ```