# Write Up

# Resource whre I Learn

### Checkpoint 1
<details>
<summary>3Blue1Brown </summary> 

https://youtu.be/eMlx5fFNoYc?si=tdTtkdiyvOygawi0
</details>

### Checkpoint 2
<details>
<summary> InnerworkingAi </summary>

https://youtu.be/XCcaAQujhXY?si=HRBC6EVan3Nv9ORW
</details>


### Checkpoint 3
<details>
<summary> hugging face </summary>
https://huggingface.co/docs/transformers/en/model_doc/big_bird
</details>

### In general
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
$$ \[\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\] $$

```python 

    def attention(self,M):

        self.M = M if torch.is_tensor(M) else torch.tensor(M, dtype= torch.float32)

        scores = torch.matmul(self.Q,torch.transpose(self.K,-2,-1))/(self.d**0.5)
        Masked_score = scores + self.M
        softmax = torch.softmax(Masked_score,dim=-1)

        return torch.matmul(softmax,self.V)
    
```

### Checkpoint 2

First calculated Window mask of by slidng the M 

```python

def WindowMask(n,w = 8, is_Casual = False):
    row = torch.arange(n).unsqueeze(0)
    col = torch.arange(n).unsqueeze(1)

    if not is_Casual:
        allowed = (row - col).abs() <=(w//2)
    else:
        allowed = ((row - col) <= 0) & ((row - col) > -w)

    M = torch.full((n,n),float("-inf"),dtype= torch.float32)
    M[allowed] = 0.0

    return M
```

Then calculate the local, global and random sparse

```python
def localSparse(n_block, block_rad):
    return WindowMask(n_block,w = 2*block_rad + 1)

def globalSparse(n_block,g_index):
    M = torch.full((n_block,n_block),float("-inf"), dtype= torch.float32)
    M[g_index,:] = 0.0
    M[:,g_index] = 0.0
    return M

def randomSparse(n_block,r,seed):

    g = torch.Generator().manual_seed(seed)

    col_idx = torch.randint(0, n_block, (n_block, r), generator=g)
    row_idx = torch.arange(n_block).unsqueeze(1).expand(n_block,r)


    M = torch.full((n_block,n_block),float("-inf"), dtype= torch.float32)

    M[row_idx,col_idx] = 0.0

    return M
```

Then i have ored result from each 

```python
    M1 = localSparse(n_block,bloc_rad)
    M2 = globalSparse(n_block,g_index)
    M3 = randomSparse(n_block,r,seed)

    allowed = (M1 == 0) | (M2 == 0) | (M3 == 0)
```

### Checkpoint 3
 No code here since motive of this to check the output of the **Dense Attention** but strategy is simple create a candidate function to calculate manualy the effecive value then match it using the function

## Test 

Test wirttern per check point

### Checkpoint1
Tested along hand work of 2x2 Matrix

```python

if __name__ == "__main__":

    Q_test = [[1,0],[0,1]]
    K_test = [[1,0],[0,1]]
    V_test = [[10,20],[30,40]]

    Dense = DenseAttention(Q_test,K_test,V_test)
    M_test = [[0, float('-inf')], [0, 0]]

    print(Dense.attention(M_test))
```

**result**
```text

tensor([[10.0000, 20.0000],
[23.3952, 33.3952]])
```

### Checkpoint2

tested along matrix and checked by seeing 
 ```python
 if __name__ == "__main__":
    print(WindowMask(6, w=3, is_Casual=True))
    print(localSparse(6, block_rad=1))
    print(globalSparse(6,[0]))
    print(randomSparse(6,r=2,seed=0))
    print(BlockSparse(12, 2, 1, [0], 2, 0))
    print(expand_mask(globalSparse(6, [0]), 2))
```

**result**

```text
tensor([[0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf],
        [-inf, 0., 0., 0., -inf, -inf],
        [-inf, -inf, 0., 0., 0., -inf],
        [-inf, -inf, -inf, 0., 0., 0.]])
tensor([[0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf],
        [-inf, 0., 0., 0., -inf, -inf],
        [-inf, -inf, 0., 0., 0., -inf],
        [-inf, -inf, -inf, 0., 0., 0.],
        [-inf, -inf, -inf, -inf, 0., 0.]])
tensor([[0., 0., 0., 0., 0., 0.],
        [0., -inf, -inf, -inf, -inf, -inf],
        [0., -inf, -inf, -inf, -inf, -inf],
        [0., -inf, -inf, -inf, -inf, -inf],
        [0., -inf, -inf, -inf, -inf, -inf],
        [0., -inf, -inf, -inf, -inf, -inf]])
tensor([[-inf, -inf, 0., 0., -inf, -inf],
        [0., -inf, -inf, -inf, -inf, 0.],
        [-inf, 0., -inf, 0., -inf, -inf],
        [-inf, 0., -inf, -inf, -inf, -inf],
        [-inf, 0., -inf, 0., -inf, -inf],
        [-inf, -inf, 0., -inf, -inf, 0.]])
tensor([[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., -inf, -inf, 0.],
        [0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., -inf],
        [0., 0., -inf, 0., 0., 0.],
        [0., -inf, 0., -inf, 0., 0.]])
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf]])
```


### Checkpoint3

```text
test pass
test pass
test pass
test pass
test pass
Window Mask Test
==================================================
For not Causual
Test 0 Pass For not causual window mask
test pass
==================================================
For Causual
Test 0 Pass For causual window mask
test pass
==================================================
==================================================
For not Causual
Test 1 Pass For not causual window mask
test pass
==================================================
For Causual
Test 1 Pass For causual window mask
test pass
==================================================
==================================================
For not Causual
Test 2 Pass For not causual window mask
test pass
==================================================
For Causual
Test 2 Pass For causual window mask
test pass
==================================================
==================================================
For not Causual
Test 3 Pass For not causual window mask
test pass
==================================================
For Causual
Test 3 Pass For causual window mask
test pass
==================================================
==================================================
For not Causual
Test 4 Pass For not causual window mask
test pass
==================================================
For Causual
Test 4 Pass For causual window mask
test pass
==================================================
==================================================
For not Causual
Test 6 Pass For not causual window mask
test pass
==================================================
For Causual
Test 6 Pass For causual window mask
test pass
==================================================
```