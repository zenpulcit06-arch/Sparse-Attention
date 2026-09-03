import Dense_ateention 
import BigBird 
import torch
import random

def candidate_window(Q, K, V, w, is_causal):
    n = Q.size()[-2]
    d_v = V.size()[-1]

    container = torch.zeros((n, d_v), dtype=torch.float32)

    for i in range(n):
        if not is_causal:
            start = i - w//2
            end = i + w//2
        else:
            start = i - w + 1
            end = i 

        start = max(0,start)
        end = min(n-1,end)

        k_sub = K[start:end+1]
        v_sub = V[start:end+1]

        q_sub  = Q[i:i+1] 

        Mask = torch.zeros((1, end-start+1), dtype=torch.float32)
        out = Dense_ateention.DenseAttention(q_sub, k_sub, v_sub).attention(Mask)
        container[i:i+1] = out

    return container

def WindowMaskHasrness(seed= None):

    print("Window Mask Test")

    if seed is not None:
        random.seed(seed)
        g = torch.Generator().manual_seed(seed)
    else:
         g = torch.Generator().manual_seed(0)

    for test in range(5):
        
        n = random.randint(7,20)
        w = random.randint(2,5)

        
        print("="*50)
        print("For not Causual")
        test_tensor = torch.zeros((n,n), dtype=torch.float32)
        for i in range(n):
            for j in range(n):
                if abs(i-j) > w//2:
                    test_tensor[i][j] = float("-inf")

        tensor = BigBird.WindowMask(n,w,False)

        if torch.equal(test_tensor,tensor):
            print(f"Test {test} Pass For not causual window mask")
        else:
            print(f"Test {test} Fails For not causual window mask")
            print(f"n: {n}, w: {w}")
            print(f"Test Tensor: \n {test_tensor}")
            print(f"Funftion Tensor: {tensor}")

        Q = torch.randn(n,10,dtype= torch.float32,generator=g)
        K = torch.randn(n,10,dtype= torch.float32,generator=g)
        V = torch.randn(n,20,dtype= torch.float32,generator=g)
        candidate_fn = lambda Q, K, V: candidate_window(Q, K, V, w, False)
        harness(tensor, n, Q, K, V, candidate_fn)


        print("="*50)
        print("For Causual")

        test_tensor = torch.zeros((n,n), dtype=torch.float32)

        for i in range(n):
            for j in range(n):
                if not (0 <= (i - j) < w):
                    test_tensor[i][j] = float(("-inf"))

        tensor = BigBird.WindowMask(n,w,True)
        if torch.equal(test_tensor,tensor):
            print(f"Test {test} Pass For causual window mask")
        else:
            print(f"Test {test} Fails For causual window mask")
            print(f"n: {n}, w: {w}")
            print(f"Test Tensor: \n {test_tensor}")
            print(f"Funftion Tensor: {tensor}")

        Q = torch.randn(n,10,dtype= torch.float32,generator=g)
        K = torch.randn(n,10,dtype= torch.float32,generator=g)
        V = torch.randn(n,20,dtype= torch.float32,generator=g)

        candidate_fn = lambda Q, K, V: candidate_window(Q, K, V, w, True)
        harness(tensor, n, Q, K, V, candidate_fn)

        print("="*50)


    print("="*50)
    n = 10
    w = 15
    test = 6
    print("For not Causual")
    test_tensor = torch.zeros((n,n), dtype=torch.float32)
    for i in range(n):
                for j in range(n):
                    if abs(i-j) > w//2:
                        test_tensor[i][j] = float("-inf")
    
    tensor = BigBird.WindowMask(n,w,False)
    
    if torch.equal(test_tensor,tensor):
                print(f"Test {test} Pass For not causual window mask")
    else:
                print(f"Test {test} Fails For not causual window mask")
                print(f"n: {n}, w: {w}")
                print(f"Test Tensor: \n {test_tensor}")
                print(f"Funftion Tensor: {tensor}")
    
    Q = torch.randn(n,10,dtype= torch.float32,generator=g)
    K = torch.randn(n,10,dtype= torch.float32,generator=g)
    V = torch.randn(n,20,dtype= torch.float32,generator=g)
    candidate_fn = lambda Q, K, V: candidate_window(Q, K, V, w, False)
    harness(tensor, n, Q, K, V, candidate_fn)
    
    
    print("="*50)
    print("For Causual")
    
    test_tensor = torch.zeros((n,n), dtype=torch.float32)
    
    for i in range(n):
                for j in range(n):
                    if not (0 <= (i - j) < w):
                        test_tensor[i][j] = float(("-inf"))
    
    tensor = BigBird.WindowMask(n,w,True)
    if torch.equal(test_tensor,tensor):
                print(f"Test {test} Pass For causual window mask")
    else:
                print(f"Test {test} Fails For causual window mask")
                print(f"n: {n}, w: {w}")
                print(f"Test Tensor: \n {test_tensor}")
                print(f"Funftion Tensor: {tensor}")
    
    Q = torch.randn(n,10,dtype= torch.float32,generator=g)
    K = torch.randn(n,10,dtype= torch.float32,generator=g)
    V = torch.randn(n,20,dtype= torch.float32,generator=g)
    
    candidate_fn = lambda Q, K, V: candidate_window(Q, K, V, w, True)
    harness(tensor, n, Q, K, V, candidate_fn)
            
    print("="*50)

def harness(M, n, Q, K, V, candidate_fn):

    reference = Dense_ateention.DenseAttention(Q,K,V).attention(M)

    candidate = candidate_fn(Q,K,V)

    valid_rows = (M == 0).any(dim = 1)

    if torch.allclose(reference[valid_rows], candidate[valid_rows],rtol=1e-5, atol=1e-6):
        print("test pass")
    else:
        print("test fail")
        print(candidate)
        print(reference)


def candidate(M, Q, K, V,block_size):
    d_v = V.size()[-1]
    n = Q.size()[-2]
    n_block = M.size()[0]

    container = torch.zeros((n,d_v),dtype= torch.float32)

    for i in range(n_block):
        allowed = torch.nonzero(M[i]==0,as_tuple= True)

        if allowed[0].numel() == 0:
            continue

        qstart = i*block_size
        qend = (i+1)*block_size

        q_block = Q[qstart:qend]

        k = []
        v = []

        for j in allowed[0].tolist():
            start = j*block_size
            end = (j+1)*block_size
            k.append(K[start:end])
            v.append(V[start:end])

        k_sub = torch.cat(k,dim=0)
        v_sub = torch.cat(v,dim=0)

        Mask = torch.zeros((block_size,allowed[0].numel()*block_size), dtype= torch.float32)

        attention = Dense_ateention.DenseAttention(q_block,k_sub,v_sub).attention(Mask)

        container[qstart:qend] = attention

    return container




if __name__ == "__main__":

    random.seed(25)
    for i in range(5):
        n_block = random.randint(15,20)
        block_size = random.randint(1,4)
        n = n_block*block_size
        block_rad = random.randint(0,9)
        r = random.randint(0,3)
        seed = random.randint(0,1000)
        g_index = [random.randint(0,n_block-1)]

        M_block = BigBird.BlockSparse(n,block_size,block_rad,g_index,r,seed)

        M_token = BigBird.expand_mask(M_block,block_size)
        g = torch.Generator().manual_seed(seed)

        Q = torch.randn(n,10,dtype= torch.float32,generator=g)
        K = torch.randn(n,10,dtype= torch.float32,generator=g)
        V = torch.randn(n,20,dtype= torch.float32,generator=g)

        candidate_fn = lambda Q, K, V: candidate(M_block, Q, K, V, block_size)

        harness(M_token,n,Q,K,V,candidate_fn)

    seed = random.randint(0,1000)
    WindowMaskHasrness(seed)