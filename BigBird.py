import torch

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

def BlockSparse(n,block_size,bloc_rad,g_index,r,seed):

    assert n % block_size == 0 , f"n={n} must be divisible by block_size={block_size}"
    n_block = n // block_size

    M1 = localSparse(n_block,bloc_rad)
    M2 = globalSparse(n_block,g_index)
    M3 = randomSparse(n_block,r,seed)

    allowed = (M1 == 0) | (M2 == 0) | (M3 == 0)

    M = torch.full((n_block,n_block),float("-inf"),dtype= torch.float32)
    M[allowed] = 0.0

    return M

def expand_mask(M,block_size):
    M = M.repeat_interleave(block_size, dim=0)
    M = M.repeat_interleave(block_size, dim=1)
    return M

if __name__ == "__main__":
    print(WindowMask(6, w=3, is_Casual=True))
    print(localSparse(6, block_rad=1))
    print(globalSparse(6,[0]))
    print(randomSparse(6,r=2,seed=0))
    print(BlockSparse(12, 2, 1, [0], 2, 0))
    print(expand_mask(globalSparse(6, [0]), 2))