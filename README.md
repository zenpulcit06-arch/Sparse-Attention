# Sparse-Attention

This project aims to build Sparse-Attention from scratch While learning it and pytorch in the process

## Layout
### (At current stage)

```text
Sparse-Attention/
└── Dense_Attention.py (Scaled Dot Product)
```


## Requirments

```bash
    pip install torch numpy matplotlib pytest
```

## How to run 
### (At this point)
```bash
python Dense__ateention.py
```


## Bug Encounters
### Checkpoint 1
<details>
<summary> <b> Bug 1: </b> Invalid data type </summary>
    durin tensor converion some are onverting to int while other are cinverting to float
    <b> Fix </b>
    ```python 
        dtype = torch.float32 
    ``` 
</details>

## Checkpoint completed
<details>
<summary> <b> Checkpoint1: </b> </summary> 
    Scaled Dot Product (Dense_ateention.py)
</details>