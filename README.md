# Sparse-Attention

This project aims to build Sparse-Attention from scratch While learning it and pytorch in the process

## Layout
### (At current stage)

```text
Sparse-Attention/
└── Dense_Ateention.py (Scaled Dot Product)
|__ BigBird.py (Sparse pattern)
```


## Requirments

```bash
    pip install torch numpy matplotlib pytest
```

## How to run 
### (At this point)
```bash
python Dense__ateention.py
python BigBird.py
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

### Checkpoint 2
<details>
<summary> <b> Bug2: </b> USed numpy instead of torch </summary>

    instead of using torcch to build tensor i used numpy
</details>

<details>
<summary> <b> Bug3: </b> Addes all index instead of or </summary>

    during calculating BigBird. I added M1+M2+M3 instead of M1 == 0 | M2 == 0 | M3 == 0
</details>

## Checkpoint completed
<details>
<summary> <b> Checkpoint1: </b> </summary>
 
    Scaled Dot Product (Dense_ateention.py)
</details>

<details>
<summary> <b> Checkpoint2: </b> </summary>
 
    BigBird sparse pattern (BigBird.py)
</details>