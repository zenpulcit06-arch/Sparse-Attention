# Sparse-Attention

This project aims to build Sparse-Attention from scratch While learning it and pytorch in the process

## Layout
### (At current stage)

```text
Sparse-Attention/
└── Dense_Ateention.py (Scaled Dot Product)
|__ BigBird.py (Sparse pattern)
|_ harness.py
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

### for harness
```bash
python harness.py
```


## Bug Encounters
### Checkpoint 1
<details>
<summary> <b> Bug 1: </b> Invalid data type </summary>

    durin tensor converion some are onverting to int while other are cinverting to float
     **Fix** 
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

### Checkpoint3
<detail>
<summary> <b> Bug4: </b> Made a wrong Window harness  </summary>
    I build a wrong window harness which only measure the dimensions not the values
</detail>

<detail>
<summary> <b> Bug5: </b> Wrong input to candidate function  </summary>
    Initaly my candidate tooks wrong function as input
</detail>


## Checkpoint completed
<details>
<summary> <b> Checkpoint1: </b> </summary>
 
    Scaled Dot Product (Dense_ateention.py)
</details>

<details>
<summary> <b> Checkpoint2: </b> </summary>
 
    BigBird sparse pattern (BigBird.py)
</details>

<details>
<summary> <b> Checkpoint3: </b> </summary>
 
    Added the harness (harness.py)
</details>