# Kuhn Poker

This repo has an almost fully-implemented version of kuhn poker, along with the History and Information set classes needed to implement the cfr algorithm. All that's left is to implement the run-trees function in cfr.py 

## Installation
To run the code you have to install 

```bash
pip install labml-nn
```

## Usage
Once your code is fully written, run
```bash
python3 train_model.py
```
Which, if done correctly, should output (the numbers shown by your model might be slightly different):
```bash
Train...[DONE]  64,078.90ms                                                                         
 A:  100.0%                                                                                         
Ab:  100.0%
 K:  0.0%
Kb:  33.4%
 Q:  33.6%
Qb:  0.0%
```
