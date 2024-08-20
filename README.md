# STAGE 2024
Code for Structural-Functional Transition in Glaucoma Assessment Challenge 2024 (STAGE 2024)

## Preparing
1. Clone this repo:
```bash
git clone https://github.com/Chen-Ziyang/STAGE2024.git
cd STAGE2024
```
2. Download the pre-trained weights and move them into `model_checkpoints` folder of each task,
   which can be found in this [Baidu Disc Link](https://pan.baidu.com/s/1XnHB-hS-HF1VHHO9aDgOCQ?pwd=bh2y).

3. Download the dataset from the [official website](https://aistudio.baidu.com/competition/detail/1167/0/datasets).

4. Create the experimental environment in your own way or download ours from this [Google Drive Link](https://drive.google.com/file/d/1vAEyFrJ_wLiLwNJfBTg6J-wC9CbzCLmn/view?usp=sharing).

5. Preprocess the data by
```bash
python preprocess.py
```

## Training
1. Task 1: Prediction of Mean Deviation
```bash
python STAGE2024_Task1/train_MixLoss.py
```
2. Task 2: Sensitivity map prediction
```bash
python STAGE2024_Task2/train_MixLoss.py
```
3. Task 3: Pattern deviation probability map prediction
```bash
python STAGE2024_Task3/train.py
```

## Inference
1. Task 1: Prediction of Mean Deviation
```bash
python STAGE2024_Task1/infer.py
```
2. Task 2: Sensitivity map prediction
```bash
python STAGE2024_Task2/infer.py
```
3. Task 3: Pattern deviation probability map prediction
```bash
python STAGE2024_Task3/infer.py
```
