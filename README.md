Emotion-Controllable Speech-Driven 3D Facial Animation Generation Based on Normalizing Flows
Official PyTorch implementation for the paper:

基于归一化流的情感可控语音驱动三维面部动画生成
Authors: 刘文静, 谢文军, 韩汇东, 李琳, 刘晓平
Published in: 小型微型计算机系统, 2025 (已录用)


We propose a normalizing flow-based approach for emotion-controllable speech-driven 3D facial animation generation. Our method leverages the powerful modeling capabilities of normalizing flows to learn the complex mapping between speech features and facial motion parameters, while incorporating emotion control mechanisms to generate diverse and expressive facial animations.

Environment
<details><summary>Click to expand</summary>
System Requirement

Linux and Windows (tested on Windows 10)
Python 3.9+
PyTorch 2.1.1
CUDA 12.1 (GPU with at least 4GB VRAM)

Virtual Environment
To run our program, first create a virtual environment. We recommend using miniconda or miniforge. Once Miniconda or Miniforge is installed, open Command Prompt (make sure to run it as Administrator on Windows) and run the following commands:
bashconda create --name 3dface python=3.9
conda activate 3dface
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
Then, navigate to the project root folder and execute:
bashpip install -r requirements.txt
</details>
Dataset
<details><summary>Click to expand</summary>
Download 3DMEAD dataset following the standard processing pipeline. This dataset represents facial animations using FLAME parameters.
Data Download and Preprocess

The dataset should be organized with *.npy files in the datasets/mead/param folder, and the .wav files should be in the datasets/mead/wav folder.
Ensure proper audio preprocessing with 16kHz sampling rate.
FLAME parameters should be organized as [shape, expression, jaw] with dimensions [300, 50, 3] respectively.

Data Structure
datasets/
├── mead/
│   ├── param/          # FLAME parameter files (.npy)
│   ├── wav/            # Audio files (.wav)
│   └── templates_mead_vert.pkl  # Template file
└── regions/            # Facial region definitions
    ├── lve.txt        # Lip vertex indices
    ├── fdd.txt        # Upper face indices
    └── eve.txt        # Eye region indices
</details>
Model Training
<details><summary>Click to expand</summary>
Our training approach consists of multiple stages, incorporating motion prior learning and emotion-controllable generation.
Stage 1: Motion Prior Training
First, train the motion autoencoder (VAE variant) to learn the motion prior:
bashpython main.py --model face_diffuser --dataset mead --max_epoch 100
Stage 2: Emotion-Controlled Training
Train the emotion-controllable model with graph-enhanced condition fusion:

On Windows and Linux:
bashpython main.py --model face_diffuser --dataset mead --max_epoch 100 --save_path outputs/model_graph

If using Slurm Workload Manager:
bashsbatch train_facediff.sh


Training Parameters

--lr: Learning rate (default: 0.00001)
--feature_dim: Latent dimension (default: 256)
--max_epoch: Number of training epochs (default: 100)
--gradient_accumulation_steps: Gradient accumulation steps (default: 1)
--train_subjects: Training subject IDs
--device: GPU device (default: "cuda:0")

</details>
Evaluation
<details><summary>Click to expand</summary>
Quantitative Evaluation
We provide code to compute the evaluation metrics mentioned in our paper:
bashpython evaluation.py --save_path "outputs/model_vae_diffusion" --max_epoch 85 --num_samples 1
Evaluation Metrics

MVE (Mean Vertex Error): Average vertex displacement error
LVE (Lip Vertex Error): Lip region motion accuracy
EVE (Eye Vertex Error): Eye region motion accuracy

Optional Parameters

--num_samples: Number of samples to generate per audio (default: 1)
--skip_steps: Number of diffusion steps to skip during inference (default: 900)
--test_subjects: Specify test subjects for evaluation

</details>
Animation Generation
<details><summary>Click to expand</summary>
Generate Predictions
Our model supports generation across 32 speaking styles (IDs), 8 emotions, and 3 intensity levels.
Available Conditions
<details><summary>Click to expand</summary>
Subject IDs:
M003, M005, M007, M009, M011, M012, M013, M019,
M022, M023, M024, M025, M026, M027, M028, M029,
M030, M031, W009, W011, W014, W015, W016, W018,
W019, W021, W023, W024, W025, W026, W028, W029
Emotions:
0: neutral, 1: happy, 2: sad, 3: surprised, 
4: fear, 5: disgusted, 6: angry, 7: contempt
Intensity Levels:
0: low, 1: medium, 2: high
</details>
Basic Generation
bashpython predict.py --subject "M009" --id "M009" --emotion 6 --intensity 1 --wav_path "path/to/audio.wav"
Advanced Generation Options

Multiple Samples: Generate diverse outputs for the same input
Temperature Control: Adjust generation diversity (0.1-0.5 recommended)
Deterministic Mode: Disable stochastic sampling for consistent results

Rendering
The generated .npy files contain FLAME parameters and can be rendered into videos using the provided rendering pipeline.
</details>
Model Architecture
<details><summary>Click to expand</summary>
Key Components

Audio Encoder: HuBERT-based feature extraction
Motion Prior: VAE-based motion autoencoder
Emotion Graph: Graph neural network for emotion modeling
Condition Fusion: Enhanced multi-modal feature integration
Normalizing Flow: Probabilistic motion generation

Emotion Graph Architecture

Node Representations: Learnable emotion and intensity embeddings
Graph Attention: Multi-head attention for emotion relationships
Intensity Modulation: Progressive intensity control mechanism

Technical Details

Input: 16kHz audio, emotion labels, intensity levels, subject IDs
Output: FLAME parameters (shape: 300, expression: 50, jaw: 3)
Training: Progressive training with motion prior and emotion control
Inference: Stochastic sampling with temperature control

</details>
File Structure
<details><summary>Click to expand</summary>
├── data_loader.py              # Data loading and preprocessing
├── evaluation.py               # Quantitative evaluation script
├── main.py                     # Main training script
├── predict.py                  # Animation generation script
├── motion_prior.py             # Motion autoencoder (VAE/VQVAE)
├── motion_pred.py              # Motion prediction models
├── model_pred_emotion_graph.py # Emotion graph-enhanced predictor
├── model_pred_other.py         # Alternative model implementations
├── utils.py                    # Utility functions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
</details>
Citation
If you find this code useful for your work, please consider citing:
bibtex@article{liu2025emotion,
    title={基于归一化流的情感可控语音驱动三维面部动画生成},
    author={刘文静 and 谢文军 and 韩汇东 and 李琳 and 刘晓平},
    journal={小型微型计算机系统},
    year={2025},
    note={已录用}
}
Acknowledgements
We thank the authors of the following projects for their contributions to the research community:

Learning to Listen
CodeTalker
TEMOS
FaceXHuBERT
FaceDiffuser

We are grateful to the creators of the 3DMEAD dataset used in this project.
Any third-party packages are owned by their respective authors and must be used under their respective licenses.
License
This repository is released under CC-BY-NC-4.0-International License.
