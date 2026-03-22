# Course Instructions and Projects

## 📚 Initial Instructions for Students

This repository is the official starting point for all course projects. Here are the steps to get started:
1. **Choose a project**: Consult the [project list below](#project-list) to read the available tracks and check which ones are free or already assigned. Then communicate your chosen project to the professor via email.
2. **Fork**: Create a **fork** of this repository in your personal GitHub account. While it is preferable to use the Fork button in the top right (to keep the history visible for evaluation), you can also create a standalone repository and keep it private if you prefer.
3. **Clone**: Clone your fork locally.
4. **Work in this root**: Consult [CONTRIBUTING.md](CONTRIBUTING.md) for the required conventions on how to structure folders (`src/`, `data/`, `notebooks/`), how to write clean code, and how to use Git professionally in a team. Replace the placeholders in the `README.md` file with the technical information about the repository and `docs/REPORT.md` with the description of the project and the work done.
5. **AI Usage Policy**: The use of generative AI tools (ChatGPT, GitHub Copilot, Claude, etc.) is **permitted, but regulated**. The use of these tools is encouraged to speed up boilerplate code writing, for debugging, or as documentation support. However, **never delegate strategic thinking and architectural choices to AI**. Elaborate your strategy, write or generate the code, and take full responsibility for every line. The use of such tools must be explicitly declared in the final report.
6. **License**: It is good practice to release your work open source. You will find a `LICENSE` file (pre-set to MIT license). Open the file, replace `[Year]` and `[Name and Surname]` with the current year and the members of your team. Remember to choose a different one if you do not want to freely share your code.
7. **Submission**: Your GitHub fork is the **final deliverable** of the project. Ensure the code is reproducible following the instructions below and that the slides for the exam presentation are placed inside the `docs/` folder. If you opted for a private repository, evaluation can take place by making the repo visible to the professor (handle `antoninofurnari`) or by sending the repository source code via email.

---

This file contains the list of available projects, complete details for each project, and the formed groups.

## Project List

| ID | Title | Module | Difficulty | Assigned |
| :---: | :--- | :--- | :--- | :--- |
| 3 | [Metric Learning for Egocentric Face Recognition](#traccia-3) | Metric Learning | Beginner | Free |
| 4 | [Few-shot Learning for Gesture Recognition](#traccia-4) | Metric Learning | Intermediate | Free |
| 5 | [Graph-based Metric Learning for Scene Understanding](#traccia-5) | Metric Learning | Advanced | Free |
| 6 | [Knowledge Distillation for Mobile Action Recognition](#traccia-6) | Knowledge Distillation | Beginner | Free |
| 7 | [Domain Adaptation for Action Recognition – Egocentric → Exocentric](#traccia-7) | Domain Adaptation | Intermediate | Free |
| 8 | [Domain Adaptation with Image-to-Image Translation (CycleGAN)](#traccia-8) | Domain Adaptation | Intermediate | Free |
| 9 | [Multi-source Domain Adaptation for Action Recognition](#traccia-9) | Domain Adaptation | Advanced | Free |
| 10 | [Contrastive Learning for Video Representation (SimCLR Video)](#traccia-10) | Self-Supervised Learning | Beginner | Free |
| 11 | [Masked Video Modeling (MAE-style) for Egocentric Video](#traccia-11) | Self-Supervised Learning | Intermediate | Free |
| 12 | [Clustering-based Self-Supervised Learning for Action Discovery](#traccia-12) | Self-Supervised Learning | Intermediate | Free |
| 13 | [Temporal Action Localization with 1D CNN](#traccia-13) | Video Understanding | Beginner | Free |
| 14 | [Action Recognition with Vision Transformer (ViT-based)](#traccia-14) | Video Understanding | Intermediate | Free |
| 15 | [Vision-Language Alignment with CLIP for Video](#traccia-15) | Vision & Language | Intermediate | Free |
| 16 | [Multimodal Action Recognition – Video + Audio + Text](#traccia-16) | Vision & Language | Advanced | Free |
| 17 | [Egocentric Video + Gaze for Procedural Understanding](#traccia-17) | Video Understanding | Intermediate | Free |
| 18 | [State-Space Models (Mamba) for Long Sequences](#traccia-18) | Advanced Sequential Modeling | Advanced | Free |
| 19 | [Transformer vs RNN for Procedural Video Understanding](#traccia-19) | Advanced Sequential Modeling | Intermediate | Free |
| 20 | [Diffusion Models for Trajectory/Motion Generation](#traccia-20) | Advanced Sequential Modeling | Intermediate | Free |
| 21 | [Deep Q-Learning for Frame Selection in Video](#traccia-21) | Reinforcement Learning | Beginner | Free |
| 22 | [Policy Gradient for Gesture Control](#traccia-22) | Reinforcement Learning | Intermediate | Free |
| 23 | [Multi-agent RL for Task Coordination](#traccia-23) | Reinforcement Learning | Advanced | Free |
| 24 | [Differentiable Task Graphs (Yao method) – Group A](#traccia-24) | Research Topic (Graphs/Procedural) | Advanced | Free |
| 25 | [Task Graphs – Softmax vs Sum Feasibility – Group B](#traccia-25) | Research Topic (Graphs/Procedural) | Advanced | Free |
| 26 | [Procedural Error Detection with Gaze – Group A](#traccia-26) | Research Topic (Egocentric/Multimodal) | Intermediate | Free |
| 27 | [Error Detection – Progress-Aware Model – Group B](#traccia-27) | Research Topic (Egocentric/Multimodal) | Intermediate | Free |
| 28 | [Graph Autoencoder for Geometric Representations](#traccia-28) | Research Topic (Graphs/Representation) | Advanced | Free |
| 29 | [Hyperbolic Embeddings for Action Hierarchy](#traccia-29) | Research Topic (Advanced Representations) | Intermediate | Free |
| 30 | [Generative Models for Data Augmentation in Egocentric Domain](#traccia-30) | Research Topic (Egocentric/Generative) | Intermediate | Free |
| 31 | [Online Episodic Memory for Action Anticipation](#traccia-31) | Research Topic (Memory/Anticipation) | Advanced | Free |

---

## Detailed Project Descriptions

<a id='traccia-3'></a>
### Track 3: Metric Learning for Egocentric Face Recognition
**Difficulty**: Beginner  
**Module**: Metric Learning  
**When to start**: After the Metric Learning lecture (24/03/26)

#### Problem Description
Recognize faces seen from an egocentric perspective (e.g., in videos from smart glasses). The challenge is that faces are often partial, showing only eyes or profiles, and lighting conditions are extreme.

#### Dataset
- **EGTEA Gaze+** (subset with annotated faces) or synthetic
- ~100–200 identities, 5–10 samples per identity
- Low resolution frames (egocentric ≈ 720p)

#### Minimum Objectives
1. CNN backbone (ResNet-18 fine-tuned)
2. Triplet loss with hard negative mining: given an anchor face, find positives (same person) and negatives (other people)
3. Retrieval evaluation: mAP @1, 5, 10 (if I show a face, does the model retrieve the same person in the top-10 results more frequently?)
4. Cluster analysis in latent space: are faces of the same person close together?

#### Extra Objectives
- Comparison between triplet loss vs ArcFace (margin-based)
- Sampling strategies for triplets (online mining vs offline)
- Robustness to glasses/masks (add occlusion during tests)

---

<a id='traccia-4'></a>
### Track 4: Few-shot Learning for Gesture Recognition
**Difficulty**: Intermediate  
**Module**: Metric Learning  
**When to start**: After the Metric Learning lecture (24/03/26)

#### Problem Description
Recognize gestures (hands or body) from the last 1–5 seen examples. This is useful when the gesture is rare or new.

#### Dataset
- **miniImageNet** adapted to 2D skeleton (or short gesture videos)
- Or a DIY dataset: 10 gestures, 5–10 examples per gesture
- Skeleton coordinates (e.g., from MediaPipe)

#### Minimum Objectives
1. Feature encoder: 1D CNN on skeleton coordinates (or 2D CNN on gesture images)
2. Prototypical Networks: calculate a prototype for each class (average of the support set embeddings)
3. Query classification: compare query embedding with prototypes (L2 distance)
4. Metric: Accuracy 5-way 1-shot, 5-way 5-shot
5. Report: how does performance change with more examples?

#### Extra Objectives
- Relation Networks (learn a distance metric instead of using L2 L2)
- Domain adaptation: pre-train on one person's gestures, fine-tuning on another
- Failure case analysis

---

<a id='traccia-5'></a>
### Track 5: Graph-based Metric Learning for Scene Understanding
**Difficulty**: Advanced  
**Module**: Metric Learning  
**When to start**: After the Metric Learning lecture (24/03/26) + deep dive

#### Problem Description
Represent scenes (e.g., kitchen, office) as graphs (objects = nodes, spatial relations = edges) and learn robust embeddings for scene-to-scene retrieval.

#### Dataset
- **Visual Genome** (scene graphs: ~100k images with annotated objects and relations)
- Subset: ~500 scenes with non-complex graphs (5–15 nodes)

#### Minimum Objectives
1. Scene graph encoder: GCN/GraphSAGE that processes the graph and produces an embedding
2. Contrastive loss: pairs of similar scenes (same place, same activity) must have close embeddings
3. Retrieval: given a query graph, find the most similar scene graphs in the dataset
4. Metric: mAP on retrieval, cluster purity

#### Extra Objectives
- Dynamic graph: extract scene graphs from videos (nodes = object tracks, edges = temporal interactions)
- Robustness to perturbations (remove nodes/edges from the test graph, check if retrieval degrades)
- Interpretability: which edges are critical for similarity?

---

<a id='traccia-6'></a>
### Track 6: Knowledge Distillation for Mobile Action Recognition
**Difficulty**: Beginner  
**Module**: Knowledge Distillation  
**When to start**: After the Knowledge Distillation lecture (07/04/26)

#### Problem Description
Compress a heavy video model (e.g., 3D ResNet-50) into a lightweight one (e.g., MobileNet) maintaining performance, for deployment on mobile devices.

#### Dataset
- **HMDB-51** or **UCF-101** (sports/daily actions)
- ~1000–2000 videos, 51 classes
- Video features available online

#### Minimum Objectives
1. **Teacher**: Pre-trained 3D ResNet-50 (baseline accuracy on the test set)
2. **Student**: MobileNet 3D (light version, e.g., 5–10x fewer parameters)
3. **KD Loss**: L_KD = α * L_CE(student, hard_labels) + β * L_KL(student, teacher)
4. Training loop: student learns from the soft output of the teacher
5. Metrics:
   - Accuracy comparison (teacher vs student no KD vs student + KD)
   - Model size (MB)
   - Inference time (ms)

#### Extra Objectives
- Temperature tuning: how does performance change with T = 1, 5, 10, 20?
- Attention transfer: not only logits, but also intermediate activation maps
- Visualization of what the teacher transmits to the student (t-SNE of the latent space)

---

<a id='traccia-7'></a>
### Track 7: Domain Adaptation for Action Recognition – Egocentric → Exocentric
**Difficulty**: Intermediate  
**Module**: Domain Adaptation  
**When to start**: After the Domain Adaptation lecture (16/04/26)

#### Problem Description
A model trained on egocentric videos (from smart glasses) does not work well on exocentric videos (third-person view). Use Domain Adaptation (DA) to transfer knowledge.

#### Dataset
- **Source (egocentric)**: EPIC-Kitchens (~1000 videos)
- **Target (exocentric)**: Kinetics subset (~500 similar videos, e.g., "chopping")
- Mapping: select 20–30 common actions

#### Minimum Objectives
1. **Baseline fine-tuning**: train on target, evaluate accuracy
2. **Adversarial DA**: gradient reversal layer
   - Shared encoder (CNN)
   - Classification head (predicts action on target)
   - Domain discriminator (predicts if egocentric=0 or exocentric=1)
   - Backprop: loss_class - λ*loss_domain (adversarial)
3. Metrics: target accuracy, domain discriminator loss
4. Report: does the model manage to confuse the discriminator? Does accuracy improve with DA vs fine-tuning?

#### Extra Objectives
- Maximum Mean Discrepancy (MMD) loss
- Visualization of feature alignment (t-SNE source vs target)
- Per-class analysis: which actions are easy/difficult to adapt?

---

<a id='traccia-8'></a>
### Track 8: Domain Adaptation with Image-to-Image Translation (CycleGAN)
**Difficulty**: Intermediate  
**Module**: Domain Adaptation  
**When to start**: After the Domain Adaptation lecture (16/04/26)

#### Problem Description
Translate images from domain A to domain B without aligned pairs (e.g., sketch → photo). Use the translation as pre-processing to improve the classifier on the target.

#### Dataset
- **Office-31** (source: Amazon, target: DSLR)
- Or **VisDA** (syn → real)
- Alternative: your own data with two natural domains

#### Minimum Objectives
1. **CycleGAN**: two generators (A→B, B→A) and two discriminators
2. Loss: adversarial (discriminator convinces) + cycle-consistency (G_AB(G_BA(x)) ≈ x)
3. Pipeline: 
   - Train CycleGAN to translate source → target
   - Use translated + original images to train classifier
4. Metrics:
   - FID score (Frechet Inception Distance) on translated images
   - Classifier accuracy on target
   - Visual quality (human)

#### Extra Objectives
- Simultaneous DANN: while CycleGAN translates, an adversarial domain discriminator
- Hybrid analysis: when does translation help and when does it not?

---

<a id='traccia-9'></a>
### Track 9: Multi-source Domain Adaptation for Action Recognition
**Difficulty**: Advanced  
**Module**: Domain Adaptation  
**When to start**: After the Domain Adaptation lecture (16/04/26)

#### Problem Description
Instead of a single source domain, use information from 3 different sources to improve on the target.

#### Dataset
- **Source 1**: HMDB-51
- **Source 2**: UCF-101
- **Source 3**: Kinetics subset
- **Target**: action-localization custom or subset of interest

#### Minimum Objectives
1. Model with: shared encoder + 3 source classifiers + target classifier
2. Domain discriminator for each source (or global)
3. Weighted ensemble: assign weight to each source based on similarity with the target
4. Training loop: optimize all simultaneously
5. Metrics: target accuracy, per-source contribution analysis

#### Extra Objectives
- Meta-learning for domain weighting: learn which sources to weight
- Incomplete batch simulation: what happens if a source is missing during training?
- Analogy study: how does performance vary with the number of sources?

---

<a id='traccia-10'></a>
### Track 10: Contrastive Learning for Video Representation (SimCLR Video)
**Difficulty**: Beginner  
**Module**: Self-Supervised Learning  
**When to start**: After the Self-Supervised Learning lecture (21/04/26)

#### Problem Description
Pre-train a video encoder without labels using a contrastive loss on pairs of augmented frames/clips from the same video.

#### Dataset
- **Kinetics-400** (subset ~50k videos) or **UCF-101** if you want to start small
- Unlabeled (labels NOT used for pre-training)

#### Minimum Objectives
1. Video augmentations: spatial crop, temporal sampling, color jitter, rotation
2. Encoder: 3D ResNet mini (e.g., ResNet-18 3D)
3. Projection head: encoder → 128-dim vector
4. Contrastive loss: SimCLR
   - Batch of N videos
   - For each video: two augmentations → two embeddings
   - Loss: maximize similarity of augmentations from the same video, minimize with the rest of the batch
5. Pre-training for K epochs (e.g., 100)
6. **Linear probe**: freeze encoder, train only FC layer on labeled dataset (e.g., 10% HMDB), measure accuracy
7. Metric: compare linear probe accuracy vs supervised training from scratch

#### Extra Objectives
- Temperature in contrastive loss: T=0.1, 0.5, 1.0, effect on convergence
- Visualization: t-SNE embeddings of similar videos should cluster together
- Momentum contrast (MoCo) for larger batch size

---

<a id='traccia-11'></a>
### Track 11: Masked Video Modeling (MAE-style) for Egocentric Video
**Difficulty**: Intermediate  
**Module**: Self-Supervised Learning  
**When to start**: After the Self-Supervised Learning lecture (21/04/26)

#### Problem Description
Mask random frames in a video and train a model to reconstruct them (autoencoder-style). Useful for egocentric because the model learns what happens in the procedural space.

#### Dataset
- **EPIC-Kitchens** (1000 videos ~30 frames each)
- Or your own procedural data

#### Minimum Objectives
1. **Input**: sequence of frames [1, 2, 3, 4, 5] (e.g., 5 frames, 1 every 0.2 sec)
2. **Masking**: mask 50% of the frames (e.g., [X, 2, X, 4, X])
3. **Encoder**: ViT 3D or CNN 3D on unmasked frames
4. **Decoder**: reconstructs masked frames from encoder embeddings
5. **Loss**: MSE between reconstructed and original frames
6. **Evaluation**:
   - Reconstruction MSE/PSNR
   - Linear probe accuracy (downstream task: action classification)
   - Comparison with supervised pre-training

#### Extra Objectives
- Masking strategies: random vs patch-based vs temporal
- Asymmetric decoder (small) for efficiency
- Visualization of reconstructed frames

---

<a id='traccia-12'></a>
### Track 12: Clustering-based Self-Supervised Learning for Action Discovery
**Difficulty**: Intermediate  
**Module**: Self-Supervised Learning  
**When to start**: After the Self-Supervised Learning lecture (21/04/26)

#### Problem Description
Discover recurring actions in **unlabeled** videos using iterative clustering. Useful when you have no annotations but the video has repetitive patterns.

#### Dataset
- Unlabeled procedural videos (e.g., YouTube DIY subset, ~500 videos)
- Or shopping, cooking videos, etc.

#### Minimum Objectives
1. Feature extraction: pre-trained backbone (e.g., CLIP, TimeSformer)
2. Feature pooling per clip (e.g., 30-frame clip → 1 512-dim vector)
3. K-means clustering (start with k=10, then experiment)
4. Pseudo-labels: assign a cluster to each clip
5. Fine-tuning: train a classifier on pseudo-labels
6. Evaluation: 
   - Cluster purity (how many videos in cluster 0 are truly similar?)
   - Downstream task accuracy if you have ground truth labels (optional)
7. Iteration: repeat clustering on fine-tuned embeddings

#### Extra Objectives
- Hierarchical clustering: discover hierarchy (macro-actions vs micro-steps)
- Temporal consistency: clips from the same video should stay in the same cluster

---

<a id='traccia-13'></a>
### Track 13: Temporal Action Localization with 1D CNN
**Difficulty**: Beginner  
**Module**: Video Understanding  
**When to start**: After the Video Understanding lecture (30/04/26)

#### Problem Description
Localize **when** an action happens in the video (find start/end frame). E.g., in a 2-minute video, find that the "chopping" action starts at frame 120 and ends at frame 350.

#### Dataset
- **ActivityNet-1.3** (subset of 20 classes, ~100 videos)
- Pre-extracted video features (C3D/SlowFast)
- Annotations: start/end frames for each action

#### Minimum Objectives
1. Pre-processing: convert video → sequence of features (e.g., features every 0.5 sec)
2. 1D CNN encoder: processes features along the temporal dimension
3. Regression head: predicts (start_frame, end_frame) for each window
4. Loss: MSE + IoU loss
5. Metric: mAP @IoU=0.5 (how many predictions are within 50% of the ground truth?)
6. Report: systematic errors (e.g., always predicts the action too short/long)?

#### Extra Objectives
- Soft-NMS post-processing: merge overlapping detections
- Class-agnostic detection: find action boundaries without knowing the class

---

<a id='traccia-14'></a>
### Track 14: Action Recognition with Vision Transformer (ViT-based)
**Difficulty**: Intermediate  
**Module**: Video Understanding  
**When to start**: After the Video Understanding lecture (30/04/26)

#### Problem Description
Use self-attention (Vision Transformer) to classify actions in videos. The Transformer views the video as a sequence of spatio-temporal patches.

#### Dataset
- **HMDB-51** or **Kinetics-400 subset**
- Video frames or pre-extracted features

#### Minimum Objectives
1. **Patch embedding**: divide video into 16x16x4 patches (space x time), project to 768-dim
2. **Positional encoding**: spatio-temporal positions
3. **Transformer**: stack of attention layers
4. **Classification**: CLS token → FC head → logits for 51 classes
5. **Training**: standard supervised CE loss
6. **Evaluation**: Top-1 accuracy, comparison with 3D CNN baseline
7. Comparative metrics: latency, # params

#### Extra Objectives
- Attention visualization: which patches does the model look at?
- Comparison ViT vs TimeSformer (video-specific version)
- Efficiency: reduce # layers, # attention heads

---

<a id='traccia-15'></a>
### Track 15: Vision-Language Alignment with CLIP for Video
**Difficulty**: Intermediate  
**Module**: Vision & Language  
**When to start**: After the Vision & Language lecture (07/05/26)

#### Problem Description
Align video features with text using a contrastive loss (CLIP style). Allows text queries for videos (e.g., "person chopping vegetables" → find similar videos).

#### Dataset
- **MSR-VTT** (subset ~500 videos with captions)
- Or **COCO-Captions** adapted with video captions

#### Minimum Objectives
1. **Video encoder**: pre-trained (TimeSformer, SlowFast)
2. **Text encoder**: pre-trained (BERT, DistilBERT)
3. **Contrastive loss**: 
   - Batch of N (video, caption) pairs
   - Maximize similarity of correct pairs
   - Minimize similarity of incorrect pairs
4. Training and evaluation:
   - Text-to-video retrieval: given text, find a similar video
   - Metric: R@1, R@5, R@10 (top-K recall)
5. Report: does zero-shot search work?

#### Extra Objectives
- Fine-tuning encoder (vs frozen)
- Zero-shot action recognition: assign a text label to actions, top-5 accuracy
- Failure analysis: which pairs does the model confuse?

---

<a id='traccia-16'></a>
### Track 16: Multimodal Action Recognition – Video + Audio + Text
**Difficulty**: Advanced  
**Module**: Vision & Language  
**When to start**: After the Vision & Language lecture (07/05/26)

#### Problem Description
Classify actions by simultaneously exploiting video, audio, and captions. More modalities = more robustness.

#### Dataset
- Synthetic (create your own videos with audio) or AudioSet + video
- Subset: 10 classes, 100 videos each

#### Minimum Objectives
1. **Video encoder**: 3D CNN
2. **Audio encoder**: 1D CNN on spectrogram (librosa)
3. **Text encoder**: BERT / DistilBERT
4. **Fusion strategy**: embedding concatenation + FC
5. Loss: standard CE
6. Evaluation: 
   - Multimodal accuracy (all 3)
   - Single modality accuracy (for comparison)
   - Contribution analysis: which modality counts the most?

#### Extra Objectives
- Missing modality: robustness when video/audio/text is missing
- Cross-modal attention
- Analysis of late vs early fusion

---

<a id='traccia-17'></a>
### Track 17: Egocentric Video + Gaze for Procedural Understanding
**Difficulty**: Intermediate  
**Module**: Video Understanding  
**When to start**: After the Video Understanding lecture (30/04/26)

#### Problem Description
Combine egocentric video (from a first-person point of view) with gaze tracking (where the person is looking) to understand what they are doing.

#### Dataset
- **EPIC-Kitchens + gaze** (subset with gaze annotations)
- Or **AriaGen2** (smart glasses with IMU, gaze)
- Features: video frames + gaze heatmap

#### Minimum Objectives
1. **Video encoder**: 2D CNN on frames (ResNet-18)
2. **Gaze encoder**: 2D CNN on gaze heatmap (Gaussian blob on gaze point)
3. **Fusion**: embedding concatenation
4. **Classification**: FC head → action
5. Evaluation: 
   - Action accuracy
   - Ablation: video only vs gaze only vs fused
6. Report: does gaze really help? In which actions?

#### Extra Objectives
- Saliency map: where does the model "look" vs where does the person look?
- Attention bottleneck: is gaze a bottleneck for some actions?
- Temporal alignment: synchronize video and gaze

---

<a id='traccia-18'></a>
### Track 18: State-Space Models (Mamba) for Long Sequences
**Difficulty**: Advanced  
**Module**: Advanced Sequential Modeling  
**When to start**: After the Advanced Sequential Modeling lecture (19/05/26)

#### Problem Description
Use State-Space Models (Mamba, Hippo) to model very long sequences (e.g., entire procedural videos of 1+ minutes) without the quadratic cost of attention.

#### Dataset
- **EPIC-Kitchens** long sequences (~30 min sessions, features every 0.5 sec = 3600 steps)
- Subset: 100 videos, 50 actions

#### Minimum Objectives
1. **Baseline LSTM**: standard model on long sequences
2. **Mamba/SSM**: implementation (or use library: mamba-ssm, ssm-lib)
3. Training: same loss (CE for action), same preprocessing
4. Metrics:
   - Perplexity / Accuracy on actions
   - Training time (wall-clock)
   - Memory usage
5. Benchmark: Mamba vs LSTM on long sequences

#### Extra Objectives
- Hippo variant (hyperbolic vs standard Mamba)
- Selective state updates (interpretability)
- Ablation: how does sequence length affect Mamba vs LSTM?

---

<a id='traccia-19'></a>
### Track 19: Transformer vs RNN for Procedural Video Understanding
**Difficulty**: Intermediate  
**Module**: Advanced Sequential Modeling  
**When to start**: After the Advanced Sequential Modeling lecture (19/05/26)

#### Problem Description
Compare Transformer and RNN on procedural step understanding tasks (e.g., Assembly, cooking): which is more effective on procedures?

#### Dataset
- **Assembly101** (subset of 5 procedures, ~50 videos)
- Annotations: step-wise labels, duration of each step

#### Minimum Objectives
1. **LSTM baseline**: encoder on video features
2. **Transformer encoder**: multi-head attention, positional encoding
3. Training: given past history, classify the current step
4. Metrics:
   - Frame-level accuracy (predicts correct step for each frame)
   - Per-class F1 score
   - Latency: Transformer vs LSTM inference

#### Extra Objectives
- Hybrid: Transformer + recurrence layer
- Attention analysis: attention heads specialized for temporal vs contextual?

---

<a id='traccia-20'></a>
### Track 20: Diffusion Models for Trajectory/Motion Generation
**Difficulty**: Intermediate  
**Module**: Advanced Sequential Modeling  
**When to start**: After the Advanced Sequential Modeling lecture (19/05/26)

#### Problem Description
Generate plausible 2D/3D trajectories (people walking) or human motion sequences conditioned by an initial context.

#### Dataset
- **Human3.6M** (skeleton motion capture data, walking/running subset)
- Or synthetic 2D trajectories (people in top-down view)

#### Minimum Objectives
1. **VAE baseline**: compresses trajectory → latent space, then generates
2. **Diffusion model** (DDPM): 
   - Forward: progressively add noise to the trajectory
   - Reverse: network learns to remove noise
   - Sampling: iteratively generate new trajectories
3. Loss: reconstruction MSE
4. Metrics:
   - Frechet distance (between generated and real trajectories)
   - Diversity: generated variance vs real variance
5. Qualitative evaluation: visualize some generated trajectories

#### Extra Objectives
- Conditioning: generate trajectory given initial velocity
- Classifier-free guidance
- Denoising steps analysis

---

<a id='traccia-21'></a>
### Track 21: Deep Q-Learning for Frame Selection in Video
**Difficulty**: Beginner  
**Module**: Reinforcement Learning  
**When to start**: After the Reinforcement Learning lecture (26/05/26)

#### Problem Description
An agent learns to select informative frames from a video to reduce computational cost while maintaining good action classification accuracy. This is useful for compressed video.

#### Dataset
- Video classification task (e.g., HMDB-51 subset, 100 videos)
- Manually or automatically assign an "informativeness" score to each frame (e.g., maximum action divergence from the last selected frame)

#### Minimum Objectives
1. **State**: current frame + history of selected frames
2. **Action space**: {select_frame, skip_frame}
3. **Reward**: +1 if action is correctly classified, -0.01 for selected frame (cost)
4. **Q-network**: CNN + FC that predicts Q(state, action)
5. **DQN**: training loop with experience replay
6. Evaluation:
   - Action accuracy vs # of selected frames
   - Comparison: random selection vs learned selection

#### Extra Objectives
- Double DQN
- Prioritized experience replay

---

<a id='traccia-22'></a>
### Track 22: Policy Gradient for Gesture Control
**Difficulty**: Intermediate  
**Module**: Reinforcement Learning  
**When to start**: After the Reinforcement Learning lecture (26/05/26)

#### Problem Description
Train an agent to control a simple environment (e.g., avatar movement) by interpreting human gestures. The agent receives a reward if it correctly interprets the gesture.

#### Dataset
- Gesture dataset (e.g., MediaPipe skeleton of 5 common gestures) + Gymnasium environment (CartPole or GridWorld)

#### Minimum Objectives
1. **Policy network**: gesture encoder (CNN on skeleton) → action logits
2. **Algorithm**: REINFORCE (vanilla policy gradient)
   - Sample actions from policy
   - Calculate return (cumulative reward)
   - Update policy: gradient ascent on log-prob
3. Evaluation:
   - Cumulative reward over time
   - Convergence speed

#### Extra Objectives
- Actor-Critic (reduces variance)
- Curriculum: start with easy tasks, then difficult ones

---

<a id='traccia-23'></a>
### Track 23: Multi-agent RL for Task Coordination
**Difficulty**: Advanced  
**Module**: Reinforcement Learning  
**When to start**: After the Reinforcement Learning lecture (26/05/26)

#### Problem Description
Two or three agents learn to coordinate parallel procedural tasks (e.g., assembly with two robotic arms, recipe preparation with two people).

#### Dataset
- Synthetic: MultiAgentEnv environment based on Gymnasium
- Define simple tasks (e.g., 5 steps each, agents must coordinate to avoid conflicts)

#### Minimum Objectives
1. **Multi-agent environment**: 2–3 agents, shared reward (coordination) or individual reward (task)
2. **Policy**: separate policy per agent
3. **Communication**: simple (e.g., state sharing) or silent (learn implicitly)
4. Training: train all policies simultaneously
5. Evaluation:
   - Task completion rate (how many times both agents finish the task?)
   - Coordination efficiency

#### Extra Objectives
- Emergent behavior: describe emergent behaviors during training
- Decentralized learning: agents do not have access to the joint state

---

<a id='traccia-24'></a>
### Track 24: Differentiable Task Graphs (Yao method) – Group A
**Difficulty**: Advanced  
**Module**: Research Topic (Graphs/Procedural)  
**When to start**: Mid-course, after Domain Adaptation

#### Problem Description
Reproduce the method by Yao et al. (2022) "Differentiable Task Graphs for Procedure Understanding". Idea: each procedure is a Directed Acyclic Graph (DAG) of steps; the model learns feasibility (which step can follow which) and predicts the sequence of steps.

#### Dataset
- **Assembly101** (subset of 5 complete procedures, ~50 videos)
- Each video has step-wise annotations (frame→step)

#### Minimum Objectives
1. **Graph construction**: extract task graph from ground truth annotations (nodes = unique steps, edges = observed transitions)
2. **Task graph encoder**: GNN (GCN) that maps task graph → feasibility embedding
3. **Prediction**: given video + graph, predict the next step
4. **Metric**: mAP on predicted step sequence vs ground truth
5. **Baseline**: simple classifier (ignores graph) for comparison
6. **Ablation**: remove graph components (edges, nodes), impact on performance

#### Extra Objectives
- Error analysis: when is the predicted feasibility wrong?
- Visualization of predicted graph vs ground truth

#### References
- Paper: Yao et al. "Differentiable Task Graphs for Procedure Understanding" (ICCV 2023)
- GitHub: official if available

---

<a id='traccia-25'></a>
### Track 25: Task Graphs – Softmax vs Sum Feasibility – Group B
**Difficulty**: Advanced  
**Module**: Research Topic (Graphs/Procedural)  
**When to start**: Mid-course, after Domain Adaptation

#### Problem Description
**Extension of Track 24**: Yao uses sum in feasibility aggregation; experiment with softmax and compare.

#### Dataset
- Same as Track 24: Assembly101

#### Minimum Objectives
1. **Baseline Yao**: reproduce standard method (sum aggregation)
2. **Softmax aggregation**: replace sum with softmax in feasibility prediction
3. **Constrained optimization**: add constraints (e.g., each step must have ≤K predecessors)
4. **Comparison**: mAP, convergence analysis, interpretability
5. **Ablation**: which modifications help/worsen?

#### Extra Objectives
- Softer constraints (probabilistic vs hard)

---

<a id='traccia-26'></a>
### Track 26: Procedural Error Detection with Gaze – Group A
**Difficulty**: Intermediate  
**Module**: Research Topic (Egocentric/Multimodal)  
**When to start**: After Video Understanding (late April)

#### Problem Description
An operator follows a procedure (cooking, assembly) recorded by an egocentric camera. The model detects if they are making a mistake by combining video + gaze (where they are looking).

#### Dataset
- **EPIC-Kitchens subset** (~300 videos) with manual "correct/mistake" annotations per step
- Or create your own mini procedures labeled correct/error

#### Minimum Objectives
1. **Video feature**: TSN or pre-extracted
2. **Gaze feature**: gaze heatmap (Gaussian on tracked point)
3. **Fusion**: embedding concatenation
4. **Binary classifier**: correct vs mistake
5. **Evaluation**: F1-score, confusion matrix, per-class analysis
6. **Ablation**: video only vs gaze only vs fused

#### Extra Objectives
- Saliency map: where does the model look when predicting an error?
- Temporal localization: not just predicting the error, but where in the step?

---

<a id='traccia-27'></a>
### Track 27: Error Detection – Progress-Aware Model – Group B
**Difficulty**: Intermediate  
**Module**: Research Topic (Egocentric/Multimodal)  
**When to start**: After Video Understanding (late April)

#### Problem Description
**Extension of Track 26**: Add an auxiliary "progress" prediction task (which step of the procedure are we in?) to improve error detection.

#### Dataset
- Same as Track 26

#### Minimum Objectives
1. **Baseline**: standard mistake detection
2. **Progress predictor**: side task that predicts what the current step is (multitask setup)
3. **Joint loss**: L = L_mistake + λ * L_progress
4. **Comparison**: F1-score, ablation on λ
5. **Analysis**: does progress help mistake detection?

#### Extra Objectives
- Adversarial loss: force progress embedding to be independent from mistake prediction

---

<a id='traccia-28'></a>
### Track 28: Graph Autoencoder for Geometric Representations
**Difficulty**: Advanced  
**Module**: Research Topic (Graphs/Representation)  
**When to start**: After Metric Learning

#### Problem Description
Graph autoencoder to learn robust geometric embeddings. Applicable to scene graph retrieval, graph clustering, etc.

#### Dataset
- **Visual Genome** (scene graphs)
- Subset: ~500 graphs with 5–15 nodes

#### Minimum Objectives
1. **Graph encoder**: GCN/GraphSAGE mapping graph → embedding
2. **Autoencoder decoder**: MLP decoding embedding → node and edge predictions
3. **Loss**: binary cross-entropy for edge reconstruction + node classification
4. **Evaluation**:
   - Link prediction accuracy
   - Clustering in the latent space (do similar scenes cluster together?)
   - "Average" graph via latent interpolation
5. **Ablation**: encoder size, decoder depth, etc.

#### Extra Objectives
- VAE variant (Gaussian distribution latent)
- Hyperbolic latent space (next track)

---

<a id='traccia-29'></a>
### Track 29: Hyperbolic Embeddings for Action Hierarchy
**Difficulty**: Intermediate  
**Module**: Research Topic (Advanced Representations)  
**When to start**: After Metric Learning

#### Problem Description
Actions often have a hierarchical structure (macro-actions contain micro-actions). Hyperbolic space preserves hierarchies better than Euclidean space.

#### Dataset
- **ActivityNet**: use the provided hierarchies (event categories)
- Subset: ~20 classes, 3-level tree

#### Minimum Objectives
1. **Euclidean baseline**: standard embedding on Euclidean space
2. **Poincare embeddings**: hyperbolic (library: geoopt)
3. **Task**: link prediction (given partial hierarchy, predict missing links)
4. **Metric**: 
   - Distortion (how well does it preserve hierarchy vs euclidean?)
   - Link prediction accuracy
5. **Visualization**: hyperboloid projected into Poincare disk

#### Extra Objectives
- Tuning hyperbolic curvature (negative K)
- Mixed curvature spaces

---

<a id='traccia-30'></a>
### Track 30: Generative Models for Data Augmentation in Egocentric Domain
**Difficulty**: Intermediate  
**Module**: Research Topic (Egocentric/Generative)  
**When to start**: After Self-Supervised Learning

#### Problem Description
EPIC-Kitchens is the largest egocentric dataset, but sometimes specific actions are rare. Generate realistic synthetic frames to augment the dataset.

#### Dataset
- **EPIC-Kitchens** (1000 videos, 50 actions)
- Focus on rare actions (e.g., "cutting" has 100 samples, "greeting" has 10)

#### Minimum Objectives
1. **VAE baseline**: compresses frame → latent, generates new ones
2. **Diffusion model**: noisy→clean denoising to generate frames
3. **Evaluation**:
   - FID score (generated frame quality)
   - Downstream action recognition: training on original+synthetic vs original only
4. **Ablation**: which generative model works best for egocentric?

#### Extra Objectives
- Conditioning: generate frame given action label
- Feature diversity: do generated frames add new visual variations?

---

<a id='traccia-31'></a>
### Track 31: Online Episodic Memory for Action Anticipation
**Difficulty**: Advanced  
**Module**: Research Topic (Memory/Anticipation)  
**When to start**: After Advanced Sequential Modeling

#### Problem Description
Humans anticipate future actions by remembering similar past situations. An episodal memory module updated online can improve anticipation.

#### Dataset
- **EPIC-Kitchens** procedural sequences (~500 videos)
- Task: given a video chunk, predict future actions 1–5 seconds ahead

#### Minimum Objectives
1. **LSTM baseline**: anticipation without memory
2. **Memory bank**: VQ-inspired quantization of queries/values
   - Query: current video feature
   - Store: past features + observed future actions
3. **Online update**: update memory during rolling window (not static)
4. **Evaluation**:
   - Top-5 accuracy anticipation at different horizons
   - Memory efficiency (size, retrieval time)
5. **Ablation**: impact of memory size, temporal decay

#### Extra Objectives
- Query learning (network that learns which memory to retrieve)
- Temporal decay (older memories are less relevant)

---


## Groups

| Group Name | Members |
| :--- | :---: |
| LeMeCla | 3 |
| BAT 🦇 (Backpropagation Attention Team) | 3 |
| Deep Team | 3 |
| Justgood AI | 3 |
| FiCo | 3 |
| FlyNow | 3 |
| Overfittony | 3 |
| The Outliers 2.0 | 3 |
| Zero e Uno | 2 |
| DataMinds | 2 |
| TEAM CassiaBranca | 2 |
| Le larunghie | 2 |
| DataLost | 2 |
| EventHorizonTeam | 2 |
| Marte | 2 |
| G16 | 1 |
| G17 | 1 |
| G18 | 1 |
| G19 | 1 |
| G20 | 1 |
| G21 | 1 |
| G22 | 1 |
| G23 | 1 |
| G24 | 1 |
| G25 | 1 |
| G26 | 1 |
| G27 | 1 |
| G28 | 1 |
| G29 | 1 |
| G30 | 1 |
| G31 | 1 |
