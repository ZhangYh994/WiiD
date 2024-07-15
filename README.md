<p align="center">
<h1 align="center"><strong>[ECCV2024] Watching it in Dark: A Target-aware Representation Learning Framework for High-Level Vision Tasks in Low Illumination</strong></h1>
  <p align="center">
        Yunan Li&emsp;
        Yihao Zhang&emsp;
        Shoude Li&emsp;
        Long Tian&emsp;
        Dou Quan&emsp;
        Chaoneng Li&emsp;
        Qiguang Miao&emsp;
    <br>
    <em>Xidian University; Xi'an Key Laboratory of Big Data and Intelligent Vision</em>
    <br>
  </p>
</p>


<div id="top" align="center">
This is the official implementaion of paper ***Watching it in Dark: A Target-aware Representation Learning Framework for High-Level Vision Tasks in Low Illumination***, which is accepted in ***ECCV 2024***. In this paper, we propose a target-aware representation learning framework designed to improve high-level task performance in low-illumination environments. We achieve a bi-directional domain alignment from both image appearance and semantic features to bridge data across different illumination conditions. To concentrate more effectively on the target, we design a target highlighting strategy, incorporated with the saliency mechanism and Temporal Gaussian Mixture Model to emphasize the location and movement of task-relevant targets. We also design a mask token-based representation learning scheme to learn a more robust target-aware feature. Our framework ensures compact and effective feature representation for high-level vision tasks in low-lit settings. Extensive experiments conducted on CODaN, ExDark, and ARID datasets validate the effectiveness of our approach for a variety of image and video-based tasks, including classification, detection, and action recognition.

<div style="text-align: center;">
    <img src="assets/principle.png" alt="Dialogue_Teaser" width=100% >
</div>


## 👀TODO

- \[x\] First Release.
- \[ \] Release Code of Image Classification.
- \[ \] Release Code of Object Detection.
- \[ \] Release Code of Action Recognition.





## 🌏 Pipeline of WiiD

<p align="center">
  <img src="assets/pipeline.png" align="center" width="100%">
</p>



## 📚 Dataset

| Data file name                                               |
| ------------------------------------------------------------ |
| [Common Objects Day and Night (CODaN)](https://github.com/Attila94/CODaN) |
| [Exclusively Dark Image Dataset (ExDark)](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) |
| [normal light data of action recognition(CVPR'22 UG2 challenge)](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json) |
| [low light data of action recognition(ARID dataset)](https://drive.google.com/file/d/10sitw9Mi9Gv1jMfyMwbv78EZSpW_lKEx/view) |





## 🐒 Model Zoo

will release





## 💻 Code

coming soon



