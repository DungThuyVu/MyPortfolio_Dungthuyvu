
# Data Science Portfolio

Welcome to my "Data Science portfolio", showcasing applied AI and data-driven projects spanning deep learning, NLP, generative AI, optimization, and IoT systems.  
Each project emphasizes "clean, reproducible code" and demonstrates experience with "real-world datasets, large language models, and applied machine learning".

---

##  Overview of Projects

| # | Project Title | Description | Key Techniques |
|---|----------------|--------------|----------------|
| 1 | Heart Attack Prediction System | Predicts cardiovascular risk from clinical data with interpretable models and feature importance analysis. | Logistic Regression, XGBoost |
| 2 | CNN-Based Plant Disease Detection | Deep learning for image-based identification of crop diseases to support precision agriculture. | CNN, TensorFlow, Keras, Image Augmentation |
| 3 | Sentiment Analysis with Transformers and LLMs | Sentiment classification and contextual text understanding using transformer-based models. | BERT, Llama, Hugging Face Transformers |
| 4 | Evolutionary AI for Robotics and Optimization | Hybrid swarm intelligence algorithms applied to robotic motion, portfolio optimization, and vehicle routing. | Particle Swarm Optimization, Ant Colony Optimization, Genetic Algorithms |
| 5 | Career Coaching AI and Job Market Analytics | Personal career advisor using job-posting data, YOLO-based logo recognition, and large language models for career path recommendations. | Web Scraping, NLP, LLMs, YOLOv8 |
| 6 | 3D Scene Reconstruction via Neural Radiance Fields (NeRF) | Neural rendering pipeline for 3D scene reconstruction from multi-view images. | PyTorch3D, NeRF, Computer Vision |
| 7 | IoT Weather Station in Oslo | Real-time weather monitoring and predictive analytics using streaming sensor data. | Arduino, Time Series Forecasting |
| 8 | AI Image Generation with ControlNet and IP-Adapter | Fine-grained image generation and style transfer using advanced diffusion control models. | ControlNet, Stable Diffusion, IP-Adapter |
| 9 | Context-Aware Banner Insertion for Sports Videos| AI-driven framework for dynamic, non-intrusive sponsorship banner placement in user-generated sports videos. | SlowFast R50, YOLOv8, RAFT-Small, SAM, Diffusion Models |

---

## Project 9 in collaboration to plartform Chall

### Activity-Specific Banner Insertion for Sports Video Platforms
Collaboration: [Chall.no](https://www.chall.no/) — a Norwegian digital platform for community-driven sports video challenges and sponsorship engagement.  
Status:🔒 Private due to GDPR compliance and data confidentiality from user-generated content on the Chall platform.

---

###  Objective
This project addresses the challenge of "automatically and contextually integrating sponsorship banners# into user-generated sports videos, leveraging efficiency for digital marketing.
Traditional static advertisement placement often obstructs visual content or fails to match the video’s context, degrading both viewer experience and marketing impact.

---

### Proposed Framework
A novel "AI-driven pipeline" was developed to achieve adaptive, context-aware banner integration:
1. Activity Recognition:  
   - Sport classification using "SlowFast R50" architecture on Kinetics-400 and Chall datasets.  
2. Context-Aware Banner Generation: 
   - Utilized text-to-image diffusion models — FLUX.1-schnell, HiDream, and DeepFloyd — to synthesize visually coherent sponsor banners.  
3. Dynamic Placement and Tracking:  
   - Employed YOLOv8 for object detection, DeepSORT for tracking, and RAFT-Small for optical flow estimation.  
   - Integrated Segment Anything Model (SAM) for saliency-based, non-intrusive placement aligned with motion and depth.

---

###  Evaluation
- Datasets: Kinetics-400 benchmark + sports videos from the Chall platform (skiing, skating, running).  
- Performance Highlights:  
  - High classification accuracy for visually distinct sports.  
  - FLUX.1-schnell produced the most semantically coherent banners.  
  - SAM-based placement yielded the most visually and contextually aligned advertisement integration.  

---

###  Research Impact
This work bridges the gap between computer vision and digital advertising, enabling adaptive, viewer-aware ad placement that respects video integrity and user experience.  
The methodology supports future deployment on Chall’s commercial video infrastructure and other UGC (user-generated content) platforms.

---

## Repository Structure
MyPortfolio_Dungthuyvu/
│

├── Project1_HeartAttackPrediction/

├── Project2_CNNPlantDisease/

├── Project3_SentimentAnalysis_LLMs/

├── Project4_EvolutionaryAI_Optimization/

├── Project5_CareerCoaching_JobAnalytics/

├── Project6_3DSceneReconstruction_NeRF/

├── Project7_IoTWeatherStation_Oslo/

├── Project8_ImageGeneration_ControlNet/

└── Project9_BannerInsertion_Thesis/ 🔒 (Private)( can be asked for details via email Jorundungthuyvu@gmail.com

---

##  About the Author

Dung Thuy Vu  
🎓 M.Sc. in Applied Computer Science - Data Science -  | AI Enthusiast | Applied Machine Learning enthusiast  
🎓 Bachelor in Data in business BI Norwegian Business School - Investment Analyst- Taxation policy analyst
📍 Norway  
🔗 [LinkedIn](https://www.linkedin.com/in/dungt-vu-108878385/
• [Email](mailto:Jorundungthuyvu@gmail.com)

---

##  License
This repository is for academic and professional showcase purposes.  
All code and content are provided for educational and research use only.  
Project 9 remains private due to GDPR and partnership data confidentiality.

---

⭐ *If you find these projects inspiring, feel free to star this repository or connect on LinkedIn! I am also open to opportunities for collaboration on personal projects in AI and data science, or looking to find team members for Kaggle  Grand Master competitions*




