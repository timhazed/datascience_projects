# Data Science & AI Systems Engineering Portfolio

**Tim Hayes**

I design and build production-grade AI systems — multi-agent pipelines, RAG architectures, deep learning models, and the data infrastructure that supports them — applying a consistent engineering methodology from exploration through deployment. What distinguishes this work is not the breadth of projects, but the rigor behind each one: failure analysis before implementation, structured evaluation, and systems designed to be extended and handed off.

Core focus areas include:

* Generative AI and Agentic Systems
* Retrieval-Augmented Generation (RAG)
* Multi-Agent Architectures
* End-to-End Data Pipelines
* Deep Learning and Computer Vision

---

## Methodology

A consistent engineering lifecycle is applied across every project in this repository:

**Exploration → Architecture → Critique → Implementation → Review → Evaluation → Refactor**

The critique phase — conducted before any code is written — is where most engineering failures are preventable and where most teams skip ahead. Structured evaluation closes the loop: it measures whether the system actually does what it claims to do under realistic conditions, not just whether the tests pass.

See [methodology/](methodology/) for the full engineering playbook: lifecycle phases, design patterns, principles, and the decision framework for selecting the right architecture.

---

## Highlighted Systems

### Agentic AI Systems

* [Agentic Healthcare Assistant](projects/agentic%20healthcare%20assistant/)
* [NewsGenie](projects/newsgenie/)
* [Exercise and Recovery Coach](projects/exercise_and_health_coach/)
* [Agentic Developer Memory](projects/developer%20memory/)
* [LLM Framework Benchmarking](projects/LLM%20Framework%20Benchmarking/)

### Retrieval-Augmented Generation (RAG)

* [HR Assistant](projects/hr_assistant/)
* [Chatbot DataPipeline](projects/Chatbot_DataPipeline/)

### Deep Learning & Computer Vision

* [3D Roof Reconstruction CNN](projects/3D%20Roof%20Reconstruction%20CNN/)
* [Aging VAE](projects/Aging%20VAE/)

### Full-Stack AI Applications

* [GeoPortugal](projects/Geoportugal/)
* [Image Designer Assistant](projects/image_designer_assistant/)

For the complete project catalog with full descriptions, skills, and tools, see [PROJECTS.md](PROJECTS.md).

---

# Projects

---

## Deep Learning & Computer Vision

### 3D Roof Reconstruction CNN

This application leverages deep learning models to predict azimuth, tilt, height, and perimeter of planes from a roof using aerial images and point cloud data.

**Approach:** Multi-input CNN (ResNet50 / EfficientNet) with multi-task outputs.

**Key Capability:** Fusion of aerial imagery and point cloud data for geometric prediction.

**Non-Augmented Models**
![Non-Augmented ResNet50 Training and Validation Loss](projects/3D%20Roof%20Reconstruction%20CNN/images/ResNet50_NonAugmented.png)
![Non-Augmented EfficientNet Training and Validation Loss](projects/3D%20Roof%20Reconstruction%20CNN/images/EfficientNet_NonAugmented.png)

**Augmented Models**
![Augmented ResNet50 Training and Validation Loss](projects/3D%20Roof%20Reconstruction%20CNN/images/ResNet50_Augmented.png)
![Augmented EfficientNet Training and Validation Loss](projects/3D%20Roof%20Reconstruction%20CNN/images/EfficientNet_Augmented.png)

🔗 [View Project](projects/3D%20Roof%20Reconstruction%20CNN/)

**Skills and Tools:** TensorFlow, Keras, Pandas, Matplotlib, Sklearn

---

### Aging VAE

A Variational Autoencoder-based system for facial age progression using latent space manipulation.

![Aging VAE Neutral Layout](projects/Aging%20VAE/images/VAE_Image_Neutral.png)
![Aging VAE Plus 30 Aged Layout](projects/Aging%20VAE/images/VAE_Image_Plus_30.png)

🔗 [View Project](projects/Aging%20VAE/)

**Skills Used**
Backend: TensorFlow, VAE, NumPy, OpenCV
Frontend: Gradio

---

### SAM v2 Image Segmentation

Segment Anything Model (SAM)-based segmentation system for aerial imagery.

![SAM v2 Image Segmentation](projects/SAM%20v2%20image%20segmentation/images/samv2_segmentation.png)

🔗 [View Project](projects/SAM%20v2%20image%20segmentation/)

---

### Point Cloud Visualizer

PyQt-based application for 3D point cloud visualization.

![Point Cloud Surfaces](projects/Point%20Cloud%20Visualizer/images/point_cloud.jpg)

🔗 [View Project](projects/Point%20Cloud%20Visualizer/)

---

### Roof Fusion

3D roof reconstruction preprocessing and visualization system.

![Aerial Image with Corners](projects/roof%20fusion/images/aerial_with_corners.png)
![Aerial 3D Image with Normals](projects/roof%20fusion/images/aerial_3d_with_normals.jpg)

🔗 [View Project](projects/roof%20fusion/)

---

## Generative AI & Agentic Systems

### Agentic Healthcare Assistant

A LangGraph-orchestrated clinical assistant that accepts natural-language queries, decomposes them into a sequential plan, and dispatches each sub-task across five specialized tool nodes (patient resolution, history retrieval, appointment booking, and disease research), returning a single synthesized response from one conversational interface.

![Agentic Healthcare Assistant Architecture](projects/agentic%20healthcare%20assistant/images/langgraph_architecture.png)
![Agentic Healthcare Assistant What drug](projects/agentic%20healthcare%20assistant/images/What_drug_am_I_taking.png)


🔗 [View Project](projects/agentic%20healthcare%20assistant/)

**Skills Used**
LangGraph, LangChain, LangGraph Checkpoint (SqliteSaver), OpenAI, Groq, Pydantic, FAISS, ragas, pypdf, SQLite, Poetry, Streamlit

### NewsGenie

LangGraph-based multi-agent news assistant that decomposes free-form queries into parallel domain fetches (Business, Sports, World) or web search, validated across four LLM configurations.

![NewsGenie Base UI](projects/newsgenie/images/base_ui.png)
![NewsGenie Sports](projects/newsgenie/images/latest_sports.png)

🔗 [View Project](projects/newsgenie/)

**Skills Used**
LangGraph, OpenAI, Groq, Pydantic, NewsAPI, Guardian API, ESPN Scoreboard, Streamlit

---

### Exercise and Recovery Coach

Multi-agent system generating personalized, safety-audited workout and recovery plans.

![Exercise and Health Coach Workout Plan](projects/exercise_and_health_coach/images/exercise_and_health_coach_workout_plan.png)

🔗 [View Project](projects/exercise_and_health_coach/)

**Skills Used**
LangChain, OpenAI, Pydantic, Gradio

---

### Agentic Developer Memory
A dynamic context isolation and state serialization engine engineered to mitigate token accumulation, eliminate message redundancy, and optimize long-term persistence within continuous AI software engineering workflows by decoupling transient execution loops from absolute system state transitions.

![Developer Memory Architecture](projects/developer%20memory/images/langgraph_architecture.png)
![Developer Memory Query Langgraph UI](projects/developer%20memory/images/QueryMemoryLanggraphUI.png)

🔗 [View Project](projects/developer%20memory/)

Skills Used
Python, LangChain, OpenAI, Groq, Pydantic (Structured Outputs), Tenacity, Tokenizer Optimization, JSON/File Serialization, Poetry, Streamlit, pytest, pytest-cov, Ruff

---

### Image Designer Assistant

Conversational multi-turn image generation system using LangChain and DALL-E.

![Image Assistant Image and Refinement](projects/image_designer_assistant/images/image_assistant_image_and_refinement.png)

🔗 [View Project](projects/exercise_and_health_coach/)

Skills Used
Backend: LangChain, LangChain-OpenAI, OpenAI (GPT-4o-mini), Pydantic, PyYAML, python-dotenv

Frontend: Gradio, CLI

Testing: pytest, pytest-cov, pytest-mock, Ruff

---

### LLM Framework Benchmarking

Framework benchmarking system comparing LangGraph, CrewAI, and AutoGen.

![Benchmark Scoring Results](projects/LLM%20Framework%20Benchmarking/benchmark_manager/final_results/benchmark_manager/benchmark_scoring_results_20250309_175727.png)
![Benchmark Leaderboard Results](projects/LLM%20Framework%20Benchmarking/benchmark_manager/final_results/benchmark_manager/benchmark_results_leaderboard_20250309_175727.png)

🔗 [View Autogen Crop Yield Benchmark](projects/LLM%20Framework%20Benchmarking/autogen_crop_yield_simple_agent/)
🔗 [View Autogen MultiAgent Benchmark](projects/LLM%20Framework%20Benchmarking/autogen_multi_agent/)
🔗 [View CrewAI Crop Yield Benchmark](projects/LLM%20Framework%20Benchmarking/crewai_crop_yield_simple_agent/)
🔗 [View CrewAI MultiAgent Benchmark](projects/LLM%20Framework%20Benchmarking/crewai_multi_agent/)
🔗 [View LangGraph Crop Yield Benchmark](projects/LLM%20Framework%20Benchmarking/langgraph_crop_yield_simple_agent/)
🔗 [View LangGraph MultiAgent Benchmark](projects/LLM%20Framework%20Benchmarking/langgraph_multi_agent/)

---

## Retrieval-Augmented Generation (RAG) & Data Systems

### Chatbot DataPipeline

Modular ETL pipeline for transforming unstructured documents into semantically searchable data.

🔗 [Data Pipeline](projects/Chatbot_DataPipeline/data_pipeline/)
🔗 [Extract](projects/Chatbot_DataPipeline/extract_and_normalize/)
🔗 [Chunker](projects/Chatbot_DataPipeline/chunks/)
🔗 [Insert DB](projects/Chatbot_DataPipeline/insert_db/)

**Skills and Tools:** Prefect, Docker, Qdrant, Sentence Transformers

---

### HR Assistant

RAG-based chatbot with intent guardrails and evidence-backed responses.

![HR Assistant Basic Question](projects/hr_assistant/images/HRAssistant_BasicQuestions.png)
![HR Assistant Question with Evidence](projects/hr_assistant/images/HRAssistant_QuestionWithEvidence.png)

🔗 [View Project](projects/hr_assistant/)

---

## Full-Stack AI Applications

### GeoPortugal

Geospatial full-stack application for exploring Portugal's administrative regions.

![Geoportugal Layout](projects/Geoportugal/images/GeoPortugal_Frontend.png)

🔗 [View Project](projects/Geoportugal)

---

## Machine Learning & Data Science Foundations

### CNN - Dogs and Cats

![Dogs and Cats Best Model](projects/CNN/Dogs%20and%20Cats/images/inception_results.jpg)

🔗 [View Project](projects/CNN/Dogs%20and%20Cats)

---

### CNN - Seedling Classification

![Seedlings Best Model](projects/CNN/Seedlings%20Classification/images/inception_results.jpg)

🔗 [View Project](projects/CNN/Seedlings%20Classification)

---

### Chatbot

![Chatbot with Blenderbot](projects/Chatbot/images/chatbot_blenderbot.jpg)

🔗 [View Project](projects/Chatbot/)

---

### Ensemble Techniques

![Easy Visa Best Model](projects/Ensemble%20Techniques/images/stacking_results.jpg)

🔗 [View Project](projects/Ensemble%20Techniques/)

---

### GenAI - Aspect Sentiment Analysis

![Expressway Logistics Final Results](projects/GenAI%20-%20Aspect%20Sentiment%20Analysis/images/final_results.jpg)

🔗 [View Project](projects/GenAI%20-%20Aspect%20Sentiment%20Analysis/)

---

### GenAI - Sentiment Analysis

🔗 [View Project](projects/GenAI%20-%20Sentiment%20Analysis/)

---

### Hypothesis Testing

![ENews Express Testing Approach](projects/Hypothesis%20Testing/images/testing_approach.jpg)

🔗 [View Project](projects/Hypothesis%20Testing/)

---

### Image Captioning

![Mountain Image With Caption](projects/Image%20Captioning/images/image_with_caption.jpg)

🔗 [View Project](projects/Image%20Captioning/)

---

### LangChain - Aspect Sentiment Analysis

![Results after three iterations](projects//Langchain%20-%20Aspect%20Sentiment%20Analysis/images/Results_ThreeIterations.png)
![Results after ten iterations](projects/Langchain%20-%20Aspect%20Sentiment%20Analysis/images/Results_TenIterations.png)

🔗 [View Project](projects/Langchain%20-%20Aspect%20Sentiment%20Analysis/)

---

### Linear Regression

![ReCell Best Fit](projects/Linear%20Regression/images/best_fit.jpg)

🔗 [View Project](projects/Linear%20Regression/)

---

### Logistic Regression and Decision Trees

![INN Hotels Precision-Recall](projects/Logistic%20Regression%20and%20Decision%20Trees/images/lg_4197_precision_recall.jpg)
![INN Hotels Classification](projects/Logistic%20Regression%20and%20Decision%20Trees/images/lg_4197_classification.jpg)

🔗 [View Project](projects/Logistic%20Regression%20and%20Decision%20Trees/)

---

### NLP

![Twitter US Airline Best Model](projects/NLP/images/lstm_model.jpg)

🔗 [View Project](projects/NLP/)

---

### Neural Networks

![INN Bank Churn Best Model](projects/Neural%20Networks/images/model4.jpg)
![INN Bank Churn Shap Values](projects/Neural%20Networks/images/model4_shap.jpg)

🔗 [View Project](projects/Neural%20Networks/)

---

### Pandas and Visualization

![Food Hub EDA](projects/Pandas%20and%20Visualization/images/eda.jpg)

🔗 [View Project](projects/Pandas%20and%20Visualization/)

---

### Pdf Summarizer

![Pdf Summarizer](projects/Pdf%20Summarizer/images/summarizer.jpg)

🔗 [View Project](projects/Pdf%20Summarizer/)

---

### Pipelining and Hypertuning

![ReneWind Best Model](projects/Pipelining%20and%20Hypertuning/images/tuned_xgb_under.jpg)

🔗 [View Project](projects/Pipelining%20and%20Hypertuning/)

---

### Unsupervised Learning

![Trade Ahead Best Clustering](projects/Unsupervised%20Learning/images/tsne_scatter_plot.jpg)

🔗 [View Project](projects/Unsupervised%20Learning/)

---

### Voice Assistant

![Voice Assistant](projects/Voice%20Assistant/images/voice_assistant.jpg)

🔗 [View Project](projects/Voice%20Assistant/)

---

## Contact

I am open to selective consulting, advisory roles, and technically challenging engagements.

* **Email:** [thayes@oldzinsoftware.com](mailto:thayes@oldzinsoftware.com)
* **Website:** [www.oldzinsoftware.com](https://www.oldzinsoftware.com)
* **LinkedIn:** [linkedin.com/in/tim-hayes-b26103](https://www.linkedin.com/in/tim-hayes-b26103/)
* **GitHub:** [github.com/timhazed](https://github.com/timhazed)

