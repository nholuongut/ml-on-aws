![](https://i.imgur.com/waxVImv.png)
### [View all Roadmaps](https://github.com/nholuongut/all-roadmaps) &nbsp;&middot;&nbsp; [Best Practices](https://github.com/nholuongut/all-roadmaps/blob/main/public/best-practices/) &nbsp;&middot;&nbsp; [Questions](https://www.linkedin.com/in/nholuong/)
<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=24&pause=1000&color=F7931E&width=435&lines=Hello%2C+I'm+Nho+Luong🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻🇳🇻)](https://git.io/typing-svg)

# **About Me🇻🇳**
- ✍️ Blogger
- ⚽ Football Player
- ♾️ DevOps Engineer
- ⭐ Open-source Contributor
- 😄 Pronouns: Mr. Nho Luong
- 📚 Lifelong Learner | Always exploring something new
- 📫 How to reach me: luongutnho@hotmail.com

![GitHub Grade](https://img.shields.io/badge/GitHub%20Grade-A%2B-brightgreen?style=for-the-badge&logo=github)
<p align="left"> <img src="https://komarev.com/ghpvc/?username=amanpathak-devops&label=Profile%20views&color=0e75b6&style=flat" alt="amanpathak-devops" /> </p>

# How to run a Neural Network(RNN) training pipeline on Airflow and deploy the AI model to AWS ECS for Inference

In today’s fast-paced world, artificial intelligence (AI) has become an indispensable tool for solving complex problems and making informed decisions. As businesses strive to harness the power of AI, it’s crucial to have a robust and efficient pipeline for training AI models. Also, it's all the more important now for developers to learn about AI and AI models. So I started delving into different AI topics and tried training and deploying my own AI models to AWS.

In this blog post, we will explore the step-by-step process of setting up an AI training pipeline using **Airflow** and deploying the trained model on **AWS ECS** for efficient and cost-effective inference.

> 📂 GitHub Repo: [Link Here]  
> 📘 Jupyter Notebook and sample files: [Huggingface Repo Here]  
> Use the scripts in the repo to set up your own process.

---

## ✨ Pre-Requisites

Before I start the walkthrough, here are some good-to-have pre-requisites:

- Basic knowledge of **Terraform & AWS**
- **GitHub account** to follow GitHub Actions
- **AWS account**
- **Terraform installed**
- **Terraform Cloud Free account** ([Guide Here](https://developer.hashicorp.com/terraform/tutorials/cloud-get-started))
- Some basic **AI/Machine Learning** understanding

---

## 💡 What is Airflow?

Airflow is an open-source platform to programmatically **orchestrate and manage complex workflows**. It uses **DAGs (Directed Acyclic Graphs)** to schedule tasks in the correct sequence.

Airflow makes it easy to coordinate steps like:

- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation

---

## 🌐 What is AWS ECS?

**AWS ECS (Elastic Container Service)** is a fully managed container orchestration service.

It supports:
- **ECS-EC2**: You manage EC2 instances.
- **ECS-Fargate**: AWS manages infrastructure (serverless).

We will use **Fargate** mode to deploy the trained model as a REST API.

---

## 🤖 Overview of the Model and the Training Process

We’ll train a **Sentiment Analysis model** using an **LSTM (RNN)** to classify the sentiment of text inputs.

Example input to inference API:
```json
{
  "text": "I am Nho Luong"
}
```

# Frameworks:

TensorFlow (model training)

FastAPI (serving inference)


# 🏗️ Training Process

Load Dataset
```json
df = pd.read_csv('data.csv')
df.drop(['id','label'], axis=1, inplace=True)
df = df.head(500)  # Sample
```

# Train/Test Split
```json
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label_text"], test_size=0.2, random_state=10)
texts_train = list(X_train)
labels_train = list(y_train)
```

# Preprocessing
```json
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

patterns, tags = [], []
for text in texts_train:
    pattern = text.lower()
    pattern = re.sub(r"[^a-z0-9]", " ", pattern)
    tokens = nltk.word_tokenize(pattern)
    tokens = [stemmer.stem(lemmatizer.lemmatize(t)) for t in tokens if t not in stop_words]
    patterns.append(" ".join(tokens))
    tags.append(labels_train[texts_train.index(text)])
```

# Tokenization
```json
tokenizer = Tokenizer(num_words=..., oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, maxlen=..., padding='post')
```

# Label Encoding
```json
encoder = LabelEncoder()
encoder.fit(tags)
encoded_y = encoder.transform(tags)
output_encoded = tf.keras.utils.to_categorical(encoded_y)
```

# Model Definition
```json
model = Sequential([
    Embedding(input_dim=..., output_dim=32, input_length=..., mask_zero=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dense(len(set(tags)), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Precision(), Recall(), 'accuracy'])
```

# Model Training
```json
history = model.fit(training, output_encoded, epochs=20, batch_size=10, validation_split=0.1)
```

# Plot Results
```json
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
```

# 💾 Save Model and Tokenizer
```json
import pickle
from mlem.api import save

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

save(model, "models/tf")
save(encoder, "encoder/tf")
```

# 🧬 Process Overview

- Airflow DAG orchestrates model training.

- Trained model & tokenizer uploaded to S3.

- FastAPI loads them from S3 and serves inference.

# 🛠️ AWS Infrastructure (via Terraform)

#### ECS Cluster
```json
resource "aws_ecs_cluster" "ml_cluster" {
  name = "ml-cluster"
}
```

# Networking

- VPC
- Public/Private Subnets
- NAT Gateway
- Internet Gateway
- Security Groups
- Load Balancer

# ECR Repositories
```json
resource "aws_ecr_repository" "ml-on-aws" {
  name = "ml-model-deploy-repo"
}
```

# S3 Buckets
```json
resource "aws_s3_bucket" "model_bucket" {
  bucket = "modelbucket"
}

resource "aws_s3_bucket" "dataset_bucket" {
  bucket = "datasetbucket"
}
```

# 📦 Airflow Dockerfile
```json
FROM puckel/docker-airflow
WORKDIR /airflow
COPY dags/* /usr/local/airflow/dags/
COPY requirements.txt /airflow
RUN pip3 install boto3
```

# 🐳 Training Dockerfile
```json
FROM python:3.9.17-slim
WORKDIR /ml-pipeline
COPY . .
RUN apt-get update && apt-get install libgomp1
RUN pip install -r requirements.txt
ENV MODEL_S3_BUCKET_NAME=modelbucket
ENV DATA_S3_BUCKET_NAME=datasetbucket
CMD ["python", "main.py"]
```

# 🪄 ECS DAG Task (Airflow ECSOperator)
```json
from airflow import DAG
from airflow.contrib.operators.ecs_operator import ECSOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id="ecs_fargate_dag_1",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False
) as dag:
    ecs_operator_task = ECSOperator(
        task_id="ecs_operator_task",
        cluster="ml-cluster",
        task_definition="ml-training-task-def",
        launch_type="FARGATE",
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ["subnet-..."],
                "securityGroups": ["sg-..."],
                "assignPublicIp": "ENABLED"
            }
        }
    )
```

# 🚀 Deployment with GitHub Actions
```json
name: Deploy Infra for ML Pipeline

on:
  workflow_dispatch:

jobs:
  run_terraform:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v2
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v1
        with:
          cli_config_credentials_token: ${{ secrets.TF_API_TOKEN }}
      - name: Terraform Init & Apply
        run: |
          cd infrastructure
          terraform init -input=false
          terraform apply --auto-approve -input=false
```

# 🛰️ Inference API (FastAPI)

Docker image is deployed to ECS using AWS Copilot.
```json
copilot init
copilot env init
copilot env deploy --name prod
copilot svc deploy
```

# 📬 Inference Demo
#### Endpoint:
```json
http://<load_balancer_dns>/parseinputs
```
#### Sample Request:
```json
{
  "text": "I am Nho Luong"
}
```
#### Sample Response:
```json
{
  "sentiment": "negative"
}
```

# 📈 Improvements

- Use SageMaker Pipelines for training
- Deploy REST API to EKS
- Add monitoring, CI/CD, autoscaling

This concludes the end-to-end process to build a training pipeline using Airflow, train an LSTM-based RNN, and deploy the model to AWS ECS Fargate for inference. This is a solid starting point to build production-ready AI pipelines.

![](https://i.imgur.com/waxVImv.png)
# I'm are always open to your feedback🚀
# **[Contact Me🇻]**
* [Name: Nho Luong]
* [Telegram](+84983630781)
* [WhatsApp](+84983630781)
* [PayPal.Me](https://www.paypal.com/paypalme/nholuongut)
* [Linkedin](https://www.linkedin.com/in/nholuong/)

![](https://i.imgur.com/waxVImv.png)
![](Donate.png)
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/nholuong)

# License🇻
* Nho Luong (c). All Rights Reserved.🌟
