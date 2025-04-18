

resource "aws_ecr_repository" "ml_model_deploy" {
  name = "ml-model-deploy-repo"
}
resource "aws_ecr_repository" "ml_inference_api" {
  name = "ml_inference_api"
}

resource "aws_ecr_repository" "ml_training_task_repo" {
  name = "ml-training-1"
}

resource "aws_cloudwatch_log_group" "airflow-logs" {
  name = "/airflow-logs"  

  retention_in_days = 30  
}

resource "aws_cloudwatch_log_group" "train-task-logs" {
  name = "/train-task-logs"  

  retention_in_days = 30  
}

resource "aws_cloudwatch_log_group" "train-task-container-logs" {
  name = "/train-task-container-logs"  

  retention_in_days = 30  
}

resource "aws_ecs_cluster" "ml_cluster" {
  name = "ml-cluster"
}

resource "aws_s3_bucket" "model_bucket" {
  bucket = var.model_bucket  
  acl    = "private"             
  
  tags = {
    Name        = "model_bucket"
  }
}

resource "aws_s3_bucket" "dataset_bucket" {
  bucket = var.dataset_bucket  
  acl    = "private"             
  
  tags = {
    Name        = "dataset_bucket"
  }
}

# get own account number
data "aws_caller_identity" "current" {}

resource "aws_ecs_task_definition" "airflow_task_def" {
  family                   = "airflow-task-def"
  network_mode             = "awsvpc"
  execution_role_arn       = var.task_role_arn
  task_role_arn = var.task_assume_role_arn
  requires_compatibilities = ["FARGATE"]
  cpu                      = 2048
  memory                   = 4096

  container_definitions = <<DEFINITION
[
  {
    "image": "${data.aws_caller_identity.current.account_id}.dkr.ecr.us-east-1.amazonaws.com/ml-model-deploy-repo:latest",
    "cpu": 2048,
    "memory": 4096,
    "name": "airflow-task",
    "networkMode": "awsvpc",
    "portMappings": [
      {
        "containerPort": 8080,
        "hostPort": 8080
      }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/airflow-logs",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }
]
DEFINITION
}

resource "aws_ecs_task_definition" "ecs_operatore_task_1" {
  family                   = "ecs-op-task-1"
  network_mode             = "awsvpc"
  execution_role_arn       = var.task_role_arn
  task_role_arn = var.task_assume_role_arn
  requires_compatibilities = ["FARGATE"]
  cpu                      = 2048
  memory                   = 4096
  container_definitions = <<DEFINITION
[
  {
    "image": "${data.aws_caller_identity.current.account_id}.dkr.ecr.us-east-1.amazonaws.com/ml-training-1:latest",
    "cpu": 2048,
    "memory": 4096,
    "name": "ecs-ttst-container-1",
    "networkMode": "awsvpc",
    "portMappings": [
      {
        "containerPort": 8080,
        "hostPort": 8080
      }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/train-task-container-logs",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }
]
DEFINITION
}


resource "aws_ecs_service" "airflow_service" {
  name            = "airflow-service"
  cluster         = aws_ecs_cluster.ml_cluster.id
  task_definition = aws_ecs_task_definition.airflow_task_def.arn
  desired_count   = var.app_count
  launch_type     = "FARGATE"

  network_configuration {
    security_groups = [var.task_sg_id]
    subnets         = var.subnet_ids
  }

  load_balancer {
    target_group_arn = var.target_grp_arn
    container_name   = "airflow-task"
    container_port   = 8080
  }
}

resource "aws_ecs_service" "ecs_service_1" {
  name            = "ecs-ttst-service-1"
  cluster         = aws_ecs_cluster.ml_cluster.id
  task_definition = aws_ecs_task_definition.ecs_operatore_task_1.arn
  desired_count   = 0
  launch_type     = "FARGATE"

  network_configuration {
    security_groups = [var.task_sg_id]
    subnets         = var.subnet_ids
  }
}





