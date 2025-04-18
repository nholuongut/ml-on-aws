terraform {
  required_version = ">= 0.14"
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = ">= 4.9.0"
    }

    tls = {
      source = "hashicorp/tls"
      version = "3.4.0"
    }
  }

  cloud {
    organization = ""

    workspaces {
      name = ""
    }
  }

  
}

provider "aws" {
  region = "us-east-1"
}


module "ecs_module" {
    source = "./ecs_module"
    subnet_ids = module.networking_module.ml_private_subnet_id
    task_sg_id = module.networking_module.ml_task_sg_id
    target_grp_arn = module.networking_module.ml_target_grp_arn
    task_role_arn = module.security_module.task_role_arn
    task_assume_role_arn = module.security_module.task_assume_role_arn
    depends_on = [ module.networking_module ]
}

module "networking_module" {
    source = "./networking"
}

module "security_module" {
    source = "./security"
}