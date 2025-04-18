
variable "subnet_ids" {
    default = []
}

variable "task_sg_id" {
    default = ""
}

variable "target_grp_arn" {
    default = ""
}

variable "app_count" {
  default = 1
}

variable "task_role_arn" {
  default = ""
}

variable "task_assume_role_arn" {
  default = ""
}

variable "model_bucket" {
  default = ""
}

variable "dataset_bucket" {
  default = ""
}