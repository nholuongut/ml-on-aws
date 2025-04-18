

output "task_role_arn" {
    value = aws_iam_role.ecs_task_execution_role.arn
}

output "task_assume_role_arn" {
    value = aws_iam_role.ecs_task_assume_role.arn
}