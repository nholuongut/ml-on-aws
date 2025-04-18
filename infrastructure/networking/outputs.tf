
output "ml_vpc_id" {
    value = aws_vpc.mlvpc.id
}

output "ml_private_subnet_id" {
    value = aws_subnet.ml_private.*.id
}

output "ml_task_sg_id" {
    value = aws_security_group.ml_ecs_task_sg.id
}

output "ml_target_grp_arn" {
    value = aws_lb_target_group.ml_lb_tg.id
}

output "load_balancer_ip" {
  value = aws_lb.ml_lb.dns_name
}