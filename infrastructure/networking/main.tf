

data "aws_availability_zones" "available_zones" {
  state = "available"
}

resource "aws_vpc" "mlvpc" {
  cidr_block = "10.1.0.0/16"
  enable_dns_hostnames = true

  tags = {
    Name = "mlvpc"
  }
}

resource "aws_internet_gateway" "ml_gw" {
  vpc_id = resource.aws_vpc.mlvpc.id

  tags = {
    Name = "ml_gw"
  }

}



resource "aws_route_table" "ml_rt_public" {
  vpc_id = resource.aws_vpc.mlvpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = resource.aws_internet_gateway.ml_gw.id
  }

  tags = {
    Name = "ml_rt_public"
  }
}

resource "aws_subnet" "ml_public" {
  count                   = 2
  cidr_block              = cidrsubnet(aws_vpc.mlvpc.cidr_block, 8, 2 + count.index)
  availability_zone       = data.aws_availability_zones.available_zones.names[count.index]
  vpc_id                  = aws_vpc.mlvpc.id
  map_public_ip_on_launch = true
  tags = {
    Name = "ml_public_subnet"
  }
}

resource "aws_subnet" "ml_private" {
  count             = 2
  cidr_block        = cidrsubnet(aws_vpc.mlvpc.cidr_block, 8, count.index)
  availability_zone = data.aws_availability_zones.available_zones.names[count.index]
  vpc_id            = aws_vpc.mlvpc.id
  tags = {
    Name = "ml_private_subnet"
  }
}

resource "aws_route_table_association" "ml_public_subnet_assoc" {
  count          = 2
  subnet_id      = element(aws_subnet.ml_public.*.id, count.index)
  route_table_id = element(aws_route_table.ml_rt_public.*.id, count.index)
}




resource "aws_eip" "ml_gateway" {
  count      = 2
  vpc        = true
  depends_on = [aws_internet_gateway.ml_gw]
}

resource "aws_nat_gateway" "ml_nat_gateway" {
  count         = 2
  subnet_id     = element(aws_subnet.ml_public.*.id, count.index)
  allocation_id = element(aws_eip.ml_gateway.*.id, count.index)
}

resource "aws_route_table" "ml_private" {
  count  = 2
  vpc_id = aws_vpc.mlvpc.id

  route {
    cidr_block = "0.0.0.0/0"
    nat_gateway_id = element(aws_nat_gateway.ml_nat_gateway.*.id, count.index)
  }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = element(aws_subnet.ml_private.*.id, count.index)
  route_table_id = element(aws_route_table.ml_private.*.id, count.index)
}


resource "aws_security_group" "ml_lb_sg" {
  name        = "ml_lb_sg"
  description = "sg for ml app lb"
  vpc_id      = resource.aws_vpc.mlvpc.id

  ingress {
    protocol   = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    from_port  = 22
    to_port    = 22
  }

  ingress {
    protocol   = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    from_port  = 80
    to_port    = 80
  }

  ingress {
    protocol   = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    from_port  = 8080
    to_port    = 8080
  }

  ingress {
    protocol   = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    from_port  = 443
    to_port    = 443
  }

  ingress {
    protocol   = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    from_port  = 3000
    to_port    = 3000
  }

  ingress {
    protocol   = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    from_port  = 5000
    to_port    = 5000
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name = "ml_lb_sg"
  }
}

resource "aws_lb" "ml_lb" {
  name            = "ml-lb"
  subnets         = aws_subnet.ml_public.*.id
  security_groups = [aws_security_group.ml_lb_sg.id]
}

resource "aws_lb_target_group" "ml_lb_tg" {
  name        = "ml-lb-target-group"
  port        = 80
  protocol    = "HTTP"
  vpc_id      = aws_vpc.mlvpc.id
  target_type = "ip"
  health_check {
    enabled             = true
    interval            = 30
    path                = "/"
    port                = 8080
    protocol            = "HTTP"
    healthy_threshold   = 3
    unhealthy_threshold = 3
    timeout             = 5
    matcher             = "200,302"
  }
}

resource "aws_lb_listener" "ml_lb_listener" {
  load_balancer_arn = aws_lb.ml_lb.id
  port              = "80"
  protocol          = "HTTP"

  default_action {
    target_group_arn = aws_lb_target_group.ml_lb_tg.id
    type             = "forward"
  }
}


resource "aws_security_group" "ml_ecs_task_sg" {
  name        = "ml_ecs_task_sg"
  vpc_id      = aws_vpc.mlvpc.id

  ingress {
    protocol        = "tcp"
    from_port       = 3000
    to_port         = 3000
    security_groups = [aws_security_group.ml_lb_sg.id]
  }
  ingress {
    protocol        = "tcp"
    from_port       = 5000
    to_port         = 5000
    security_groups = [aws_security_group.ml_lb_sg.id]
  }
  ingress {
    protocol        = "tcp"
    from_port       = 80
    to_port         = 80
    security_groups = [aws_security_group.ml_lb_sg.id]
  }
  ingress {
    protocol        = "tcp"
    from_port       = 8080
    to_port         = 8080
    security_groups = [aws_security_group.ml_lb_sg.id]
  }


  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }
}




