terraform {
  required_version = ">= 1.9, < 2.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.0" }
  }
  backend "local" {}
}

variable "aws_account_id" {
  type = string
  validation {
    condition     = can(regex("^[0-9]{12}$", var.aws_account_id))
    error_message = "Supply the intended AWS account ID."
  }
}
variable "admin_cidr" {
  type = string
  validation {
    condition     = can(cidrnetmask(var.admin_cidr)) && endswith(var.admin_cidr, "/32")
    error_message = "Only the benchmark controller's single IPv4 /32 may use SSH."
  }
}
variable "ssh_public_key_path" { type = string }
variable "max_hours" {
  type    = number
  default = 24
  validation {
    condition     = var.max_hours >= 1 && var.max_hours <= 48
    error_message = "Use a finite lifetime between one and 48 hours."
  }
}

provider "aws" {
  region              = "us-east-1"
  profile             = "afri-research"
  allowed_account_ids = [var.aws_account_id]
  default_tags { tags = { Project = "afri-asr-compression", ManagedBy = "terraform" } }
}
data "aws_availability_zones" "available" { state = "available" }
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}
resource "aws_vpc" "main" {
  cidr_block           = "10.31.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
}
resource "aws_internet_gateway" "main" { vpc_id = aws_vpc.main.id }
resource "aws_subnet" "main" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.31.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true
}
resource "aws_route_table" "main" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}
resource "aws_route_table_association" "main" {
  subnet_id      = aws_subnet.main.id
  route_table_id = aws_route_table.main.id
}
resource "aws_security_group" "node" {
  name_prefix = "afri-asr-"
  vpc_id      = aws_vpc.main.id
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
resource "aws_key_pair" "node" {
  key_name_prefix = "afri-asr-"
  public_key      = file(pathexpand(var.ssh_public_key_path))
}
resource "aws_instance" "cpu" {
  ami                                  = data.aws_ami.ubuntu.id
  instance_type                        = "m7i-flex.large"
  subnet_id                            = aws_subnet.main.id
  vpc_security_group_ids               = [aws_security_group.node.id]
  key_name                             = aws_key_pair.node.key_name
  instance_initiated_shutdown_behavior = "terminate"
  user_data_replace_on_change          = true
  user_data                            = <<-CLOUD
    #cloud-config
    bootcmd:
      - [systemd-run, --unit=afri-benchmark-expiry, --on-active=${ceil(var.max_hours * 3600)}s, /usr/sbin/shutdown, -h, now]
    package_update: true
    packages: [git, cmake, build-essential, python3-venv, libsndfile1, libgomp1, rsync, ca-certificates, openssh-server, iptables]
    runcmd:
      - [iptables, -I, INPUT, -s, ${var.admin_cidr}, -p, tcp, --dport, '22', -j, ACCEPT]
      - [systemctl, enable, --now, ssh.service]
  CLOUD
  root_block_device {
    volume_type           = "gp3"
    volume_size           = 30
    encrypted             = true
    delete_on_termination = true
  }
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }
  tags = { Name = "afri-asr-compression-bounded" }
}
output "instance_id" { value = aws_instance.cpu.id }
output "public_ip" { value = aws_instance.cpu.public_ip }
output "ami" { value = aws_instance.cpu.ami }
output "vpc_id" { value = aws_vpc.main.id }
output "key_name" { value = aws_key_pair.node.key_name }
