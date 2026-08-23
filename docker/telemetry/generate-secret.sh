#!/bin/bash
# 生成一个 32 字符的强密码（字母数字 + 特殊符号）
openssl rand -base64 32 | tr -d '\n' | cut -c1-32
