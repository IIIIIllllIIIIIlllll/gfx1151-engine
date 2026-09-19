#!/usr/bin/env bash
# 兼容包装:API 服务器与 CLI 工具现由根目录 build.sh 统一编译,产物在 build/。
exec bash "$(dirname "$0")/../build.sh" api
