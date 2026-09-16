#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  exec python3 app.py
fi

echo "ERRO: python3 não foi encontrado."
echo "Instale Python 3."
read -r -p "Enter para fechar..."
