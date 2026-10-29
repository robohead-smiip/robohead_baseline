#!/bin/bash

while [ "$#" -gt 0 ]; do
  [[ $1 =~ ^--[a-z][a-z0-9-]*$ ]] || { echo "Invalid option: $1" >&2; return 1; }
  option_name=${1#--}
  option_name=${option_name//-/_}
  declare -p "$option_name" &>/dev/null || { echo "Unknown option: $1" >&2; return 1; }
  [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; return 1; }
  printf -v "$option_name" '%s' "$2"
  shift 2
done
