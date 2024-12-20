#!/bin/bash

echo "Exit Code: $1"

if [[ $1 -ne 0 ]]
then
  exit 1
fi