#!/bin/bash
sed -n '1~'$2'p' $1> $1.reduced
