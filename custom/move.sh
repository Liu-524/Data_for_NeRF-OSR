#!/bin/bash
echo $1
mkdir $1/train
mkdir $1/train/rgb
mkdir $1/validation
mkdir $1/validation/rgb
mkdir $1/test
mkdir $1/test/rgb
cp $1/rgb/* $1/train/rgb/
ls $1/rgb | grep .*0\.png | xargs -I {} cp $1/rgb/{} $1/test/rgb/
ls $1/rgb | grep .*0\.png | xargs -I {} cp $1/rgb/{} $1/validation/rgb/