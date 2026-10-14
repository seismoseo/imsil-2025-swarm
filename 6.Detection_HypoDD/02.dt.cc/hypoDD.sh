#!/bin/sh

hypoDD hypoDD.inp > hypoDD.sum
rm -rf reloc/*
mv *.reloc.0* reloc
cat hypoDD.sum
