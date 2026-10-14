#!/bin/sh

hypoDD hypoDD_kim2011.inp > hypoDD.sum
rm -rf reloc/*
mv *.reloc.0* reloc
cat hypoDD.sum
