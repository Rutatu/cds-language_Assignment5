#!/usr/bin/env bash

VENVNAME=lda
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME