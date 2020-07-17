#!/bin/bash

sed -i '2d' draft_insert.py
sed -i "2i seed=33"  draft_insert.py


python draft_insert.py