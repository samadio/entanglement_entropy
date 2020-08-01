#!/bin/bash

sed -i '1s/^/results=' resullts_L*
sed -i '1 i\from numpy import array' resullts_L*
