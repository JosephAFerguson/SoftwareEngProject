#!/bin/bash

for (( i=0; i<${1}; i++)); do
    address="$((1000 + i)) test drive"
    price=$(shuf -i 50-200 -n 1)
    price=$((price*10))
    sqft=$(shuf -i 220-800 -n 1)
    room=$(shuf -i 0-3 -n 1)
    bed=$((room+1))
    bath=$(shuf -i 1-3 -n 1)

    curl -X POST http://localhost:3000/api/v1/rental/post \
      -H "Content-Type: application/json" \
      -d "{\"address\":\"$address\", \"price\":$price, \"sqft\":$sqft,
           \"roommates\":$room, \"bednum\":$bed, \"bathnum\":$bath
          }"
done

