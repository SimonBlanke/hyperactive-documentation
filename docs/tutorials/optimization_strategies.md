# Optimization Strategies


## Custom Optimization Strategy


### How it works

Optimization strategies are designed to automatically pass useful data from one optimization algorithm to the next:

- The best parameter found in one optimization run ist automatically passed to the next. 
- The search-data if passed as memory-warm-start to all consecutive optimization runs.
- If an algorithm accepts warm-start-smbo as a parameter the search-data is also automatically passed.

Without optimization strategies the steps above can be manually done, but by chaining together the algorithms into strategies it is automatically done for you.

