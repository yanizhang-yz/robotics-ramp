import math

logits = [2.0, 1.0, 0.1]

# Step 1: exponentiate each logit
exp_logits = [math.exp(z) for z in logits]
print("Exponentiated:", exp_logits)

# Step 2: sum them
total = sum(exp_logits)
print("Total Sum:", total)

# Step 3: divide each by the sum
probabilities = [e / total for e in exp_logits]
print("Probabilities:", probabilities)
print("Sum of the probabilities", sum(probabilities))