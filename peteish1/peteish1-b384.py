import csv
import matplotlib.pyplot as plt

with open('peteish1/peteish1.csv') as f:
    reader = csv.DictReader(f)
    data1 = list(reader)
    data1 = [x for x in data1 if float(x['throughput/total_tokens']) <= 2e11]
with open('peteish1/peteish1-b384.csv') as f:
    reader = csv.DictReader(f)
    data2 = list(reader)
    data2 = [x for x in data2 if float(x['throughput/total_tokens']) <= 2e11]

fig = plt.figure(figsize=(6, 4))
plt.plot([float(x['throughput/total_tokens']) for x in data1], [float(x['eval/downstream/hellaswag_len_norm']) for x in data1], label='peteish1')
plt.plot([float(x['throughput/total_tokens']) for x in data2], [float(x['eval/downstream/hellaswag_len_norm']) for x in data2], label='peteish1-b384')
# save plot
plt.xlabel('total_tokens')
plt.ylabel('hellaswag_len_norm')
plt.legend()
plt.savefig('peteish1/peteish1-b384.png', dpi=300, bbox_inches='tight')
