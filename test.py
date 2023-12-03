import numpy as np

model = 'lda'
website = 'all'
airline = 'Lufthansa'

for model in ['lda', 'bertopic']:
    data = np.load(f'output/{model}_coherence_{website}_{airline}.npy')
    np.savetxt(f'output/{model}_coherence_{website}_{airline}.csv',data, delimiter=',')
    print(data)