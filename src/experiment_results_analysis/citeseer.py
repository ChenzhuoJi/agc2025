import joblib
data = joblib.load(r'data\intermediate\citeseer\p1_theta0_1.pkl')

print('加载成功')

print(data.keys())
sim = data['similarity_matrix']
ho = data['high_order_matrix']
targets = data['targets']
print(sim.shape)
print(ho.shape)
print(targets[:5])
