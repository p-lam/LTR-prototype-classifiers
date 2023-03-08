# tmp = torch.linalg.norm(prototypes, ord=2, dim=1).detach().cpu().numpy()
# max_val, mid_val, min_val = tmp.max(), tmp.mean(), tmp.min()
# c = min(1/mid_val, mid_val)
# y_range = [min_val-c, max_val+c]
# fig = plt.figure(figsize=(15,3), dpi=64, facecolor='w', edgecolor='k')
# ax1 = fig.add_subplot(111)

# ax1.set_ylabel('norm', fontsize=16)
# ax1.set_ylim(y_range)

# plt.plot(tmp, linewidth=2)
# plt.title('norms of per-class weights from centroids vs. class cardinality', fontsize=20)
# plt.show()