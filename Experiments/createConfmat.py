from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

cm = [ [75.0, 25.0], [27.0, 73.0] ]
cm = np.array( cm )

# Classes
classes = ['False', 'True']

figure, ax = plot_confusion_matrix(conf_mat = cm,
                                   class_names = classes,
                                   show_absolute = False,
                                   show_normed = True,
                                   colorbar = True)

plt.savefig('../Confmat/confmatrixRFMIrawdata.png', format='png', dpi=300, bbox_inches = 'tight')
#plt.show()
