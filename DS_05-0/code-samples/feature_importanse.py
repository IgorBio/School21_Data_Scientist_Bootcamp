import matplotlib.pyplot as plt


def f_importances(weights, names, top=-1):
    weights, names = zip(*sorted(list(zip(weights, names))))

    # Show all features
    if top == -1:
        top = len(names)
    plt.figure(figsize=(12, 5))
    plt.barh(range(top), weights[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.show()
