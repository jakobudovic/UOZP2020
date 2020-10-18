# flatten an array that you don't know the dimensions of
def flatten_rec(arr, new_arr):
    if len(arr) == 1:
        new_arr.append(arr)
    else:
        for l in arr:
            flatten_rec(l, new_arr)


arr = [[['Branka'], [['Edo'], ['Polona']]], [['Helena'], ['Zala']]]
arr1 = []
flatten_rec(arr, arr1)
print(arr1)